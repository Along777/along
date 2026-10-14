"""FastAPI server the porch talks to (see README "Porch <-> brain API").

Startup returns immediately; models load and warm in the background while
/health reports warming=true and /turn answers 503, so the porch can start first.
"""
from __future__ import annotations

import asyncio
import json
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from . import __version__
from .audiofx import VoiceFx, from_wav, to_wav
from .config import BRAIN_DIR, Config
from .fillers import FillerBank, bank_version, load_lines
from .ollama import OllamaClient
from .pipeline import Brain, FakeLlm, TurnMeta
from .stt import FakeStt, ParakeetOnnx
from .tts import FakeTts, KokoroOnnx

TEST_IMAGE = BRAIN_DIR.parent / "img" / "skeleton-porch.jpg"


def gpu_now() -> dict | None:
    """One nvidia-smi reading for the dashboard; None without an NVIDIA GPU."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        name, util, used, total, temp = [s.strip() for s in out.splitlines()[0].split(",")]
        return {"name": name, "util_pct": int(float(util)), "mem_used_mb": int(float(used)),
                "mem_total_mb": int(float(total)), "temp_c": int(float(temp))}
    except Exception:
        return None


def build_engines(cfg: Config, fake: bool):
    if fake or cfg.stt.engine == "fake":
        stt = FakeStt()
    else:
        stt = ParakeetOnnx(cfg.stt.model, cfg.path(cfg.stt.model_dir), cfg.stt.quantization)
    if fake or cfg.tts.engine == "fake":
        tts = FakeTts()
    else:
        tts = KokoroOnnx(cfg.path(cfg.tts.model), cfg.path(cfg.tts.voices), cfg.tts.voice, cfg.tts.lang, cfg.tts.speed)
    fx = VoiceFx(cfg.fx, pitch=isinstance(tts, KokoroOnnx))
    return stt, tts, fx


class State:
    def __init__(self, cfg: Config, fake: bool, log=print, cache_root: Path | None = None):
        self.cfg, self.fake, self.log = cfg, fake, log
        self.ready = False
        self.warming = True
        self.error: str | None = None
        self.started = time.time()
        self.lock = asyncio.Lock()
        self.brain: Brain | None = None
        self.ollama: OllamaClient | None = None
        self.ollama_version: str | None = None
        self.bank = FillerBank(cache_root or BRAIN_DIR / "cache" / "fillers")
        self._gpu: tuple[float, dict | None] = (0.0, None)

    async def gpu(self) -> dict | None:
        t, g = self._gpu
        if time.time() - t > 5:  # nvidia-smi takes ~50 ms; the porch polls every few seconds
            g = await asyncio.to_thread(gpu_now)
            self._gpu = (time.time(), g)
        return g

    async def boot(self) -> None:
        try:
            cfg = self.cfg
            persona_dir = cfg.path(cfg.persona.dir)
            lines, lines_text = load_lines(persona_dir)
            stt, tts, fx = await asyncio.to_thread(build_engines, cfg, self.fake)
            if self.fake:
                llm = FakeLlm()
            else:
                self.ollama = llm = OllamaClient(cfg.ollama)
                for _ in range(30):  # after a reboot Ollama may come up after us
                    self.ollama_version = await llm.version()
                    if self.ollama_version:
                        break
                    await asyncio.sleep(2)
                if not self.ollama_version:
                    raise RuntimeError(f"Ollama is not answering at {cfg.ollama.url} - start it, then restart the brain")
                for m in {cfg.ollama.vision_model, cfg.ollama.reply_model}:
                    caps = await llm.show(m)
                    if not caps.present:
                        raise RuntimeError(f"model {m} is not pulled - run: ollama pull {m}")
                if not llm.caps[cfg.ollama.vision_model].vision:
                    raise RuntimeError(f"{cfg.ollama.vision_model} has no vision capability")
            self.brain = Brain(cfg, llm, stt, tts, fx, persona_dir, lines.get("deflect", []))
            await self.warm(first=True)
            version = bank_version(lines_text, tts.name, cfg.fx)
            await asyncio.to_thread(self.bank.load_or_render, version, lines, self.brain.speak_sync, self.log)
            self.ready = True
            self.log(f"brain ready ({'FAKE engines' if self.fake else 'stt=' + stt.name + ' tts=' + tts.name})")
        except Exception as e:  # keep serving /health so the porch and doctor can see why
            self.error = f"{type(e).__name__}: {e}"
            self.log("BOOT FAILED: " + self.error)
            traceback.print_exc()
        finally:
            self.warming = False

    async def warm(self, first: bool = False) -> None:
        b = self.brain
        assert b is not None
        t = time.perf_counter()
        await b.transcribe(np.zeros(16000, np.float32), 16000)
        await asyncio.to_thread(b.tts.synth, "Ready.")
        if not self.fake:
            cfg = self.cfg.ollama
            timeout = 300 if first else 120  # a cold first load can take a while
            if TEST_IMAGE.is_file():
                await b.llm.describe(TEST_IMAGE.read_bytes(), b.vision_prompt, timeout=timeout)
            if cfg.reply_model != cfg.vision_model or not TEST_IMAGE.is_file():
                await b.llm.reply(b.persona, "Costumes: unclear\nThey said nothing yet.", timeout=timeout)
        self.log(f"warm in {time.perf_counter() - t:.1f}s")

    async def rewarm_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.ollama.warm_interval_s)
            if not (self.ready and self.ollama) or self.lock.locked():
                continue
            loaded = await self.ollama.loaded()
            names = {m.get("name") for m in loaded} | {m.get("model") for m in loaded}
            if self.cfg.ollama.reply_model not in names:
                self.log("model was evicted - warming it again")
                try:
                    await self.warm()
                except Exception as e:
                    self.log(f"re-warm failed: {e}")

    async def health(self) -> dict:
        out: dict = {
            "ok": self.error is None,
            "ready": self.ready,
            "warming": self.warming,
            "error": self.error,
            "version": __version__,
            "fake": self.fake,
            "uptime_s": round(time.time() - self.started),
            "fillers_version": self.bank.version if self.ready else None,
        }
        if self.brain:
            out["engines"] = {"vlm": self.cfg.ollama.vision_model if not self.fake else "fake",
                              "llm": self.cfg.ollama.reply_model if not self.fake else "fake",
                              "stt": self.brain.stt.name, "tts": self.brain.tts.name}
            out["stats"] = self.brain.stats()
        if not self.fake:
            out["gpu"] = await self.gpu()
        if self.ollama:
            loaded = await self.ollama.loaded()
            out["ollama"] = {"version": self.ollama_version, "loaded": [
                {"name": m.get("name"), "size_mb": round((m.get("size") or 0) / 2**20),
                 "vram_mb": round((m.get("size_vram") or 0) / 2**20)} for m in loaded]}
        return out


def create_app(cfg: Config, fake: bool = False, cache_root: Path | None = None) -> FastAPI:
    state = State(cfg, fake, cache_root=cache_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        boot = asyncio.create_task(state.boot())
        rewarm = asyncio.create_task(state.rewarm_loop())
        app.state.boot_task = boot
        yield
        rewarm.cancel()
        boot.cancel()
        if state.ollama:
            await state.ollama.close()

    app = FastAPI(title="skeleton-brain", version=__version__, lifespan=lifespan)
    app.state.skeleton = state

    def not_ready():
        if state.error:
            return JSONResponse({"error": "boot_failed", "detail": state.error}, status_code=503)
        return JSONResponse({"error": "warming"}, status_code=503, headers={"Retry-After": "5"})

    @app.get("/health")
    async def health():
        return await state.health()

    @app.post("/turn")
    async def turn(meta: str = Form("{}"), photo: UploadFile | None = File(None), clip: UploadFile | None = File(None)):
        if not state.ready or state.brain is None:
            return not_ready()
        if state.lock.locked():
            return JSONResponse({"error": "busy"}, status_code=503, headers={"Retry-After": "1"})
        async with state.lock:
            try:
                m = TurnMeta.from_dict(json.loads(meta or "{}"))
            except (ValueError, TypeError):
                return JSONResponse({"error": "bad_meta"}, status_code=400)
            photo_b = await photo.read() if photo else None
            clip_b = await clip.read() if clip else None
            if photo_b is not None:
                if len(photo_b) > cfg.limits.max_photo_bytes:
                    return JSONResponse({"error": "too_large", "detail": "photo"}, status_code=400)
                if photo_b[:2] != b"\xff\xd8":
                    return JSONResponse({"error": "bad_photo"}, status_code=400)
            if clip_b is not None:
                try:
                    x, sr = from_wav(clip_b)
                except ValueError as e:
                    return JSONResponse({"error": "bad_clip", "detail": str(e)}, status_code=400)
                if len(x) / sr > cfg.limits.max_clip_seconds:
                    return JSONResponse({"error": "too_large", "detail": "clip"}, status_code=400)
            try:
                return await state.brain.run_turn(m, photo_b or None, clip_b or None)
            except Exception as e:
                traceback.print_exc()
                return JSONResponse({"error": "internal", "detail": f"{type(e).__name__}: {e}"}, status_code=500)

    @app.post("/warm")
    async def warm(photo: UploadFile = File(...)):
        """Run the vision model once on a frame of the porch's real size. The first image at a new
        size is several seconds slower (one-time setup in the model); this keeps that off a real turn."""
        if not state.ready or state.brain is None:
            return not_ready()
        if state.lock.locked():
            return JSONResponse({"error": "busy"}, status_code=503)
        data = await photo.read()
        if not data or len(data) > cfg.limits.max_photo_bytes or data[:2] != b"\xff\xd8":
            return JSONResponse({"error": "bad_photo"}, status_code=400)
        async with state.lock:
            t = time.perf_counter()
            await state.brain.describe(data)
        return {"ok": True, "vision_ms": round((time.perf_counter() - t) * 1000)}

    @app.get("/fillers")
    async def fillers():
        if not state.ready:
            return not_ready()
        return state.bank.manifest

    @app.get("/fillers/{fid}.wav")
    async def filler_wav(fid: str):
        p = state.bank.path_for(fid)
        if p is None:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return FileResponse(p, media_type="audio/wav")

    @app.post("/speak")
    async def speak(body: dict):
        if not state.ready or state.brain is None:
            return not_ready()
        text = str(body.get("text", "")).strip()[:300]
        if not text:
            return JSONResponse({"error": "empty"}, status_code=400)
        audio, sr = await state.brain.speak(text, fx=bool(body.get("fx", True)))
        return Response(to_wav(audio, sr), media_type="audio/wav")

    return app
