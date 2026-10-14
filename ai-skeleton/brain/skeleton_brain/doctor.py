"""`skeleton-brain doctor`: prove every stage works on this machine, with timings.

Prints ok/warn/fail lines like the local_llm setup script; exits non-zero on any
fail so it can gate a setup script.
"""
from __future__ import annotations

import asyncio
import platform
import shutil
import statistics
import subprocess
import sys
import time

from .audiofx import resample, to_wav
from .config import Config
from .fillers import load_lines
from .ollama import OllamaClient
from .pipeline import Brain, FakeLlm, TurnMeta, build_user_message
from .server import TEST_IMAGE, build_engines

C = {"ok": "\033[1;32m   ok:\033[0m", "warn": "\033[1;33m warn:\033[0m", "fail": "\033[1;31m fail:\033[0m"}


class Report:
    def __init__(self):
        self.fails = 0
        self.warns = 0

    def say(self, title: str) -> None:
        print(f"\n\033[1;36m== {title}\033[0m")

    def line(self, kind: str, msg: str) -> None:
        if kind == "fail":
            self.fails += 1
        elif kind == "warn":
            self.warns += 1
        print(f"{C[kind]} {msg}")


def gpu_info() -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        name, cc, drv, tot, used, temp = [s.strip() for s in out.splitlines()[0].split(",")]
        return {"name": name, "cc": cc, "driver": drv, "mem_total_mb": int(float(tot)),
                "mem_used_mb": int(float(used)), "temp_c": int(float(temp))}
    except Exception:
        return None


async def run(cfg: Config, fake: bool = False, turns: int = 10) -> int:
    r = Report()
    r.say("Machine")
    print(f"   python {sys.version.split()[0]} on {platform.platform()}")
    g = gpu_info()
    if g:
        r.line("ok", f"{g['name']} (compute {g['cc']}, driver {g['driver']}, "
                     f"{g['mem_used_mb']}/{g['mem_total_mb']} MB, {g['temp_c']} C)")
        if g["mem_total_mb"] < 7000:
            r.line("warn", "under 8 GB VRAM - use the 2b/4b vision model")
    else:
        r.line("warn", "no NVIDIA GPU visible (fine for laptop tests; the 5090 brain should show one)")

    llm = None
    if not fake:
        r.say("Ollama")
        llm = OllamaClient(cfg.ollama)
        v = await llm.version()
        if not v:
            r.line("fail", f"Ollama not answering at {cfg.ollama.url} (start it: ollama serve, or the service/app)")
            return 1
        r.line("ok", f"Ollama {v}")
        for m in sorted({cfg.ollama.vision_model, cfg.ollama.reply_model}):
            caps = await llm.show(m)
            if not caps.present:
                r.line("fail", f"{m} not pulled - run: ollama pull {m}")
            else:
                r.line("ok", f"{m}: {', '.join(caps.capabilities) or 'no capabilities listed'}")
                if m == cfg.ollama.vision_model and not caps.vision:
                    r.line("fail", f"{m} has no vision")
        if cfg.ollama.vision_model != cfg.ollama.reply_model:
            r.line("warn", "vision and reply models differ - two models must stay loaded "
                           "(check OLLAMA_MAX_LOADED_MODELS is not 1)")
        if r.fails:
            return 1

    r.say("Engines")
    t = time.perf_counter()
    try:
        stt, tts, fx = build_engines(cfg, fake)
    except Exception as e:
        r.line("fail", f"{type(e).__name__}: {e}")
        return 1
    r.line("ok", f"stt={stt.name}  tts={tts.name}  (loaded in {time.perf_counter() - t:.1f}s)")
    lines, _ = load_lines(cfg.path(cfg.persona.dir))
    brain = Brain(cfg, FakeLlm() if fake else llm, stt, tts, fx, cfg.path(cfg.persona.dir), lines.get("deflect", []))

    r.say("Voice + ears")
    phrase = "Trick or treat, I am a pirate."
    t = time.perf_counter()
    raw, sr = tts.synth(phrase)
    t_tts = time.perf_counter() - t
    r.line("ok", f"tts {t_tts * 1000:.0f} ms for {len(raw) / sr:.1f}s of audio")
    t = time.perf_counter()
    heard = await brain.transcribe(resample(raw, sr, 16000), 16000)
    t_stt = time.perf_counter() - t
    if fake:
        r.line("ok", f"stt (fake) -> {heard!r}")
    elif "trick" in heard.lower():
        r.line("ok", f"stt {t_stt * 1000:.0f} ms -> {heard!r}")
    else:
        r.line("fail", f"stt heard {heard!r} for {phrase!r}")

    photo = TEST_IMAGE.read_bytes() if TEST_IMAGE.is_file() else None
    if not fake:
        r.say("Eyes + reply (first call may load the model)")
        t = time.perf_counter()
        warm_ok = True
        try:
            if photo:
                await llm.describe(photo, brain.vision_prompt, timeout=300)
            await llm.reply(brain.persona, "Costumes: unclear\nThey said nothing yet.", timeout=300)
        except Exception as e:
            warm_ok = False
            r.line("fail", f"model warm-up failed: {type(e).__name__}: {e}")
        if warm_ok:
            r.line("ok", f"model warm in {time.perf_counter() - t:.1f}s")
            if photo:
                t = time.perf_counter()
                desc, _ = await llm.describe(photo, brain.vision_prompt, timeout=60)
                r.line("ok", f"describe {1000 * (time.perf_counter() - t):.0f} ms -> {desc!r}")
            else:
                desc = "unclear"
                r.line("warn", f"{TEST_IMAGE} missing - vision not tested")
            t = time.perf_counter()
            text, data = await llm.reply(brain.persona, build_user_message(desc, "trick or treat", TurnMeta(group_size=1)),
                                         timeout=60)
            thinking = (data.get("message") or {}).get("thinking") or ""
            r.line("ok", f"reply {1000 * (time.perf_counter() - t):.0f} ms -> {text!r}")
            if thinking:
                r.line("warn", f"model returned {len(thinking)} chars of thinking - use an -instruct tag")
            else:
                r.line("ok", "no thinking text (instruct model)")
            loaded = await llm.loaded()
            for m in loaded:
                size, vram = m.get("size") or 0, m.get("size_vram") or 0
                pct = round(100 * vram / size) if size else 0
                kind = "ok" if pct >= 99 or not g else "warn"
                where = f"{pct}% GPU" if g else "CPU (no NVIDIA GPU)"
                r.line(kind, f"{m.get('name')} resident: {size / 2**30:.1f} GB, {where}")
            if not loaded:
                r.line("warn", "no model resident after the calls (keep_alive not honoured?)")

    r.say(f"Bench: {turns} full turns (photo + speech)")
    clip = to_wav(resample(raw, sr, 16000), 16000)
    totals, stages = [], {}
    for i in range(turns):
        res = await brain.run_turn(TurnMeta(turn_id=f"bench-{i}", group_size=2, turn_index=0), photo, clip)
        totals.append(res["timings_ms"]["total"])
        for k, v in res["timings_ms"].items():
            stages.setdefault(k, []).append(v)
        if i == 0:
            print(f"   first reply: {res['reply_text']!r} ({res['source']})")
    totals.sort()
    p50 = totals[len(totals) // 2]
    p90 = totals[min(len(totals) - 1, int(len(totals) * 0.9))]
    med = ", ".join(f"{k} {statistics.median(v):.0f}" for k, v in stages.items() if k != "total")
    kind = "ok" if p90 <= 3000 or fake else ("warn" if p90 <= 6000 else "fail")
    r.line(kind, f"brain time p50 {p50} ms, p90 {p90} ms  (median per stage, ms: {med})")
    if llm:
        await llm.close()

    r.say("Summary")
    if r.fails:
        print(f"\033[1;31m {r.fails} fail(s), {r.warns} warning(s)\033[0m")
        return 1
    print(f"\033[1;32m all checks passed\033[0m ({r.warns} warning(s))")
    return 0


def main(cfg: Config, fake: bool, turns: int) -> int:
    return asyncio.run(run(cfg, fake, turns))
