"""One porch turn: (photo, clip) -> costume line + transcript -> reply -> voice.

Vision (GPU, Ollama) and speech-to-text (CPU thread) run in parallel. Every stage
has a timeout with a safe fallback, so a slow stage costs a canned line, never a hang.
"""
from __future__ import annotations

import asyncio
import base64
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audiofx import VoiceFx, from_wav, resample, to_wav
from .config import Config
from .stt import SttEngine
from .textfilter import TextFilter, load_blocklist
from .tts import TtsEngine


@dataclass
class TurnMeta:
    turn_id: str = ""
    group_size: int = 0
    turn_index: int = 0
    costume_hint: str | None = None
    debug: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "TurnMeta":
        return cls(
            turn_id=str(d.get("turn_id", ""))[:64],
            group_size=max(0, min(int(d.get("group_size", 0) or 0), 50)),
            turn_index=max(0, min(int(d.get("turn_index", 0) or 0), 50)),
            costume_hint=(str(d["costume_hint"])[:200] if d.get("costume_hint") else None),
            debug=bool(d.get("debug", False)),
        )


class FakeLlm:
    """Stands in for Ollama in tests and porch development."""

    async def describe(self, jpeg: bytes, prompt: str, *, timeout=None):
        return "1 person: a costume I cannot quite make out", {}

    async def reply(self, system: str, user: str, *, timeout=None):
        if "trick or treat" in user.lower():
            return "Trick or treat indeed. Take some candy, friend, and mind the cobwebs.", {}
        return "What a splendid costume. Happy Halloween, traveler.", {}


def build_user_message(description: str, transcript: str, meta: TurnMeta) -> str:
    if description:
        lines = [f"Costumes: {description}"]
    else:  # no photo this turn (camera off or paused): answer what they said, do not invent a costume
        lines = ["You cannot see them right now. Do not mention or guess a costume; answer what they said."]
    if meta.group_size:
        lines.append(f"People at the door: {meta.group_size}")
    lines.append(f'They said: "{transcript}"' if transcript else "They said nothing yet.")
    if meta.turn_index > 0:
        lines.append("You already greeted them; this is a follow-up. Do not greet again.")
    lines.append("Reply now, in character, in one short sentence.")
    return "\n".join(lines)


class Brain:
    def __init__(self, cfg: Config, llm, stt: SttEngine, tts: TtsEngine, fx: VoiceFx,
                 persona_dir: Path, deflect_lines: list[str]):
        self.cfg, self.llm, self.stt, self.tts, self.fx = cfg, llm, stt, tts, fx
        self.persona = (persona_dir / "persona.txt").read_text(encoding="utf-8").strip()
        self.vision_prompt = (persona_dir / "vision_prompt.txt").read_text(encoding="utf-8").strip()
        self.filter = TextFilter(load_blocklist(persona_dir / "blocklist.txt"), deflect_lines)
        self.fallback_lines = deflect_lines
        self._stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._tts_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self.turn_times_ms: list[float] = []

    # -- stages ---------------------------------------------------------------
    async def describe(self, jpeg: bytes) -> str:
        try:
            text, _ = await asyncio.wait_for(
                self.llm.describe(jpeg, self.vision_prompt, timeout=self.cfg.ollama.describe_timeout_s + 1),
                self.cfg.ollama.describe_timeout_s)
            return text or "unclear"
        except Exception:
            return "unclear"

    async def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if audio.size == 0:
            return ""
        if sr != 16000:
            audio = resample(audio, sr, 16000)
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(loop.run_in_executor(self._stt_pool, self.stt.transcribe, audio, 16000),
                                          self.cfg.stt.timeout_s)
        except Exception:
            return ""

    async def reply(self, description: str, transcript: str, meta: TurnMeta) -> tuple[str, str]:
        try:
            text, _ = await asyncio.wait_for(
                self.llm.reply(self.persona, build_user_message(description, transcript, meta),
                               timeout=self.cfg.ollama.reply_timeout_s + 1),
                self.cfg.ollama.reply_timeout_s)
        except asyncio.TimeoutError:
            return random.choice(self.fallback_lines), "canned_timeout"
        except Exception:
            return random.choice(self.fallback_lines), "canned_error"
        return self.filter.clean(text)

    async def speak(self, text: str, *, fx: bool = True) -> tuple[np.ndarray, int]:
        loop = asyncio.get_running_loop()
        audio, sr = await asyncio.wait_for(loop.run_in_executor(self._tts_pool, self.tts.synth, text),
                                           self.cfg.tts.timeout_s)
        if fx:
            audio = await loop.run_in_executor(self._tts_pool, self.fx, audio, sr)
        return audio, sr

    def speak_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Blocking version for pre-rendering fillers."""
        audio, sr = self.tts.synth(text)
        return self.fx(audio, sr), sr

    # -- the turn -------------------------------------------------------------
    async def run_turn(self, meta: TurnMeta, photo: bytes | None, clip: bytes | None) -> dict:
        t0 = time.perf_counter()
        timings: dict[str, int] = {}

        async def timed(name, coro):
            s = time.perf_counter()
            try:
                return await coro
            finally:
                timings[name] = round((time.perf_counter() - s) * 1000)

        audio_in, sr_in = (from_wav(clip) if clip else (np.zeros(0, np.float32), 16000))
        if photo:
            vision = timed("vision", self.describe(photo))
        else:
            vision = _const(meta.costume_hint or "")  # "" = no photo at all, not an unclear one
        description, transcript = await asyncio.gather(vision, timed("stt", self.transcribe(audio_in, sr_in)))

        reply_text, source = await timed("llm", self.reply(description, transcript, meta))
        try:
            audio, sr = await timed("tts", self.speak(reply_text))
        except Exception:
            reply_text, source = random.choice(self.fallback_lines), "canned_error"
            audio, sr = await timed("tts", self.speak(reply_text))
        wav = to_wav(audio, sr)
        total = round((time.perf_counter() - t0) * 1000)
        timings["total"] = total
        self.turn_times_ms = (self.turn_times_ms + [total])[-200:]
        return {
            "turn_id": meta.turn_id,
            "reply_text": reply_text,
            "source": source,
            "description": description,
            "heard": bool(transcript),
            "transcript": transcript if meta.debug else None,
            "audio": {"format": "wav", "sample_rate": sr, "b64": base64.b64encode(wav).decode("ascii")},
            "duration_s": round(len(audio) / sr, 2),
            "timings_ms": timings,
        }

    def stats(self) -> dict:
        t = sorted(self.turn_times_ms)
        if not t:
            return {"turns": 0}
        return {"turns": len(t), "p50_ms": t[len(t) // 2], "p90_ms": t[min(len(t) - 1, int(len(t) * 0.9))]}


async def _const(v):
    return v
