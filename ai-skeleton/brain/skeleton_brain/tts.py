"""Text-to-speech engines. synth() returns (mono float32 samples, sample_rate)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"


class TtsEngine:
    name = "base"

    def synth(self, text: str) -> tuple[np.ndarray, int]:  # pragma: no cover
        raise NotImplementedError


class KokoroOnnx(TtsEngine):
    def __init__(self, model: Path, voices: Path, voice: str, lang: str, speed: float):
        from kokoro_onnx import Kokoro

        for p in (model, voices):
            if not p.is_file():
                raise FileNotFoundError(f"{p} is missing - run: skeleton-brain fetch-models")
        self.k = Kokoro(str(model), str(voices))
        if voice not in self.k.get_voices():
            raise ValueError(f"unknown Kokoro voice {voice!r}")
        self.voice, self.lang, self.speed = voice, lang, speed
        self.name = f"kokoro/{voice}"

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        audio, sr = self.k.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
        return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)


class FakeTts(TtsEngine):
    """A short two-tone chirp per word, so the porch can be tested with no models at all."""

    name = "fake"
    sr = 24000

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        words = max(1, len(text.split()))
        t = np.arange(int(self.sr * 0.12)) / self.sr
        chirp = np.concatenate([0.25 * np.sin(2 * np.pi * 330 * t), 0.25 * np.sin(2 * np.pi * 247 * t)])
        gap = np.zeros(int(self.sr * 0.06), dtype=np.float32)
        out = np.concatenate([np.concatenate([chirp, gap]) for _ in range(min(words, 20))])
        return out.astype(np.float32), self.sr


def fetch_kokoro(model: Path, voices: Path) -> None:
    for p in (model, voices):
        if p.is_file() and p.stat().st_size > 1_000_000:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        url = KOKORO_RELEASE + p.name
        tmp = p.with_suffix(p.suffix + ".part")
        print(f"   downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(p)
