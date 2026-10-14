"""Local cache of the brain's pre-rendered lines, so greetings and fallbacks work offline."""
from __future__ import annotations

import hashlib
import io
import json
import random
import wave
from pathlib import Path

import numpy as np


def wav_to_float(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    return (x.reshape(-1, ch).mean(axis=1) if ch > 1 else x), sr


def float_to_wav(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class LocalFillers:
    def __init__(self, root: Path, rng: random.Random | None = None):
        self.root = root
        self.version: str | None = None
        self.items: dict[str, list[tuple[str, np.ndarray, int, str]]] = {}
        self.last: dict[str, str] = {}
        self.rng = rng or random.Random()

    def count(self) -> int:
        return sum(len(v) for v in self.items.values())

    def sync(self, client) -> str:
        """Download whatever the brain's current bank has that we don't (sha256-checked)."""
        mf = client.manifest()
        d = self.root / mf["version"]
        d.mkdir(parents=True, exist_ok=True)
        for it in mf["items"]:
            p = d / f"{it['id']}.wav"
            if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == it["sha256"]:
                continue
            data = client.filler(it["id"])
            if hashlib.sha256(data).hexdigest() != it["sha256"]:
                raise RuntimeError(f"filler {it['id']} failed its checksum")
            p.write_bytes(data)
        (d / "manifest.json").write_text(json.dumps(mf, indent=1), encoding="utf-8")
        self.load(d)
        return mf["version"]

    def load_latest(self) -> bool:
        """No brain? Use the newest bank already on disk."""
        dirs = sorted((p for p in self.root.glob("*/manifest.json")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not dirs:
            return False
        self.load(dirs[0].parent)
        return True

    def load(self, d: Path) -> None:
        mf = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        items: dict[str, list] = {}
        for it in mf["items"]:
            p = d / f"{it['id']}.wav"
            if p.is_file():
                x, sr = wav_to_float(p.read_bytes())
                items.setdefault(it["category"], []).append((it["id"], x, sr, it["text"]))
        self.items, self.version = items, mf["version"]

    def pick(self, category: str):
        """(audio, sr, text) for a random line in the category, never the same one twice in a row."""
        opts = self.items.get(category) or []
        if not opts:
            return None
        if len(opts) > 1:
            opts = [o for o in opts if o[0] != self.last.get(category)]
        fid, x, sr, text = self.rng.choice(opts)
        self.last[category] = fid
        return x, sr, text
