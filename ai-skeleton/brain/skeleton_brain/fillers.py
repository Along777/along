"""Render persona/lines.toml into WAVs the porch caches (greet, thinking, fallback ...).

The version is a hash of the lines + voice + effects, so changing any of them makes
the brain re-render on its next start and the porch re-download on its next sync.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from .audiofx import to_wav

PORCH_CATEGORIES = ("greet", "thinking", "fallback", "no_hear", "still_here", "test")


def load_lines(persona_dir: Path) -> tuple[dict[str, list[str]], str]:
    text = (persona_dir / "lines.toml").read_text(encoding="utf-8")
    data = tomllib.loads(text)
    return {k: [str(s) for s in v] for k, v in data.items() if isinstance(v, list)}, text


def bank_version(lines_text: str, tts_name: str, fx_cfg) -> str:
    h = hashlib.sha256()
    h.update(lines_text.encode())
    h.update(tts_name.encode())
    h.update(json.dumps(asdict(fx_cfg), sort_keys=True).encode())
    return h.hexdigest()[:12]


class FillerBank:
    def __init__(self, root: Path):
        self.root = root
        self.version: str | None = None
        self.manifest: dict = {"version": None, "items": []}

    @property
    def dir(self) -> Path:
        return self.root / str(self.version)

    def load_or_render(self, version: str, lines: dict[str, list[str]],
                       speak: Callable[[str], tuple[np.ndarray, int]], log=print) -> dict:
        self.version = version
        mf = self.dir / "manifest.json"
        if mf.exists():
            self.manifest = json.loads(mf.read_text(encoding="utf-8"))
            return self.manifest
        self.dir.mkdir(parents=True, exist_ok=True)
        items, sr_out = [], 24000
        todo = [(c, i, t) for c in PORCH_CATEGORIES for i, t in enumerate(lines.get(c, []))]
        log(f"rendering {len(todo)} filler lines (version {version}) ...")
        for cat, i, text in todo:
            audio, sr = speak(text)
            sr_out = sr
            wav = to_wav(audio, sr)
            fid = f"{cat}-{i:02d}"
            (self.dir / f"{fid}.wav").write_bytes(wav)
            items.append({"id": fid, "category": cat, "text": text,
                          "sha256": hashlib.sha256(wav).hexdigest(), "duration_s": round(len(audio) / sr, 2)})
        self.manifest = {"version": version, "sample_rate": sr_out, "items": items}
        mf.write_text(json.dumps(self.manifest, indent=1), encoding="utf-8")
        return self.manifest

    def path_for(self, fid: str) -> Path | None:
        if not any(it["id"] == fid for it in self.manifest.get("items", [])):
            return None
        p = self.dir / f"{fid}.wav"
        return p if p.is_file() else None
