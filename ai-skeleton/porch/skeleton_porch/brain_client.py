"""HTTP client for the brain. Plain requests calls (no shared session across threads)."""
from __future__ import annotations

import json

import requests


class BrainError(RuntimeError):
    pass


class BrainClient:
    def __init__(self, url: str, turn_timeout_s: float):
        self.url = url.rstrip("/")
        self.turn_timeout_s = turn_timeout_s

    def health(self, timeout: float = 1.5) -> dict | None:
        try:
            r = requests.get(f"{self.url}/health", timeout=timeout)
            return r.json() if r.ok else None
        except (requests.RequestException, ValueError):
            return None

    def turn(self, meta: dict, jpeg: bytes | None, wav: bytes | None) -> dict:
        files = {}
        if jpeg:
            files["photo"] = ("photo.jpg", jpeg, "image/jpeg")
        if wav:
            files["clip"] = ("clip.wav", wav, "audio/wav")
        try:
            r = requests.post(f"{self.url}/turn", data={"meta": json.dumps(meta)}, files=files or None,
                              timeout=(1.5, self.turn_timeout_s))
        except requests.RequestException as e:
            raise BrainError(f"{type(e).__name__}") from e
        if not r.ok:
            raise BrainError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def warm(self, jpeg: bytes, timeout: float = 60.0) -> dict | None:
        try:
            r = requests.post(f"{self.url}/warm", files={"photo": ("warm.jpg", jpeg, "image/jpeg")}, timeout=timeout)
            return r.json() if r.ok else None
        except (requests.RequestException, ValueError):
            return None

    def speak(self, text: str, timeout: float = 30.0) -> bytes:
        r = requests.post(f"{self.url}/speak", json={"text": text}, timeout=timeout)
        r.raise_for_status()
        return r.content

    def manifest(self, timeout: float = 5.0) -> dict:
        r = requests.get(f"{self.url}/fillers", timeout=timeout)
        r.raise_for_status()
        return r.json()

    def filler(self, fid: str, timeout: float = 10.0) -> bytes:
        r = requests.get(f"{self.url}/fillers/{fid}.wav", timeout=timeout)
        r.raise_for_status()
        return r.content
