"""Thin async client for the local Ollama server (127.0.0.1 only).

Every request sends keep_alive=-1 and the same num_ctx: Ollama runs as its own
service, so environment variables set in a shell never reach it, and a request
with a different num_ctx forces a full model reload.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

import httpx

from .config import Ollama as OllamaCfg


@dataclass
class ModelCaps:
    name: str
    present: bool
    capabilities: list[str] = field(default_factory=list)

    @property
    def vision(self) -> bool:
        return "vision" in self.capabilities

    @property
    def thinking(self) -> bool:
        return "thinking" in self.capabilities


class OllamaClient:
    def __init__(self, cfg: OllamaCfg):
        self.cfg = cfg
        self.http = httpx.AsyncClient(base_url=cfg.url, timeout=httpx.Timeout(60.0, connect=2.0))
        self.caps: dict[str, ModelCaps] = {}

    async def close(self) -> None:
        await self.http.aclose()

    async def version(self) -> str | None:
        try:
            r = await self.http.get("/api/version")
            r.raise_for_status()
            return r.json().get("version")
        except httpx.HTTPError:
            return None

    async def show(self, model: str) -> ModelCaps:
        try:
            r = await self.http.post("/api/show", json={"model": model})
            if r.status_code == 404:
                caps = ModelCaps(model, False)
            else:
                r.raise_for_status()
                caps = ModelCaps(model, True, list(r.json().get("capabilities") or []))
        except httpx.HTTPError:
            caps = ModelCaps(model, False)
        self.caps[model] = caps
        return caps

    async def loaded(self) -> list[dict]:
        """Models currently in memory (/api/ps): name, size, size_vram, ..."""
        try:
            r = await self.http.get("/api/ps")
            r.raise_for_status()
            return list(r.json().get("models") or [])
        except httpx.HTTPError:
            return []

    def _options(self, max_tokens: int, temperature: float) -> dict:
        return {"num_ctx": self.cfg.num_ctx, "num_predict": max_tokens, "temperature": temperature}

    async def chat(self, model: str, messages: list[dict], *, max_tokens: int, temperature: float,
                   timeout: float | None = None) -> dict:
        body: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
            "options": self._options(max_tokens, temperature),
        }
        caps = self.caps.get(model)
        if caps and caps.thinking:
            body["think"] = False  # only sent when the model says it supports the switch
        r = await self.http.post("/api/chat", json=body, timeout=timeout or 60.0)
        r.raise_for_status()
        return r.json()

    async def describe(self, jpeg: bytes, prompt: str, *, timeout: float | None = None) -> tuple[str, dict]:
        data = await self.chat(
            self.cfg.vision_model,
            [{"role": "user", "content": prompt, "images": [base64.b64encode(jpeg).decode("ascii")]}],
            max_tokens=self.cfg.describe_max_tokens,
            temperature=0.2,
            timeout=timeout,
        )
        return _first_line(data.get("message", {}).get("content", "")), data

    async def reply(self, system: str, user: str, *, timeout: float | None = None) -> tuple[str, dict]:
        data = await self.chat(
            self.cfg.reply_model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=self.cfg.reply_max_tokens,
            temperature=self.cfg.reply_temperature,
            timeout=timeout,
        )
        return (data.get("message", {}).get("content", "") or "").strip(), data


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip().strip("-*").strip()
        if line:
            return line[:200]
    return ""
