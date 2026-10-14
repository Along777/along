"""config.toml + optional config.local.toml -> frozen dataclasses.

Relative paths in the config are resolved against the brain/ folder, so the
server behaves the same no matter which directory it is started from.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

BRAIN_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Server:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass(frozen=True)
class Ollama:
    url: str = "http://127.0.0.1:11434"
    vision_model: str = "qwen3-vl:8b-instruct"
    reply_model: str = "qwen3-vl:8b-instruct"
    num_ctx: int = 4096
    describe_timeout_s: float = 2.5
    reply_timeout_s: float = 3.0
    describe_max_tokens: int = 48
    reply_max_tokens: int = 50
    reply_temperature: float = 0.8
    warm_interval_s: float = 30.0


@dataclass(frozen=True)
class Stt:
    engine: str = "parakeet"
    model: str = "nemo-parakeet-tdt-0.6b-v3"
    model_dir: str = "models/parakeet-tdt-0.6b-v3"
    quantization: str = "int8"
    timeout_s: float = 2.0


@dataclass(frozen=True)
class Tts:
    engine: str = "kokoro"
    model: str = "models/kokoro/kokoro-v1.0.int8.onnx"
    voices: str = "models/kokoro/voices-v1.0.bin"
    voice: str = "bm_george"
    lang: str = "en-gb"
    speed: float = 0.95
    timeout_s: float = 4.0


@dataclass(frozen=True)
class Fx:
    enabled: bool = True
    highpass_hz: float = 110.0
    pitch_semitones: float = -2.0
    reverb_room: float = 0.18
    reverb_wet: float = 0.12
    target_rms_dbfs: float = -20.0
    limiter_db: float = -1.0


@dataclass(frozen=True)
class Persona:
    dir: str = "../persona"


@dataclass(frozen=True)
class Limits:
    max_photo_bytes: int = 1_000_000
    max_clip_seconds: float = 6.0


@dataclass(frozen=True)
class Config:
    server: Server = field(default_factory=Server)
    ollama: Ollama = field(default_factory=Ollama)
    stt: Stt = field(default_factory=Stt)
    tts: Tts = field(default_factory=Tts)
    fx: Fx = field(default_factory=Fx)
    persona: Persona = field(default_factory=Persona)
    limits: Limits = field(default_factory=Limits)

    def path(self, p: str) -> Path:
        """Resolve a config path relative to the brain/ folder."""
        q = Path(p)
        return q if q.is_absolute() else (BRAIN_DIR / q).resolve()


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _build(cls, data: dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys in [{cls.__name__.lower()}]: {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        val = data[name]
        if is_dataclass(default) and isinstance(val, dict):
            kwargs[name] = _build(type(default), val)
        else:
            kwargs[name] = type(default)(val) if isinstance(default, (int, float)) and not isinstance(default, bool) else val
    return cls(**kwargs)


def load(path: Path | None = None) -> Config:
    path = path or BRAIN_DIR / "config.toml"
    data: dict = {}
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    local = path.with_name("config.local.toml")
    if local.exists():
        data = _merge(data, tomllib.loads(local.read_text(encoding="utf-8")))
    return _build(Config, data)
