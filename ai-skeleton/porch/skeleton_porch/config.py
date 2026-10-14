"""config.toml + optional config.local.toml -> frozen dataclasses (paths relative to porch/)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PORCH_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Brain:
    url: str = "http://127.0.0.1:8080"
    health_interval_s: float = 10.0
    turn_timeout_s: float = 6.0


@dataclass(frozen=True)
class Camera:
    name: str = "Arducam"
    index: int = -1
    width: int = 1280
    height: int = 720
    fps: int = 0
    rotate: int = 0
    flip: bool = False
    photo_max_side: int = 768
    jpeg_quality: int = 85


@dataclass(frozen=True)
class Detector:
    model: str = "models/yolo26n.pt"
    imgsz: int = 416
    fps: float = 5.0
    conf: float = 0.35
    min_box_height: float = 0.25
    roi: tuple = (0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class Audio:
    host_api: str = "Windows WASAPI"
    mic: str = "USB PnP"
    speaker: str = ""
    keepalive_dbfs: float = -65.0
    gain_db: float = 0.0
    tail_margin_s: float = 0.3


@dataclass(frozen=True)
class Vad:
    frame_ms: int = 30
    start_ms: int = 120
    min_speech_ms: int = 200
    end_silence_ms: int = 550
    max_utterance_s: float = 5.0
    no_speech_s: float = 4.0
    min_threshold_dbfs: float = -60.0
    floor_margin_db: float = 10.0
    preroll_ms: int = 300


@dataclass(frozen=True)
class Behavior:
    arrive_s: float = 0.5
    empty_rearm_s: float = 3.0
    cooldown_s: float = 8.0
    thinking_after_s: float = 1.2
    max_turns: int = 4
    still_here_s: float = 60.0
    new_wave_extra: int = 1
    new_wave_s: float = 1.5


@dataclass(frozen=True)
class Control:
    port: int = 8081
    hotkey: str = "f12"


@dataclass(frozen=True)
class Debug:
    save_captures: bool = False


@dataclass(frozen=True)
class Log:
    conversations: bool = True  # text only: what was heard and said (never photos or audio)


@dataclass(frozen=True)
class Config:
    brain: Brain = field(default_factory=Brain)
    camera: Camera = field(default_factory=Camera)
    detector: Detector = field(default_factory=Detector)
    audio: Audio = field(default_factory=Audio)
    vad: Vad = field(default_factory=Vad)
    behavior: Behavior = field(default_factory=Behavior)
    control: Control = field(default_factory=Control)
    debug: Debug = field(default_factory=Debug)
    log: Log = field(default_factory=Log)

    def path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (PORCH_DIR / q).resolve()


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
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
        elif isinstance(default, tuple):
            kwargs[name] = tuple(val)
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            kwargs[name] = type(default)(val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def load(path: Path | None = None) -> Config:
    path = path or PORCH_DIR / "config.toml"
    data: dict = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    local = path.with_name("config.local.toml")
    if local.exists():
        data = _merge(data, tomllib.loads(local.read_text(encoding="utf-8")))
    return _build(Config, data)
