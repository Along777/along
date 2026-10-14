"""Voice post-processing and WAV helpers (stdlib wave + numpy, no sox/ffmpeg)."""
from __future__ import annotations

import io
import wave
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .config import Fx


class VoiceFx:
    """High-pass, pitch down, small-room reverb, loudness, limiter (pedalboard, in-process)."""

    def __init__(self, cfg: Fx, pitch: bool = True):
        self.cfg = cfg
        self.board = None
        if cfg.enabled:
            from pedalboard import HighpassFilter, Limiter, Pedalboard, PitchShift, Reverb

            chain = [HighpassFilter(cutoff_frequency_hz=cfg.highpass_hz)]
            if pitch and cfg.pitch_semitones:
                chain.append(PitchShift(semitones=cfg.pitch_semitones))
            chain.append(Reverb(room_size=cfg.reverb_room, wet_level=cfg.reverb_wet, dry_level=1.0 - cfg.reverb_wet,
                                damping=0.6, width=0.0))
            self.board = Pedalboard(chain)
            self.limiter = Pedalboard([Limiter(threshold_db=cfg.limiter_db, release_ms=80)])

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        x = audio.astype(np.float32).reshape(-1)
        if self.board is None:
            return np.clip(x, -1, 1)
        x = self.board(x, sr).reshape(-1)
        x = normalize_rms(x, self.cfg.target_rms_dbfs)
        x = self.limiter(x, sr).reshape(-1)
        # tiny fades so the speaker never pops
        n = min(len(x) // 4, int(sr * 0.008))
        if n > 0:
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            x[:n] *= ramp
            x[-n:] *= ramp[::-1]
        return np.clip(x, -1, 1)


def normalize_rms(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    voiced = x[np.abs(x) > 1e-3]
    if voiced.size == 0:
        return x
    rms = float(np.sqrt(np.mean(voiced**2)))
    gain = 10 ** (target_dbfs / 20) / max(rms, 1e-6)
    return (x * min(gain, 20.0)).astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def to_wav(x: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def from_wav(data: bytes) -> tuple[np.ndarray, int]:
    """PCM16 WAV -> mono float32. Raises ValueError on anything else."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError) as e:
        raise ValueError(f"not a WAV file: {e}") from e
    if width != 2:
        raise ValueError("clip must be 16-bit PCM")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr
