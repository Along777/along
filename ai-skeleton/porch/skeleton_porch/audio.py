"""Speaker and mic, each one stream that stays open all night.

Player: queues clips into a single OutputStream and adds faint keep-alive noise,
so a Bluetooth speaker never auto-sleeps or clips the first syllable.
Mic: 16 kHz frames -> EnergyVad -> utterances. The app gates the mic shut while
anything is playing (plus a tail margin), so the skeleton can never hear itself.
"""
from __future__ import annotations

import collections
import math
import queue
import threading
import time
from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .config import Vad as VadCfg

SR_IN = 16000
SR_OUT = 24000


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    return 20 * math.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + 1e-9)


# -- voice activity -----------------------------------------------------------------
@dataclass
class VadEvent:
    kind: str                       # "start" | "end" | "noise" (started, but too little speech to keep)
    t: float
    audio: np.ndarray | None = None  # "end": the utterance incl. pre-roll, 16 kHz


class EnergyVad:
    """Adaptive energy gate. Pure: feed() 16 kHz frames with timestamps, get events."""

    def __init__(self, cfg: VadCfg, sr: int = SR_IN):
        self.cfg, self.sr = cfg, sr
        self.frame = sr * cfg.frame_ms // 1000
        self.start_frames = max(1, math.ceil(cfg.start_ms / cfg.frame_ms))
        self.end_frames = max(1, math.ceil(cfg.end_silence_ms / cfg.frame_ms))
        self.max_frames = max(1, int(cfg.max_utterance_s * 1000 / cfg.frame_ms))
        self.min_loud = max(1, math.ceil(cfg.min_speech_ms / cfg.frame_ms))
        self.keep_tail = max(1, 200 // cfg.frame_ms)
        self.floor_db = -60.0
        self.warmup = max(1, 1000 // cfg.frame_ms)  # learn the room for 1 s before listening
        self.pre: collections.deque = collections.deque(maxlen=max(1, cfg.preroll_ms // cfg.frame_ms))
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self.loud_run = 0
        self.quiet_run = 0
        self.loud_total = 0
        self.utt: list[np.ndarray] = []
        self.pre.clear()

    @property
    def threshold_db(self) -> float:
        return max(self.floor_db + self.cfg.floor_margin_db, self.cfg.min_threshold_dbfs)

    def feed(self, frame: np.ndarray, t: float) -> list[VadEvent]:
        level = dbfs(frame)
        if self.warmup > 0:
            self.warmup -= 1
            self.floor_db += 0.3 * (level - self.floor_db)
            self.pre.append(frame)
            return []
        loud = level > self.threshold_db
        if not self.in_speech:
            # noise floor: drops fast, rises slowly (speech must not drag it up)
            a = 0.3 if level < self.floor_db else 0.01
            self.floor_db += a * (level - self.floor_db)
            self.pre.append(frame)
            self.loud_run = self.loud_run + 1 if loud else 0
            if self.loud_run >= self.start_frames:
                self.in_speech = True
                self.utt = list(self.pre)
                self.quiet_run = 0
                self.loud_total = self.loud_run
                return [VadEvent("start", t)]
            return []
        self.utt.append(frame)
        self.loud_total += loud
        self.quiet_run = 0 if loud else self.quiet_run + 1
        if self.quiet_run >= self.end_frames or len(self.utt) >= self.max_frames:
            enough = self.loud_total >= self.min_loud
            frames = self.utt[: len(self.utt) - max(0, self.quiet_run - self.keep_tail)]
            audio = np.concatenate(frames) if frames else np.zeros(0, np.float32)
            self.reset()
            if not enough:  # a bang, a gust, a car door: not speech
                return [VadEvent("noise", t)]
            return [VadEvent("end", t, audio.astype(np.float32))]
        return []


# -- devices ------------------------------------------------------------------------
class DeviceNotFound(RuntimeError):
    pass


def list_devices() -> list[dict]:
    import sounddevice as sd

    apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append({"index": i, "name": d["name"], "api": apis[d["hostapi"]]["name"],
                    "inputs": d["max_input_channels"], "outputs": d["max_output_channels"],
                    "rate": int(d["default_samplerate"])})
    return out


def find_device(want: str, kind: str, host_api: str) -> dict:
    """Pick a device by name fragment within the preferred host API. Never a hands-free endpoint."""
    import sounddevice as sd

    key = "inputs" if kind == "input" else "outputs"
    devs = [d for d in list_devices() if d[key] > 0 and "hands-free" not in d["name"].lower()
            and "hands free" not in d["name"].lower()]
    pref = [d for d in devs if d["api"] == host_api] or devs
    if want:
        hits = [d for d in pref if want.lower() in d["name"].lower()]
        if not hits:
            names = sorted({d["name"] for d in pref})
            raise DeviceNotFound(f"no {kind} device matching {want!r} ({host_api}). Have: {names}")
        return hits[0]
    for i, api in enumerate(sd.query_hostapis()):
        if api["name"] == host_api:
            idx = api["default_input_device" if kind == "input" else "default_output_device"]
            for d in pref:
                if d["index"] == idx:
                    return d
    if not pref:
        raise DeviceNotFound(f"no {kind} devices at all")
    return pref[0]


def pick_speaker(want: str, host_api: str) -> tuple[dict, bool]:
    """(device, is_fallback). A Bluetooth speaker that dropped out must not silence the porch:
    fall back to the Windows default output and let the caller switch back when it returns."""
    try:
        return find_device(want, "output", host_api), False
    except DeviceNotFound:
        if not want:
            raise
        return find_device("", "output", host_api), True


def _extra(api: str):
    import sounddevice as sd

    return sd.WasapiSettings(auto_convert=True) if "WASAPI" in api else None


# -- speaker ------------------------------------------------------------------------
class Player:
    def __init__(self, device: dict, keepalive_dbfs: float = -65.0, gain_db: float = 0.0):
        import sounddevice as sd

        self.device = device
        self.sr = SR_OUT
        self.channels = max(1, min(2, device["outputs"]))
        self.lock = threading.Lock()
        self.q: collections.deque = collections.deque()
        self.cur: np.ndarray | None = None
        self.cur_tag: str | None = None
        self.pos = 0
        self.audible_until = 0.0
        self.finished: collections.deque = collections.deque(maxlen=50)
        self.underflows = 0
        self.gain_db = gain_db
        self.silent = False  # "pause speaker": clips still run on time (the mic gate still works), but inaudibly
        self.on_block = None  # e.g. the jaw: called with (time the block is heard, the block) for every block
        amp = 10 ** (keepalive_dbfs / 20) if keepalive_dbfs < 0 else 0.0
        rng = np.random.default_rng(7)
        self.noise = (rng.standard_normal(self.sr) * amp).astype(np.float32) if amp else None
        self.noise_pos = 0
        self.stream = sd.OutputStream(device=device["index"], samplerate=self.sr, channels=self.channels,
                                      dtype="float32", callback=self._cb, latency="low",
                                      extra_settings=_extra(device["api"]))

    @property
    def gain(self) -> float:
        return 10 ** (self.gain_db / 20)

    def start(self) -> None:
        self.stream.start()

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()

    def play(self, audio: np.ndarray, sr: int, tag: str = "") -> None:
        x = resample(np.asarray(audio, dtype=np.float32).reshape(-1), sr, self.sr)
        with self.lock:
            self.q.append((x, tag))

    def cut(self, tag: str, fade_s: float = 0.08) -> None:
        """Fade out a playing clip with this tag (and drop queued ones), e.g. a thinking filler once the reply is in."""
        with self.lock:
            self.q = collections.deque(item for item in self.q if item[1] != tag)
            if self.cur is not None and self.cur_tag == tag:
                end = min(len(self.cur), self.pos + int(fade_s * self.sr))
                tail = self.cur[self.pos:end] * np.linspace(1, 0, end - self.pos, dtype=np.float32)
                self.cur = np.concatenate([self.cur[:self.pos], tail])

    def flush(self) -> None:
        with self.lock:
            self.q.clear()
            self.cur = None
            self.audible_until = 0.0

    def busy(self, now: float | None = None, margin: float = 0.0) -> bool:
        now = time.monotonic() if now is None else now
        with self.lock:
            return self.cur is not None or bool(self.q) or now < self.audible_until + margin

    def playing_tag(self) -> str | None:
        with self.lock:
            return self.cur_tag if self.cur is not None else (self.q[0][1] if self.q else None)

    def _cb(self, outdata, frames, time_info, status) -> None:
        if status.output_underflow:
            self.underflows += 1
        out = np.zeros(frames, np.float32)
        with self.lock:
            filled = 0
            while filled < frames:
                if self.cur is None:
                    if not self.q:
                        break
                    self.cur, self.cur_tag = self.q.popleft()
                    self.pos = 0
                n = min(frames - filled, len(self.cur) - self.pos)
                out[filled:filled + n] = self.cur[self.pos:self.pos + n]
                filled += n
                self.pos += n
                if self.pos >= len(self.cur):
                    self.finished.append(self.cur_tag)
                    self.cur = None
            if filled:
                self.audible_until = time.monotonic() + filled / self.sr + float(self.stream.latency)
        out *= 0.0 if self.silent else self.gain
        if self.on_block is not None:  # before the keep-alive noise: the jaw follows the voice only
            try:
                self.on_block(time.monotonic() + float(self.stream.latency), out)
            except Exception:
                pass
        if self.noise is not None:
            idx = (np.arange(frames) + self.noise_pos) % len(self.noise)
            out += self.noise[idx]
            self.noise_pos = (self.noise_pos + frames) % len(self.noise)
        np.clip(out, -1, 1, out=out)
        outdata[:] = out[:, None]


# -- microphone ---------------------------------------------------------------------
class Mic:
    def __init__(self, device: dict, cfg: VadCfg):
        import sounddevice as sd

        self.device = device
        self.cfg = cfg
        self.vad = EnergyVad(cfg, SR_IN)
        self.q: queue.Queue = queue.Queue(maxsize=2000)
        self.open = False
        self.level_db = -120.0
        self.levels: collections.deque = collections.deque(maxlen=200)  # (t, dBFS) per 30 ms frame, for meters
        self.last_utterance_s = 0.0
        self.overflows = 0
        self.dropped = 0
        self.stream = sd.InputStream(device=device["index"], samplerate=SR_IN, channels=1, dtype="float32",
                                     blocksize=self.vad.frame, callback=self._cb,
                                     extra_settings=_extra(device["api"]))

    def start(self) -> None:
        self.stream.start()

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()

    def _cb(self, indata, frames, time_info, status) -> None:
        if status.input_overflow:
            self.overflows += 1
        try:
            self.q.put_nowait((indata[:, 0].copy(), time.monotonic()))
        except queue.Full:
            self.dropped += 1

    def set_open(self, on: bool) -> None:
        if on != self.open:
            self.vad.reset()  # never carry pre-roll across a gate change (it could hold our own voice)
            self.open = on

    def poll(self) -> list[VadEvent]:
        events: list[VadEvent] = []
        while True:
            try:
                frame, t = self.q.get_nowait()
            except queue.Empty:
                return events
            self.level_db = dbfs(frame)
            self.levels.append((t, self.level_db))
            if self.open:
                for e in self.vad.feed(frame, t):
                    if e.kind == "end" and e.audio is not None:
                        self.last_utterance_s = len(e.audio) / SR_IN
                    events.append(e)

    def peak_db(self, window_s: float = 0.3) -> float:
        """Loudest frame in the last window (what a level meter should show)."""
        now = time.monotonic()
        vals = [lv for t, lv in list(self.levels) if now - t <= window_s]
        return max(vals) if vals else -120.0

    def record(self, seconds: float) -> np.ndarray:
        """Blocking raw capture for the doctor (drains the live queue first)."""
        while not self.q.empty():
            self.q.get_nowait()
        chunks, need = [], int(seconds * SR_IN)
        got = 0
        while got < need:
            frame, _ = self.q.get(timeout=2.0)
            chunks.append(frame)
            got += len(frame)
        return np.concatenate(chunks)[:need]
