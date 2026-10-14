"""Move Mister Bones' jaw with his voice: speaker loudness -> servo angle -> Arduino Nano over USB.

The speaker callback feeds every audio block (with the time it will be heard) into JawDriver.
A 40 Hz thread asks the driver for the jaw angle `delay_s` later, so the mouth lines up with a
Bluetooth speaker that plays a little late. Firmware: ai-skeleton/jaw/jaw.ino.
"""
from __future__ import annotations

import collections
import math
import threading
import time

import numpy as np

from .config import Jaw as JawCfg

HELLO, RELAX = 254, 255
CH340 = {(0x1A86, 0x7523), (0x1A86, 0x55D4)}  # the USB chip on ELEGOO Nano boards


class JawDriver:
    """Pure logic (unit-tested): audio levels in, servo angles out."""

    def __init__(self, cfg: JawCfg):
        self.cfg = cfg
        self.levels: collections.deque = collections.deque(maxlen=600)  # (t_heard, dBFS)
        self.open = 0.0          # smoothed openness, 0 (shut) .. 1 (wide)
        self.last_loud = -1e9
        self.relaxed = True
        self.lock = threading.Lock()

    def feed(self, t_heard: float, block: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64)))) if block.size else 0.0
        with self.lock:
            self.levels.append((t_heard, 20 * math.log10(rms + 1e-9)))

    def level_at(self, t: float) -> float:
        """Loudness of the audio being heard at time t (the newest block that started by then)."""
        with self.lock:
            best = -120.0
            for th, db in reversed(self.levels):
                if th <= t:
                    best = db
                    break
        return best

    def tick(self, now: float) -> int | None:
        """The angle to send now, RELAX once after speech ends, or None for no change."""
        c = self.cfg
        db = self.level_at(now - c.delay_s)
        x = min(1.0, max(0.0, (db - c.gate_db) / max(1e-6, c.full_db - c.gate_db)))
        if x > 0:
            self.last_loud = now
        # open fast on a syllable, close a little slower (looks like talking, not flapping)
        k = 0.7 if x > self.open else 0.35
        self.open += k * (x - self.open)
        if now - self.last_loud > c.relax_after_s:
            if self.relaxed:
                return None
            self.relaxed = True
            self.open = 0.0
            return RELAX
        self.relaxed = False
        return int(round(c.closed_deg + (c.open_deg - c.closed_deg) * self.open))


def find_port(want: str = "") -> str | None:
    if want:
        return want
    from serial.tools import list_ports

    for p in list_ports.comports():
        if (p.vid, p.pid) in CH340 or "CH340" in (p.description or ""):
            return p.device
    return None


class Jaw:
    """The serial link + a 40 Hz loop around JawDriver."""

    def __init__(self, cfg: JawCfg, log=print):
        import serial

        self.cfg, self.log = cfg, log
        port = find_port(cfg.port)
        if not port:
            raise RuntimeError("no jaw controller found - is the Nano plugged in? (CH340 USB device)")
        self.ser = serial.Serial(port, 115200, timeout=1.0, write_timeout=0.2)
        time.sleep(2.0)  # opening the port resets the Nano; give it time to boot
        self.ser.reset_input_buffer()
        self.ser.write(bytes([HELLO]))
        hello = self.ser.readline().decode(errors="ignore").strip()
        if hello != "JAW1":
            self.ser.close()
            raise RuntimeError(f"{port} did not answer like the jaw (got {hello!r}) - flash jaw/jaw.ino")
        self.port = port
        self.driver = JawDriver(cfg)
        self.last_sent: int | None = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="jaw", daemon=True)
        self.thread.start()

    def feed(self, t_heard: float, block: np.ndarray) -> None:
        self.driver.feed(t_heard, block)

    def send(self, value: int) -> None:
        try:
            self.ser.write(bytes([value]))
            self.last_sent = value
        except Exception as e:  # never let the jaw take the porch down
            self.log(f"jaw: {e}")

    def _run(self) -> None:
        last_t = 0.0
        while not self._stop.wait(0.025):
            now = time.monotonic()
            v = self.driver.tick(now)
            if v is None:
                continue
            # resend a held angle every 0.2 s, or the board's 0.5 s quiet rule would relax the jaw mid-word
            if v != self.last_sent or (v != RELAX and now - last_t > 0.2):
                self.send(v)
                last_t = now

    def close(self) -> None:
        self._stop.set()
        self.thread.join(timeout=1)
        self.send(RELAX)
        self.ser.close()
