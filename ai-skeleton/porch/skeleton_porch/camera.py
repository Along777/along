"""Camera: picked by its Windows name, read on a thread that keeps only the newest frame."""
from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np

from .config import Camera as CameraCfg


def list_cameras() -> list[str]:
    """DirectShow device names, in the same order as OpenCV's CAP_DSHOW indexes."""
    if sys.platform != "win32":
        return []
    from pygrabber.dshow_graph import FilterGraph

    return list(FilterGraph().get_input_devices())


def find_camera(cfg: CameraCfg) -> tuple[int, str]:
    names = list_cameras()
    if cfg.index >= 0:
        return cfg.index, names[cfg.index] if cfg.index < len(names) else f"camera {cfg.index}"
    if cfg.name:
        for i, n in enumerate(names):
            if cfg.name.lower() in n.lower():
                return i, n
        raise RuntimeError(f"no camera matching {cfg.name!r}. Have: {names}")
    return 0, names[0] if names else "camera 0"


def to_jpeg(frame: np.ndarray, max_side: int, quality: int) -> bytes:
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


class Camera:
    """Self-healing: a stream that stops, or turns all-black (seen on the Arducam after a
    re-plug: 1 fps of pure zeros), is closed and reopened, found again by name."""

    STUCK_BLACK = 15      # all-zero frames in a row
    STUCK_FAILS = 40      # failed reads in a row (~2 s)

    def __init__(self, cfg: CameraCfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.t = 0.0
        self.count = 0
        self.fails = 0
        self.reopens = 0
        self._open()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self.thread.start()

    def _fourcc(self) -> str:
        v = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))

    def _open(self) -> None:
        self.index, self.name = find_camera(self.cfg)
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open {self.name} (index {self.index}) - is another app using it?")
        # MJPG only. The Arducam's uncompressed YUY2 modes return empty frames at 1 fps, and once it has
        # been switched to YUY2 it stays there until it is unplugged. DirectShow can quietly fall back to
        # YUY2 when the size is set, so set MJPG, then the size, then MJPG again, and check what stuck.
        mjpg = cv2.VideoWriter.fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        if self._fourcc() != "MJPG":
            self.cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        if self.cfg.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        self.mode = (f"{self._fourcc()} {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                     f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        self.stuck_format = self._fourcc() != "MJPG"

    def _reopen(self) -> None:
        self.reopens += 1
        try:
            self.cap.release()
        except Exception:
            pass
        while not self._stop.is_set():
            time.sleep(1.0)
            try:
                self._open()
                return
            except Exception:
                continue  # unplugged: keep trying once a second

    def _fix(self, f: np.ndarray) -> np.ndarray:
        r = self.cfg.rotate % 360
        if r == 90:
            f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
        elif r == 180:
            f = cv2.rotate(f, cv2.ROTATE_180)
        elif r == 270:
            f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.flip(f, 1) if self.cfg.flip else f

    def _run(self) -> None:
        failed = black = 0
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if not ok or f is None:
                self.fails += 1
                failed += 1
                if failed >= self.STUCK_FAILS:
                    self._reopen()
                    failed = 0
                time.sleep(0.05)
                continue
            failed = 0
            if not f[::16, ::16].any():  # every sampled pixel exactly zero: a stuck stream, not a dark night
                black += 1
                if black >= self.STUCK_BLACK:
                    self._reopen()
                    black = 0
                continue
            black = 0
            f = self._fix(f)
            with self.lock:
                self.frame, self.t = f, time.monotonic()
                self.count += 1

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self.lock:
            return self.frame, self.t

    def ok(self, now: float, stale_s: float = 5.0) -> bool:
        return self.frame is not None and now - self.t < stale_s

    def jpeg(self, frame: np.ndarray) -> bytes:
        return to_jpeg(frame, self.cfg.photo_max_side, self.cfg.jpeg_quality)

    def close(self) -> None:
        self._stop.set()
        self.thread.join(timeout=2)
        self.cap.release()
