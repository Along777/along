"""Person detection with Ultralytics YOLO26n on the CPU, limited to the watch zone."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PORCH_DIR, Detector as DetectorCfg

# Keep Ultralytics offline (no analytics, no update checks) and its settings file in
# the project instead of AppData. Must be set before ultralytics is imported, and the
# folder must exist or Ultralytics falls back to \tmp.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", str(PORCH_DIR / "cache" / "ultralytics"))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


class PersonDetector:
    def __init__(self, cfg: DetectorCfg, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"{model_path} missing - run: skeleton-porch fetch-models")
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(str(model_path))
        self.detect(np.zeros((cfg.imgsz, cfg.imgsz, 3), np.uint8))  # warm-up

    def detect(self, frame: np.ndarray) -> list[Box]:
        h, w = frame.shape[:2]
        res = self.model.predict(frame, imgsz=self.cfg.imgsz, conf=self.cfg.conf, classes=[0],
                                 device="cpu", verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        rx0, ry0, rx1, ry1 = self.cfg.roi
        out = []
        for (x0, y0, x1, y1), c in zip(xyxy, conf):
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            if not (rx0 <= cx <= rx1 and ry0 <= cy <= ry1):
                continue
            if (y1 - y0) / h < self.cfg.min_box_height:
                continue
            out.append(Box(float(x0), float(y0), float(x1), float(y1), float(c)))
        return out


def fetch_yolo(model_path: Path) -> None:
    """One-time download of the weights into porch/models/ (needs the internet once)."""
    if model_path.is_file():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_OFFLINE"] = "0"
    from ultralytics.utils.downloads import attempt_download_asset

    attempt_download_asset(str(model_path))  # full path: saved right there, not in the CWD
    if not model_path.is_file():
        raise RuntimeError(f"download of {model_path.name} failed")
