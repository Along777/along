"""`skeleton-porch doctor`: camera, detector, mic, speaker, brain, power -- each with a verdict."""
from __future__ import annotations

import socket
import sys
import time

import cv2
import numpy as np

from .audio import Mic, Player, dbfs, find_device, list_devices
from .brain_client import BrainClient
from .camera import Camera, list_cameras
from .config import PORCH_DIR, Config
from .fillers import LocalFillers
from .power import power_status

C = {"ok": "\033[1;32m   ok:\033[0m", "warn": "\033[1;33m warn:\033[0m", "fail": "\033[1;31m fail:\033[0m"}


class Report:
    def __init__(self):
        self.fails = self.warns = 0

    def say(self, t: str) -> None:
        print(f"\n\033[1;36m== {t}\033[0m")

    def line(self, kind: str, msg: str) -> None:
        self.fails += kind == "fail"
        self.warns += kind == "warn"
        print(f"{C[kind]} {msg}")


def chime(sr: int = 24000) -> np.ndarray:
    t = np.arange(int(sr * 0.25)) / sr
    env = np.minimum(1, np.minimum(t, t[::-1]) * 40)
    return np.concatenate([0.3 * env * np.sin(2 * np.pi * f * t) for f in (523, 659, 784)]).astype(np.float32)


def run(cfg: Config, listen_s: float = 4.0) -> int:
    r = Report()
    interactive = sys.stdin.isatty()

    r.say("Camera")
    names = list_cameras()
    for i, n in enumerate(names):
        print(f"   [{i}] {n}")
    cam = None
    try:
        cam = Camera(cfg.camera)
        t0 = time.monotonic()
        while cam.latest()[0] is None and time.monotonic() - t0 < 6:
            time.sleep(0.05)
        frame, _ = cam.latest()
        if frame is None:
            r.line("fail", f"{cam.name} opened but sent no frames (camera privacy setting? another app?)")
        else:
            c0 = cam.count
            time.sleep(2.0)
            fps = (cam.count - c0) / 2.0
            frame, _ = cam.latest()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bright = float(gray.mean())
            snap = PORCH_DIR / "debug" / "doctor-snapshot.jpg"
            snap.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snap), frame)
            kind = "ok" if fps >= 10 else "warn"
            r.line(kind, f"{cam.name}: {frame.shape[1]}x{frame.shape[0]} at {fps:.0f} fps, brightness {bright:.0f}/255")
            if bright < 25:
                r.line("warn", "image is very dark - fine at night only if people still get boxes in `preview`")
            print(f"   snapshot (local only): {snap}")
    except Exception as e:
        r.line("fail", f"camera: {e}")

    r.say("Person detector")
    if cam is not None and cam.latest()[0] is not None:
        try:
            from .detector import PersonDetector

            det = PersonDetector(cfg.detector, cfg.path(cfg.detector.model))
            frame, _ = cam.latest()
            times, n = [], 0
            for _ in range(5):
                t = time.perf_counter()
                n = len(det.detect(frame))
                times.append((time.perf_counter() - t) * 1000)
            ms = float(np.median(times))
            r.line("ok" if ms < 1000 / cfg.detector.fps else "warn",
                   f"YOLO {cfg.detector.imgsz}px: {ms:.0f} ms/frame on CPU; people in view now: {n}")
        except Exception as e:
            r.line("fail", f"detector: {e}")
    else:
        r.line("warn", "skipped (no camera)")
    if cam:
        cam.close()

    r.say("Microphone")
    try:
        dev = find_device(cfg.audio.mic, "input", cfg.audio.host_api)
        print(f"   using: {dev['name']} ({dev['api']})")
        mic = Mic(dev, cfg.vad)
        mic.start()
        print(f"   listening {listen_s:.0f}s - say something like 'trick or treat' ...", flush=True)
        x = mic.record(listen_s)
        mic.close()
        f = mic.vad.frame
        levels = sorted(dbfs(x[i:i + f]) for i in range(0, len(x) - f, f))
        floor, peak = levels[len(levels) // 10], levels[-1]
        if peak < -100:
            r.line("fail", "pure silence - check Windows mic privacy settings and that this is the right mic")
        else:
            r.line("ok", f"noise floor {floor:.0f} dBFS, loudest {peak:.0f} dBFS")
            if peak - floor < 15:
                r.line("warn", "speech barely above the room noise - talk louder, move the mic, or check its gain")
    except Exception as e:
        r.line("fail", f"mic: {e}")

    r.say("Speaker")
    try:
        dev = find_device(cfg.audio.speaker, "output", cfg.audio.host_api)
        print(f"   using: {dev['name']} ({dev['api']})")
        p = Player(dev, cfg.audio.keepalive_dbfs, cfg.audio.gain_db)
        p.start()
        fill = LocalFillers(PORCH_DIR / "cache" / "fillers")
        line = fill.pick("test") if fill.load_latest() else None
        if line:
            p.play(line[0], line[1], "test")
        else:
            p.play(chime(), 24000, "chime")
        while p.busy():
            time.sleep(0.05)
        time.sleep(0.3)
        p.close()
        answer = None
        if interactive:
            try:
                answer = input("   did you hear it? [y/n] ").strip().lower()
            except EOFError:  # stdin is not really a person (e.g. run from a script)
                answer = None
        if answer:
            heard = answer.startswith("y")
            r.line("ok" if heard else "fail", "speaker heard" if heard else "speaker NOT heard - check device/volume")
        else:
            r.line("ok", f"played {'the test line' if line else 'a chime'} (ask a human whether it was heard)")
    except Exception as e:
        r.line("fail", f"speaker: {e}")

    r.say("Brain")
    client = BrainClient(cfg.brain.url, cfg.brain.turn_timeout_s)
    t = time.perf_counter()
    h = client.health(timeout=3)
    ms = (time.perf_counter() - t) * 1000
    if not h:
        r.line("fail", f"no answer from {cfg.brain.url} (is it running? same network? firewall/VPN?)")
    elif not h.get("ready"):
        r.line("warn", f"answering in {ms:.0f} ms but not ready: {h.get('error') or 'warming up'}")
    else:
        r.line("ok", f"ready in {ms:.0f} ms round trip; engines {h.get('engines')}")
        try:
            fill = LocalFillers(PORCH_DIR / "cache" / "fillers")
            v = fill.sync(client)
            r.line("ok", f"{fill.count()} lines cached locally (v{v})")
        except Exception as e:
            r.line("fail", f"line sync: {e}")

    r.say("Power + phone page")
    ac, pct = power_status()
    if ac is False:
        r.line("warn", f"on BATTERY ({pct}%) - plug in for the night")
    elif ac:
        r.line("ok", "on mains power" + (f", battery {pct}%" if pct is not None else ""))
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", cfg.control.port))
        r.line("ok", f"port {cfg.control.port} free for the phone page")
    except OSError:
        r.line("warn", f"port {cfg.control.port} busy (porch already running?)")
    finally:
        s.close()

    r.say("Summary")
    print(f"   all audio devices: skeleton-porch devices")
    if r.fails:
        print(f"\033[1;31m {r.fails} fail(s), {r.warns} warning(s)\033[0m")
        return 1
    print(f"\033[1;32m all checks passed\033[0m ({r.warns} warning(s))")
    return 0


def print_devices(cfg: Config) -> None:
    print("cameras (DirectShow order = OpenCV index):")
    for i, n in enumerate(list_cameras()):
        print(f"  [{i}] {n}")
    for kind in ("input", "output"):
        try:
            pick = find_device(cfg.audio.mic if kind == "input" else cfg.audio.speaker, kind, cfg.audio.host_api)
        except Exception as e:
            pick = None
            print(f"! {kind}: {e}")
        print(f"\n{kind} devices ({cfg.audio.host_api}):")
        key = "inputs" if kind == "input" else "outputs"
        for d in list_devices():
            if d[key] > 0 and d["api"] == cfg.audio.host_api:
                mark = "->" if pick and d["index"] == pick["index"] else "  "
                print(f" {mark} [{d['index']:2d}] {d['name']}  ({d['rate']} Hz)")
