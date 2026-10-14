"""skeleton-porch run | devices | doctor | preview | selftest-turn | record-voice | stats | fetch-models"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import config as config_mod
from .config import PORCH_DIR


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="skeleton-porch", description="AI porch skeleton - the porch side")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the real thing")
    r.add_argument("--debug", action="store_true", help="save photos+clips to porch/debug/ and log what was heard")
    r.add_argument("--preview", action="store_true", help="show the camera with boxes and state")
    sub.add_parser("devices", help="list cameras, mics and speakers (-> marks the ones config picks)")
    d = sub.add_parser("doctor", help="check camera, detector, mic, speaker, brain, power")
    d.add_argument("--listen", type=float, default=4.0)
    sub.add_parser("preview", help="camera + person boxes only, to aim the camera and tune the watch zone")
    s = sub.add_parser("selftest-turn", help="one full turn through the brain with a live photo")
    s.add_argument("--text", default="Trick or treat, I am a pirate.")
    v = sub.add_parser("record-voice", help="record a voice reference for cloning (Phase 2)")
    v.add_argument("--seconds", type=float, default=15.0)
    sub.add_parser("mictest", help="guided mic test (quiet, normal, soft, 2 m away); saves a log to porch/logs/tests")
    sub.add_parser("stats", help="latency + sources from porch/logs/turns.jsonl")
    sub.add_parser("fetch-models", help="download the person-detector weights (once)")
    a = p.parse_args(argv)
    if sys.platform == "win32":
        os.system("")  # enable ANSI colours in old consoles
    cfg = config_mod.load()

    if a.cmd == "fetch-models":
        from .detector import fetch_yolo

        fetch_yolo(cfg.path(cfg.detector.model))
        print(f"ok: {cfg.path(cfg.detector.model)}")
        return 0
    if a.cmd == "devices":
        from .doctor import print_devices

        print_devices(cfg)
        return 0
    if a.cmd == "doctor":
        from .doctor import run as doctor_run

        return doctor_run(cfg, a.listen)
    if a.cmd == "run":
        from .app import PorchApp

        PorchApp(cfg, debug=a.debug, preview=a.preview).run()
        return 0
    if a.cmd == "preview":
        return preview(cfg)
    if a.cmd == "selftest-turn":
        return selftest_turn(cfg, a.text)
    if a.cmd == "record-voice":
        return record_voice(cfg, a.seconds)
    if a.cmd == "mictest":
        return mictest(cfg)
    if a.cmd == "stats":
        return stats()
    return 2


def mictest(cfg) -> int:
    """Measure the mic the way the porch hears it, ask the brain what it understood, and save it all."""
    from datetime import datetime

    import numpy as np
    import sounddevice as sd

    from .audio import SR_IN, EnergyVad, _extra, dbfs, find_device
    from .brain_client import BrainClient
    from .fillers import float_to_wav

    dev = find_device(cfg.audio.mic, "input", cfg.audio.host_api)
    out = PORCH_DIR / "logs" / "tests"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    client = BrainClient(cfg.brain.url, 30)
    h = client.health()
    brain_ok = bool(h and h.get("ready"))
    frame = SR_IN * cfg.vad.frame_ms // 1000
    print(f"mic:   {dev['name']} ({dev['api']})")
    print(f"brain: {'ready' if brain_ok else 'NOT reachable - no transcripts'} at {cfg.brain.url}")
    print("Stop the porch first if it is running. Recordings stay on this laptop (porch/logs/tests).")
    phases = [
        ("quiet", "Stay quiet - this measures the room", 4.0),
        ("normal", "Say 'trick or treat' twice in your normal voice, from about 1 m (where kids will stand)", 5.0),
        ("soft", "Say it again quietly, like a shy kid, from the same spot", 5.0),
        ("far", "Step back to about 2 m and say it in your normal voice", 5.0),
    ]
    log = {"when": stamp, "mic": dev["name"], "min_threshold_dbfs": cfg.vad.min_threshold_dbfs,
           "floor_margin_db": cfg.vad.floor_margin_db, "phases": {}}
    floor = None
    for key, prompt, secs in phases:
        try:
            input(f"\n>> {prompt}.\n   Press Enter, then start. Recording {secs:.0f} s ...")
        except EOFError:
            time.sleep(1)
        x = sd.rec(int(secs * SR_IN), samplerate=SR_IN, channels=1, dtype="float32", device=dev["index"],
                   extra_settings=_extra(dev["api"]))
        sd.wait()
        x = x[:, 0].copy()
        levels = sorted(dbfs(x[i:i + frame]) for i in range(0, len(x) - frame + 1, frame))
        p10, p90, peak = levels[len(levels) // 10], levels[int(len(levels) * 0.9)], levels[-1]
        if key == "quiet":
            floor = float(np.median(levels))
        vad = EnergyVad(cfg.vad)
        vad.warmup = 0
        if floor is not None:
            vad.floor_db = floor
        thr = vad.threshold_db
        events = []
        for i in range(0, len(x) - frame + 1, frame):
            events += vad.feed(x[i:i + frame], i / SR_IN)
        utts = [e.audio for e in events if e.kind == "end"]
        if vad.in_speech and vad.utt:  # still talking when the take ended
            utts.append(np.concatenate(vad.utt))
        wav = out / f"mictest-{stamp}-{key}.wav"
        wav.write_bytes(float_to_wav(x, SR_IN))
        res = {"room_p10_db": round(p10, 1), "loud_p90_db": round(p90, 1), "peak_db": round(peak, 1),
               "threshold_db": round(thr, 1), "wav": wav.name}
        print(f"   room {p10:.0f} dB | loud parts {p90:.0f} dB | peak {peak:.0f} dB | speech threshold {thr:.0f} dB")
        if key != "quiet":
            res.update({"detected": bool(utts), "margin_db": round(p90 - thr, 1)})
            heard = None
            if brain_ok:
                clip = utts[0] if utts else x
                try:
                    r = client.turn({"turn_id": f"mictest-{key}", "debug": True}, None,
                                    float_to_wav(clip[: int(6 * SR_IN)], SR_IN))
                    heard = r.get("transcript")
                except Exception as e:
                    heard = f"(brain error: {e})"
            res["heard"] = heard
            print(f"   counted as speech: {'YES' if utts else 'NO'} (loud parts {p90 - thr:+.0f} dB vs threshold)")
            if heard is not None:
                print(f"   the brain heard: {heard!r}")
        log["phases"][key] = res

    n = log["phases"].get("normal", {})
    tips = []
    if not n.get("detected"):
        tips.append("Your normal voice did NOT count as speech. Raise the mic: Settings > System > Sound > Input > "
                    f"'{dev['name']}' > Volume near 100 (and not muted), or move the mic closer to where kids talk.")
    elif n.get("margin_db", 0) < 6:
        tips.append("Normal speech only just clears the threshold. More mic volume or a closer mic would make it reliable.")
    if n.get("heard") is not None and "trick" not in str(n.get("heard")).lower():
        tips.append("The brain could not make out 'trick or treat' - the mic is too quiet, too far, or muffled.")
    if floor is not None and n.get("loud_p90_db") is not None:
        suggest = round(max(floor + 8, n["loud_p90_db"] - 12))
        if suggest < cfg.vad.min_threshold_dbfs - 2:
            tips.append(f"Try in porch/config.local.toml:  [vad]  min_threshold_dbfs = {suggest}")
        log["suggested_min_threshold_dbfs"] = suggest
    if not tips:
        tips.append("Mic looks good: speech clears the threshold with room to spare and the brain understood it.")
    log["tips"] = tips
    path = out / f"mictest-{stamp}.json"
    path.write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("\n== Result")
    for t in tips:
        print(f" - {t}")
    print(f"\nlog: {path}")
    return 0


def preview(cfg) -> int:
    import cv2

    from .camera import Camera
    from .detector import PersonDetector

    cam = Camera(cfg.camera)
    det = PersonDetector(cfg.detector, cfg.path(cfg.detector.model))
    print(f"{cam.name}: q in the window to quit")
    x0, y0, x1, y1 = cfg.detector.roi
    while True:
        frame, _ = cam.latest()
        if frame is None:
            time.sleep(0.05)
            continue
        t = time.perf_counter()
        boxes = det.detect(frame)
        ms = (time.perf_counter() - t) * 1000
        img = frame.copy()
        h, w = img.shape[:2]
        cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h) - 1), (255, 200, 0), 2)
        for b in boxes:
            cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), (0, 220, 0), 2)
            cv2.putText(img, f"{b.conf:.2f}", (int(b.x0), int(b.y0) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        cv2.putText(img, f"people={len(boxes)}  {ms:.0f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("preview (q to quit)", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.close()
    cv2.destroyAllWindows()
    return 0


def selftest_turn(cfg, text: str) -> int:
    import base64

    from .audio import Player, pick_speaker, resample
    from .brain_client import BrainClient
    from .camera import Camera
    from .fillers import float_to_wav, wav_to_float

    client = BrainClient(cfg.brain.url, max(cfg.brain.turn_timeout_s, 60))
    h = client.health(timeout=3)
    if not (h and h.get("ready")):
        print(f"brain not ready at {cfg.brain.url}: {h}")
        return 1
    jpeg = None
    try:
        cam = Camera(cfg.camera)
        t0 = time.monotonic()
        while cam.latest()[0] is None and time.monotonic() - t0 < 6:
            time.sleep(0.05)
        f, _ = cam.latest()
        jpeg = cam.jpeg(f) if f is not None else None
        cam.close()
    except Exception as e:
        print(f"(no camera: {e})")
    speech, sr = wav_to_float(client.speak(text))
    clip = float_to_wav(resample(speech, sr, 16000), 16000)
    t = time.perf_counter()
    res = client.turn({"turn_id": "selftest", "group_size": 1, "turn_index": 0, "debug": True}, jpeg, clip)
    rt = (time.perf_counter() - t) * 1000
    print(json.dumps({k: v for k, v in res.items() if k != "audio"}, indent=1))
    print(f"round trip {rt:.0f} ms (brain {res['timings_ms'].get('total')} ms)")
    audio, asr = wav_to_float(base64.b64decode(res["audio"]["b64"]))
    dev, fallback = pick_speaker(cfg.audio.speaker, cfg.audio.host_api)
    if fallback:
        print(f"speaker '{cfg.audio.speaker}' not connected - playing on {dev['name']}")
    p = Player(dev, cfg.audio.keepalive_dbfs, cfg.audio.gain_db)
    p.start()
    p.play(audio, asr, "reply")
    while p.busy():
        time.sleep(0.05)
    time.sleep(0.3)
    p.close()
    return 0


def record_voice(cfg, seconds: float) -> int:
    import numpy as np
    import sounddevice as sd

    from .audio import _extra, find_device
    from .fillers import float_to_wav

    dev = find_device(cfg.audio.mic, "input", cfg.audio.host_api)
    sr = 48000
    print(f"mic: {dev['name']}. Read in your dry, amused skeleton voice for {seconds:.0f}s.")
    print("Tip: a few lines from persona/lines.toml, with feeling. Starting in 3...", flush=True)
    time.sleep(3)
    print("RECORDING", flush=True)
    x = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=dev["index"],
               extra_settings=_extra(dev["api"]))
    sd.wait()
    x = x[:, 0]
    peak = float(np.abs(x).max())
    if peak > 0:
        x = x * min(0.9 / peak, 4.0)
    out = PORCH_DIR / "voices" / "reference.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(float_to_wav(x, sr))
    print(f"saved {out} (local only; gitignored). Peak was {20 * np.log10(peak + 1e-9):.0f} dBFS.")
    return 0


def stats() -> int:
    path = PORCH_DIR / "logs" / "turns.jsonl"
    if not path.exists():
        print("no turns logged yet")
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def pct(vals, q):
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(len(vals) * q))] if vals else None

    lat = [r["speech_end_to_reply_ms"] for r in rows if "speech_end_to_reply_ms" in r]
    rtt = [r["roundtrip_ms"] for r in rows]
    brain = [r["brain_ms"].get("total") for r in rows if r.get("brain_ms", {}).get("total") is not None]
    sources: dict[str, int] = {}
    for r in rows:
        sources[r.get("source", "?")] = sources.get(r.get("source", "?"), 0) + 1
    print(f"{len(rows)} replies; sources {sources}")
    print(f"end of speech -> reply queued: p50 {pct(lat, .5)} ms, p90 {pct(lat, .9)} ms (n={len(lat)})")
    print(f"porch round trip:             p50 {pct(rtt, .5)} ms, p90 {pct(rtt, .9)} ms")
    print(f"brain time:                   p50 {pct(brain, .5)} ms, p90 {pct(brain, .9)} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
