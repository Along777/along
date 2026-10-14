"""The porch loop: sensors -> state machine -> speaker, 20 times a second.

Threads: camera reader, person detector (5 fps), brain health check, one worker
per turn, the phone page, the F12 hook. Audio runs in PortAudio callbacks.
"""
from __future__ import annotations

import base64
import collections
import json
import queue
import statistics
import threading
import time
from datetime import datetime

from .audio import Mic, Player, find_device, pick_speaker
from .brain_client import BrainClient
from .camera import Camera
from .config import PORCH_DIR, Config
from .control import Control
from .fillers import LocalFillers, float_to_wav, wav_to_float
from .power import allow_sleep, keep_awake, power_status
from .statemachine import CancelTurn, Flush, Fsm, Obs, PlayReply, Say, SubmitTurn, TurnOutcome

TICK_S = 0.05


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


class PorchApp:
    def __init__(self, cfg: Config, debug: bool = False, preview: bool = False):
        self.cfg = cfg
        self.debug = debug or cfg.debug.save_captures
        self.preview = preview
        self.client = BrainClient(cfg.brain.url, cfg.brain.turn_timeout_s)
        self.fillers = LocalFillers(PORCH_DIR / "cache" / "fillers")
        self.fsm = Fsm(cfg.behavior, cfg.vad.no_speech_s, cfg.brain.turn_timeout_s + 1.0)
        self.results: queue.Queue = queue.Queue()
        self.cancelled: set[int] = set()
        self.marks: dict[int, dict] = {}
        self.brain_ok = False
        self.health: dict | None = None
        self.lock = threading.Lock()
        self.people_hist: collections.deque = collections.deque()
        self.frames: collections.deque = collections.deque(maxlen=60)
        self.last_boxes: list = []
        self.det_ms: collections.deque = collections.deque(maxlen=50)
        self.last_reply = ""
        self.turns_total = 0
        self.lat_ms: collections.deque = collections.deque(maxlen=50)
        self.stop = threading.Event()
        self.camera = None
        self.detector = None
        self.log_path = PORCH_DIR / "logs" / "turns.jsonl"

    def log(self, msg: str) -> None:
        print(f"[{stamp()}] {msg}", flush=True)

    # -- startup --------------------------------------------------------------------
    def start(self) -> None:
        cfg = self.cfg
        mic_dev = find_device(cfg.audio.mic, "input", cfg.audio.host_api)
        spk_dev, self.spk_fallback = pick_speaker(cfg.audio.speaker, cfg.audio.host_api)
        self.log(f"mic: {mic_dev['name']} | speaker: {spk_dev['name']} ({spk_dev['api']})")
        if self.spk_fallback:
            self.log(f"SPEAKER '{cfg.audio.speaker}' NOT CONNECTED - using {spk_dev['name']} until it is back")
        self.player = Player(spk_dev, cfg.audio.keepalive_dbfs, cfg.audio.gain_db)
        self.player.start()
        self.spk_checked = time.monotonic()
        self.mic = Mic(mic_dev, cfg.vad)
        self.mic.start()
        try:
            self.camera = Camera(cfg.camera)
            from .detector import PersonDetector

            self.detector = PersonDetector(cfg.detector, cfg.path(cfg.detector.model))
            self.log(f"camera: {self.camera.name}; detector ready")
            threading.Thread(target=self._detect_loop, name="detector", daemon=True).start()
        except Exception as e:
            self.log(f"CAMERA OFF ({e}) - running voice-triggered")
            self.camera = None
        self._refresh_brain()
        threading.Thread(target=self._health_loop, name="health", daemon=True).start()
        self.control = Control(cfg.control.port, self.status, self.test_line, self.change_gain, self.log)
        try:
            self.control.start_http()
            self.log(f"phone page: http://<this-laptop-ip>:{cfg.control.port}/")
        except OSError as e:
            self.log(f"phone page unavailable on :{cfg.control.port} ({e})")
        if self.control.start_hotkey(cfg.control.hotkey):
            self.log(f"{cfg.control.hotkey.upper()} toggles mute")
        if keep_awake():
            self.log("keeping Windows awake (closing the lid can still sleep it)")
        if self.debug:
            self.log("DEBUG: saving photos + clips to porch/debug/ -- never on Halloween night")

    def _refresh_brain(self) -> None:
        h = self.client.health()
        self.health = h
        self.brain_ok = bool(h and h.get("ready"))
        if self.brain_ok:
            try:
                v = self.fillers.sync(self.client)
                self.log(f"brain ready at {self.cfg.brain.url}; {self.fillers.count()} lines synced (v{v})")
            except Exception as e:
                self.log(f"filler sync failed: {e}")
        else:
            why = (h or {}).get("error") or ("warming up" if h and h.get("warming") else "not reachable")
            self.log(f"brain {why} at {self.cfg.brain.url}")
        if self.fillers.count() == 0 and self.fillers.load_latest():
            self.log(f"using cached lines v{self.fillers.version} ({self.fillers.count()})")
        if self.fillers.count() == 0:
            self.log("WARNING: no greeting/fallback lines yet - start the brain once so they sync")

    def _health_loop(self) -> None:
        while not self.stop.wait(self.cfg.brain.health_interval_s):
            h = self.client.health()
            ok = bool(h and h.get("ready"))
            if ok != self.brain_ok:
                self.log(f"brain {'back' if ok else 'DOWN - fallback lines only'}")
            self.health, self.brain_ok = h, ok
            if ok and h.get("fillers_version") and h["fillers_version"] != self.fillers.version:
                try:
                    self.fillers.sync(self.client)
                    self.log(f"lines updated to v{self.fillers.version}")
                except Exception as e:
                    self.log(f"filler sync failed: {e}")

    # -- camera side ----------------------------------------------------------------
    def _detect_loop(self) -> None:
        period = 1.0 / max(0.5, self.cfg.detector.fps)
        while not self.stop.is_set():
            t0 = time.monotonic()
            frame, ft = self.camera.latest()
            if frame is not None and t0 - ft < 2.0:
                boxes = self.detector.detect(frame)
                self.det_ms.append((time.monotonic() - t0) * 1000)
                score = len(boxes) * 1e7 + sum(b.area for b in boxes)
                with self.lock:
                    self.people_hist.append((t0, len(boxes)))
                    self.frames.append((t0, frame, score))
                    self.last_boxes = boxes
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def people_now(self, now: float) -> int:
        with self.lock:
            while self.people_hist and now - self.people_hist[0][0] > 1.0:
                self.people_hist.popleft()
            counts = [c for _, c in self.people_hist]
        return int(statistics.median_low(counts)) if counts else 0

    def best_frame(self, since: float):
        with self.lock:
            cands = [(s, t, f) for t, f, s in self.frames if t >= since - 0.5]
            if not cands and self.frames:
                t, f, s = self.frames[-1]
                return f
        return max(cands, key=lambda c: (c[0], c[1]))[2] if cands else None

    # -- the loop -------------------------------------------------------------------
    def run(self) -> None:
        self.start()
        self.log("porch running - Ctrl+C to stop")
        try:
            while not self.stop.is_set():
                now = time.monotonic()
                events = self.mic.poll()
                speech_started = any(e.kind == "start" for e in events)
                speech_dropped = any(e.kind == "noise" for e in events)
                utt = next((e for e in reversed(events) if e.kind == "end"), None)
                try:
                    done = self.results.get_nowait()
                except queue.Empty:
                    done = None
                muted = self.control.muted
                busy = self.player.busy(now, margin=self.cfg.audio.tail_margin_s)
                self.mic.set_open(not busy and not muted)
                cam_ok = self.camera is not None and self.camera.ok(now)
                obs = Obs(now=now, people=self.people_now(now) if cam_ok else 0, camera_ok=cam_ok,
                          audio_busy=busy, speech_started=speech_started, speech_dropped=speech_dropped,
                          utterance=utt.audio if utt else None, turn_done=done,
                          brain_ok=self.brain_ok, muted=muted)
                prev = self.fsm.state
                actions = self.fsm.tick(obs)
                if self.fsm.state != prev:
                    self.log(f"{prev.value:>8} -> {self.fsm.state.value:<8} people={obs.people}")
                for a in actions:
                    self._execute(a, now, utt)
                self._speaker_watch(now, busy)
                if self.preview:
                    self._show_preview(obs)
                time.sleep(max(0.0, TICK_S - (time.monotonic() - now)))
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _execute(self, a, now: float, utt) -> None:
        if isinstance(a, Say):
            x = self.fillers.pick(a.category)
            if x is None:
                self.log(f"(no '{a.category}' line cached)")
                return
            self.player.play(x[0], x[1], tag=a.category)
            self.log(f'say [{a.category}] "{x[2]}"')
        elif isinstance(a, PlayReply):
            o = a.outcome
            self.player.play(o.audio, o.sr, tag="reply")
            self.last_reply = o.reply_text
            self.turns_total += 1
            m = self.marks.pop(o.turn_id, {})
            row = {"t": datetime.now().isoformat(timespec="seconds"), "turn": o.turn_id, "source": o.source,
                   "brain_ms": o.timings, "roundtrip_ms": round((now - m.get("t_submit", now)) * 1000)}
            if m.get("t_speech_end"):
                row["speech_end_to_reply_ms"] = round((now - m["t_speech_end"]) * 1000)
                self.lat_ms.append(row["speech_end_to_reply_ms"])
            if self.debug:
                row["reply"] = o.reply_text
                row["description"] = o.description
            self._log_turn(row)
            self.log(f'reply ({o.source}, brain {o.timings.get("total", "?")} ms, '
                     f'round trip {row["roundtrip_ms"]} ms): "{o.reply_text}"')
        elif isinstance(a, SubmitTurn):
            jpeg = None
            if a.with_photo and self.camera is not None:
                f = self.best_frame(a.photo_since)
                if f is not None:
                    jpeg = self.camera.jpeg(f)
            wav = float_to_wav(a.clip, 16000) if a.clip is not None else None
            t_end = (utt.t - self.cfg.vad.end_silence_ms / 1000) if (utt and a.clip is not None) else None
            self.marks[a.turn_id] = {"t_submit": now, "t_speech_end": t_end}
            if self.debug:
                self._save_debug(a.turn_id, jpeg, wav)
            speech = f"{len(a.clip) / 16000:.1f}s" if a.clip is not None else "none"
            self.log(f"turn {a.turn_id}: photo={'yes' if jpeg else 'no'} speech={speech}")
            threading.Thread(target=self._turn_worker, args=(a, jpeg, wav, now), daemon=True).start()
        elif isinstance(a, CancelTurn):
            self.cancelled.add(a.turn_id)
            self.log(f"turn {a.turn_id} timed out - fallback line")
        elif isinstance(a, Flush):
            self.player.flush()

    def _turn_worker(self, a: SubmitTurn, jpeg, wav, t_submit: float) -> None:
        meta = {"turn_id": str(a.turn_id), "group_size": a.group_size, "turn_index": a.turn_index,
                "costume_hint": a.costume_hint, "debug": self.debug}
        try:
            r = self.client.turn(meta, jpeg, wav)
            audio, sr = wav_to_float(base64.b64decode(r["audio"]["b64"]))
            out = TurnOutcome(a.turn_id, True, audio, sr, r.get("description"), r.get("reply_text", ""),
                              r.get("source", ""), r.get("timings_ms", {}))
            if self.debug and r.get("transcript") is not None:
                self.log(f'heard: "{r["transcript"]}" | saw: "{r.get("description")}"')
        except Exception as e:
            self.log(f"turn {a.turn_id} failed: {e}")
            out = TurnOutcome(a.turn_id, False)
        if a.turn_id in self.cancelled:
            return
        self.results.put(out)

    # -- helpers --------------------------------------------------------------------
    def _speaker_watch(self, now: float, busy: bool) -> None:
        """Every 5 s while quiet: reopen a dead output stream, or switch back to the real speaker."""
        if busy or now - self.spk_checked < 5.0:
            return
        self.spk_checked = now
        dead = not self.player.stream.active
        if not (dead or self.spk_fallback):
            return
        try:
            dev, fallback = pick_speaker(self.cfg.audio.speaker, self.cfg.audio.host_api)
        except Exception as e:
            self.log(f"no speaker at all: {e}")
            return
        if not dead and fallback:
            return  # still waiting for the real speaker
        gain = self.player.gain_db
        try:
            self.player.close()
        except Exception:
            pass
        self.player = Player(dev, self.cfg.audio.keepalive_dbfs, gain)
        self.player.start()
        self.spk_fallback = fallback
        self.log(f"speaker {'reopened' if dead else 'back'}: {dev['name']}")

    def _log_turn(self, row: dict) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass

    def _save_debug(self, tid: int, jpeg: bytes | None, wav: bytes | None) -> None:
        d = PORCH_DIR / "debug"
        d.mkdir(parents=True, exist_ok=True)
        base = d / f"{datetime.now():%Y%m%d-%H%M%S}-turn{tid}"
        if jpeg:
            base.with_suffix(".jpg").write_bytes(jpeg)
        if wav:
            base.with_suffix(".wav").write_bytes(wav)

    def status(self) -> dict:
        ac, pct = power_status()
        lat = sorted(self.lat_ms)
        return {"state": self.fsm.state.value, "muted": self.control.muted,
                "people": self.people_now(time.monotonic()) if self.camera else None,
                "brain_ok": self.brain_ok, "gain_db": round(self.player.gain_db, 1),
                "last_reply": self.last_reply, "turns": self.turns_total,
                "p50_ms": lat[len(lat) // 2] if lat else None,
                "power": ("mains" if ac else "BATTERY") + (f" {pct}%" if pct is not None else "")}

    def test_line(self) -> None:
        x = self.fillers.pick("test")
        if x:
            self.player.play(x[0], x[1], tag="test")

    def change_gain(self, d: float) -> None:
        self.player.gain_db = max(-24.0, min(12.0, self.player.gain_db + d))
        self.log(f"volume {self.player.gain_db:+.0f} dB")

    def _show_preview(self, obs: Obs) -> None:
        import cv2

        frame, _ = self.camera.latest() if self.camera else (None, 0)
        if frame is None:
            return
        img = frame.copy()
        h, w = img.shape[:2]
        x0, y0, x1, y1 = self.cfg.detector.roi
        cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h) - 1), (255, 200, 0), 2)
        with self.lock:
            boxes = list(self.last_boxes)
        for b in boxes:
            cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), (0, 220, 0), 2)
        txt = f"{self.fsm.state.value}  people={obs.people}  mic={self.mic.level_db:.0f}dB  vad={self.mic.vad.threshold_db:.0f}dB"
        cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("porch skeleton (q to stop)", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop.set()

    def shutdown(self) -> None:
        self.stop.set()
        self.log("stopping")
        for fn in (lambda: self.control.stop(), lambda: self.mic.close(), lambda: self.player.close(),
                   lambda: self.camera and self.camera.close(), allow_sleep):
            try:
                fn()
            except Exception:
                pass
        if self.preview:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
