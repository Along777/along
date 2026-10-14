"""The porch dashboard and kill switch, served on the LAN (the laptop sits in a tote outside), plus F12.

GET  /            dashboard (dashboard.html)        GET /status     everything, as JSON
GET  /frame.jpg   live camera frame with boxes      POST /mute  /unmute  /test  /poke  /gain?d=3  /say?cat=greet
The camera frame is served from memory only and never saved.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PAGE = (Path(__file__).with_name("dashboard.html")).read_text(encoding="utf-8")
SAY_OK = {"greet", "thinking", "fallback", "no_hear", "still_here", "test"}


class Control:
    def __init__(self, port: int, app, log=print):
        self.port = port
        self.app = app
        self.muted = False
        self.lock = threading.Lock()
        self.log = log
        self.server: ThreadingHTTPServer | None = None
        self.listener = None

    def set_muted(self, on: bool, why: str) -> None:
        with self.lock:
            if self.muted != on:
                self.muted = on
                self.log(f"{'MUTED' if on else 'unmuted'} ({why})")

    def toggle(self, why: str) -> None:
        self.set_muted(not self.muted, why)

    def start_http(self) -> None:
        ctl = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlparse(self.path).path
                try:
                    if path == "/":
                        self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                    elif path == "/status":
                        self._send(200, json.dumps(ctl.app.status(), default=str).encode(), "application/json")
                    elif path == "/frame.jpg":
                        jpg = ctl.app.frame_jpeg()
                        if jpg:
                            self._send(200, jpg, "image/jpeg")
                        else:
                            self._send(204, b"", "image/jpeg")
                    else:
                        self._send(404, b"not found", "text/plain")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:  # never let a dashboard bug touch the porch loop
                    self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

            def do_POST(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if u.path == "/mute":
                    ctl.set_muted(True, "dashboard")
                elif u.path == "/unmute":
                    ctl.set_muted(False, "dashboard")
                elif u.path == "/test":
                    ctl.app.say("test")
                elif u.path == "/poke":
                    ctl.app.poke()
                elif u.path == "/say":
                    cat = q.get("cat", [""])[0]
                    if cat not in SAY_OK:
                        self._send(400, b'{"error":"unknown category"}', "application/json")
                        return
                    ctl.app.say(cat)
                elif u.path == "/gain":
                    try:
                        d = float(q.get("d", ["0"])[0])
                    except ValueError:
                        d = 0.0
                    ctl.app.change_gain(max(-6.0, min(6.0, d)))
                else:
                    self._send(404, b"not found", "text/plain")
                    return
                self._send(200, b'{"ok":true}', "application/json")

        self.server = ThreadingHTTPServer(("0.0.0.0", self.port), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, name="control-http", daemon=True).start()

    def start_hotkey(self, key_name: str) -> bool:
        try:
            from pynput import keyboard
        except Exception as e:
            self.log(f"hotkey unavailable: {e}")
            return False
        target = getattr(keyboard.Key, key_name.lower(), None)
        if target is None:
            self.log(f"unknown hotkey {key_name!r}")
            return False

        def on_press(k):  # must not block (Windows drops slow hooks)
            if k == target:
                threading.Thread(target=self.toggle, args=(key_name.upper(),), daemon=True).start()

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()
        return True

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
        if self.listener:
            self.listener.stop()
