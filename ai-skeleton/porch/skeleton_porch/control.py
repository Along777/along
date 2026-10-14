"""The kill switch: a phone page on the LAN (the laptop sits in a tote outside) plus F12.

GET  /          the page            GET /status   JSON
POST /mute      POST /unmute        POST /test    POST /gain?d=3 (or -3)
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Skeleton</title>
<style>
:root{color-scheme:dark}body{margin:0;padding:16px;font:16px/1.4 system-ui,sans-serif;background:#111;color:#eee}
h1{font-size:20px;margin:0 0 12px}button{width:100%;padding:22px;margin:6px 0;font-size:22px;border:0;border-radius:12px;color:#fff}
#mute{background:#b3261e}#unmute{background:#1e7d32}.row{display:flex;gap:8px}.row button{background:#333;font-size:18px;padding:16px}
#st{background:#1c1c1c;border-radius:12px;padding:12px;margin-top:10px;white-space:pre-wrap;font-size:14px}
.big{font-size:26px;font-weight:700;margin:8px 0}
</style></head><body>
<h1>Porch skeleton</h1><div class="big" id="state">...</div>
<button id="mute" onclick="post('mute')">MUTE</button>
<button id="unmute" onclick="post('unmute')">Unmute</button>
<div class="row"><button onclick="post('gain?d=-3')">Vol -</button><button onclick="post('test')">Test line</button>
<button onclick="post('gain?d=3')">Vol +</button></div>
<div id="st">loading...</div>
<script>
async function post(p){try{await fetch('/'+p,{method:'POST'});}catch(e){}tick();}
async function tick(){try{const s=await (await fetch('/status')).json();
document.getElementById('state').textContent=s.muted?'MUTED':('live - '+s.state);
document.getElementById('state').style.color=s.muted?'#ff6b6b':'#7ee787';
document.getElementById('st').textContent=
'people: '+s.people+'\\nbrain: '+(s.brain_ok?'ok':'DOWN')+'\\nvolume: '+s.gain_db+' dB\\nlast reply: '+(s.last_reply||'-')+
'\\nturns tonight: '+s.turns+'   reply p50: '+(s.p50_ms??'-')+' ms\\npower: '+s.power;}
catch(e){document.getElementById('state').textContent='porch not answering';}}
tick();setInterval(tick,2000);
</script></body></html>"""


class Control:
    def __init__(self, port: int, status: Callable[[], dict], on_test: Callable[[], None],
                 on_gain: Callable[[float], None], log=print):
        self.port = port
        self.muted = False
        self.lock = threading.Lock()
        self.status, self.on_test, self.on_gain, self.log = status, on_test, on_gain, log
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
                if path == "/":
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/status":
                    self._send(200, json.dumps(ctl.status()).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                u = urlparse(self.path)
                if u.path == "/mute":
                    ctl.set_muted(True, "phone")
                elif u.path == "/unmute":
                    ctl.set_muted(False, "phone")
                elif u.path == "/test":
                    ctl.on_test()
                elif u.path == "/gain":
                    try:
                        d = float(parse_qs(u.query).get("d", ["0"])[0])
                    except ValueError:
                        d = 0.0
                    ctl.on_gain(max(-6.0, min(6.0, d)))
                else:
                    self._send(404, b"not found", "text/plain")
                    return
                self._send(200, b'{"ok":true}', "application/json")

        self.server = ThreadingHTTPServer(("0.0.0.0", self.port), H)
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
