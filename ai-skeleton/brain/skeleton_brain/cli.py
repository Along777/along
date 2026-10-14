"""skeleton-brain serve | doctor | eval | fetch-models | make-fillers"""
from __future__ import annotations

import argparse
import os
import sys

from . import config as config_mod


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="skeleton-brain", description="AI porch skeleton - the brain")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the HTTP server the porch talks to")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--fake", action="store_true", help="canned engines, no models (porch development)")
    d = sub.add_parser("doctor", help="check every stage on this machine, with timings")
    d.add_argument("--fake", action="store_true")
    d.add_argument("--turns", type=int, default=10)
    sub.add_parser("eval", help="run the scripted persona cases against the reply model")
    sub.add_parser("fetch-models", help="download the speech-to-text and voice models")
    sub.add_parser("make-fillers", help="render the greet/thinking/fallback lines now")
    a = p.parse_args(argv)
    cfg = config_mod.load()
    os.environ.setdefault("DO_NOT_TRACK", "1")
    if sys.platform == "win32":  # so the doctor's ok/warn colours render in old consoles
        os.system("")

    if a.cmd == "fetch-models":
        from .stt import fetch_parakeet
        from .tts import fetch_kokoro

        print(f"speech-to-text: {cfg.stt.model} ({cfg.stt.quantization}) -> {cfg.path(cfg.stt.model_dir)}")
        fetch_parakeet(cfg.stt.model, cfg.path(cfg.stt.model_dir), cfg.stt.quantization)
        print(f"voice: {cfg.path(cfg.tts.model).name} + {cfg.path(cfg.tts.voices).name}")
        fetch_kokoro(cfg.path(cfg.tts.model), cfg.path(cfg.tts.voices))
        print("done")
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never reach the internet at runtime

    if a.cmd == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(cfg, fake=a.fake), host=a.host or cfg.server.host, port=a.port or cfg.server.port,
                    log_level="info", workers=1)
        return 0
    if a.cmd == "doctor":
        from . import doctor

        return doctor.main(cfg, a.fake, a.turns)
    if a.cmd == "eval":
        from . import evalcases

        return evalcases.main(cfg)
    if a.cmd == "make-fillers":
        from .fillers import FillerBank, bank_version, load_lines
        from .server import BRAIN_DIR, build_engines

        stt, tts, fx = build_engines(cfg, fake=False)
        lines, text = load_lines(cfg.path(cfg.persona.dir))

        def speak(t):
            audio, sr = tts.synth(t)
            return fx(audio, sr), sr

        bank = FillerBank(BRAIN_DIR / "cache" / "fillers")
        mf = bank.load_or_render(bank_version(text, tts.name, cfg.fx), lines, speak)
        print(f"{len(mf['items'])} lines in {bank.dir}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
