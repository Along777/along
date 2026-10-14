# NOTES_FOR_AI.md — handoff for the next agent

Read this before changing anything. It records decisions, their reasons, the traps already hit,
and what is verified vs. not. Deadline: **Halloween, Sat 2026-10-31.**

## 0. Machines

- **Porch** = Aaron's HP Victus 15 laptop, Windows 11, Ryzen 5 7535HS, **7.2 GB RAM**, AMD GPU (unused).
  Devices: **Arducam 1080P Low Light** (DirectShow index 1; the built-in HP camera is index 0),
  lav mic **"Microphone (USB PnP Audio Device)"**, Bluetooth speaker **Anker SoundCore 2**
  (A2DP endpoint "Headphones (SoundCore 2)"; it also has an aux-in jack).
- **Brain** = Linux workstation, Ryzen 9 9950X3D, RTX 5090 32 GB, Ubuntu. Aaron is new to Linux.
  Reached by `git pull` (and SSH, planned in Phase 2). It also hosts the `local_llm` kit
  (`C:\Users\ccvar\ai\local_llm`), whose setup pins `OLLAMA_MAX_LOADED_MODELS=1`.

## 1. Decisions (and why)

1. **Echo fix vs. the guide.** The guide's pseudocode plays a filler *while* recording, so speech-to-text
   would transcribe the skeleton. Order is greet (mic gated) -> listen -> POST -> thinking filler only
   after 0.8 s -> reply. The app gates the mic whenever **anything** is playing plus `tail_margin_s`
   (`app.py`: `mic.set_open(not busy and not muted)`), so half-duplex holds by construction.
2. **One Ollama model for both jobs** (`qwen3-vl:8b-instruct` on the 5090, `2b-instruct` on the laptop).
   The plain `qwen3-vl:Nb` tags are the **thinking** variants — always use `-instruct`. One resident model
   sidesteps `MAX_LOADED_MODELS=1`. Every request sends `keep_alive=-1` and the **same `num_ctx`**
   (a different num_ctx reloads the model). `think` is only sent if `/api/show` lists the capability.
3. **No torch in the brain baseline.** Vision + reply run in Ollama; speech-to-text is Parakeet TDT 0.6B v3
   via `onnx-asr` (int8, CPU); voice is `kokoro-onnx` (int8, CPU). The PyTorch `kokoro` package needs
   Python < 3.13 and its English front end (misaki) pip-installs a spaCy model at runtime, which fails in
   a uv venv. Chatterbox-Turbo (cloning) is the Phase-2 upgrade: it pins `torch==2.6` (no RTX 5090 kernels)
   and ignores `exaggeration`, so it needs a uv override to torch 2.8.0+cu128 or its own venv.
4. **Python 3.12 via uv** on both machines (`.python-version`, `uv.lock`). kokoro-onnx needs < 3.14.
5. **Voice-activity endpointing** instead of a fixed 4 s clip (`audio.EnergyVad`): 1 s warm-up to learn the
   room, start after 120 ms loud, end after 700 ms quiet, 5 s cap, 300 ms pre-roll, and an utterance needs
   200 ms of loud audio in total or it is reported as `noise` (bangs, gusts).
6. **Group logic** lives in the pure `statemachine.Fsm` (no I/O, injected clock, fully unit-tested): greet once
   per arrival, follow-ups skip vision (`costume_hint`), re-arm after 3 s empty, bigger wave = new greeting,
   `still_here` after 60 s, `max_turns` per group, MUTED overrides everything.
7. **Safety net** (`textfilter.py`): strip stage directions/emoji, expand contractions, clamp 40 words at a
   sentence boundary, blocklist + name/school/address/"come closer"/AI/camera -> canned `deflect` line.
   Persona text is the guide's prompt verbatim **plus one line** (no guessing gender) — Aaron approved via the plan.
8. **Privacy:** nothing about visitors is written to disk unless `--debug`; turn logs hold timings only;
   Ultralytics runs with `YOLO_OFFLINE=1`; the brain runs with `HF_HUB_OFFLINE=1`.
   The repo is public and Pages publishes everything — run `scripts/check_public.ps1` before every push.

## 2. Traps already hit (don't re-learn them)

- **The Claude desktop app is an MSIX package: its shells' writes to `%APPDATA%`/`%LOCALAPPDATA%` are
  virtualized** into `...\Packages\Claude_*\LocalCache`. uv's managed Python landed there and its junction
  pointed at a path that doesn't exist outside the app. Fix used everywhere: `UV_PYTHON_INSTALL_DIR=~\.uv\python`
  and `UV_CACHE_DIR=~\.uv\cache` (the porch scripts set them). Anything Aaron must see from his own terminal
  must not live under AppData.
- uv was installed by winget at
  `%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe`;
  shells started before the install don't have it on PATH (scripts search for it).
- Ultralytics falls back to `\tmp\Ultralytics` if `YOLO_CONFIG_DIR` doesn't exist — `detector.py` creates it first.
- `Grep`/ripgrep skips `.venv` (gitignored); read installed packages with `Select-String`.
- Windows PowerShell mangles double quotes in `python -c` here-strings: write a script file instead.
- Commit messages with `->`/`<`/`>` in a PowerShell here-string break; use `git commit -F <file>`.
- `.ps1` files must stay ASCII-only (PowerShell 5.1 reads BOM-less files as ANSI).
- The SoundCore 2 shows a 16 kHz **"Headset (SoundCore 2)"** input — its hands-free mic. Never open it:
  Windows would drop the speaker to phone-call quality. Config pins the mic to "USB PnP".

## 3. The workstation as set up (2026-10-13)

This repo is public: host names, usernames, addresses and network details live only in the machines'
gitignored `config.local.toml` files and the operator's private notes, never here.

- RTX 5090 workstation, NVIDIA driver 595.x, Ollama 0.32.x. Repo at `~/ai/along`, branch `ai-skeleton-build`.
- The brain serves on a non-default port set in its `brain/config.local.toml`, because another local LLM web
  UI already owns 8080 there. That UI shares the same Ollama — nobody uses it on Halloween night.
- VPN clients run on the workstation; they do not block LAN traffic to the brain or SSH.
- The laptop reaches the brain through an SSH alias in its own `~/.ssh/config`. In PowerShell, `$(...)` inside a
  double-quoted remote command is evaluated locally — run SSH from bash with single quotes.
- Deploy: push, then `ssh <brain> 'cd ~/ai/along && git pull --ff-only && systemctl --user restart skeleton-brain'`.

## 4. Verified vs. not

Verified (2026-10-13):
- Unit tests: brain 17, porch 22.
- Laptop: real Parakeet int8 transcribes word-perfectly (~0.3 s). Kokoro int8 takes ~2.4 s/line on the laptop CPU.
  YOLO26n runs at 28 ms/frame at 416 px. The voice sample plays on the SoundCore.
- Laptop Ollama path with `qwen3-vl:2b-instruct`: caps, describe (0.8 s warm), reply (~1 s).
- **5090 end to end from the laptop:** STT 87 ms, reply 349 ms, TTS 871 ms. Brain total 1.3 s, round trip 1.5 s.
  `qwen3-vl:8b-instruct` sits 100% in VRAM (5.5 GB).
- **Persona eval on the 5090:**
  - Before tuning, the model roasted costumes ("seen better costumes... never worse") and invited a kid
    to "come on in". The contraction expander also produced "it is seen".
  - After tuning: 36/40 were clean as spoken, 3 were caught by the filter (canned line), 1 was a false flag (fixed).

Not yet verified:
- **The Arducam is unreliable.** After some opens it sticks in YUY2 at 1 fps with all-black frames (in every
  mode, even the default). Only a USB re-plug brought it back. `camera.py` now reopens a black or stalled stream,
  but a stuck device may still need the re-plug. Suspect USB selective suspend; try a port directly on the laptop.
- Mic levels with real speech at kid distance.
- The full `run` loop with a person in view.
- Speaker fallback/return when the SoundCore drops out. Seen tonight: the SoundCore went to "Unknown" when it
  turned itself off.
