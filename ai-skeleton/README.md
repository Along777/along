# AI porch skeleton — the code

The build guide is [index.html](index.html). This folder is the software that makes it talk.

```
porch laptop (Windows)                         brain (Linux, RTX 5090)
camera -> person detector (YOLO26n, CPU)       POST /turn  {photo, speech clip}
greet -> listen -> POST /turn  ───────────▶     costume line ∥ speech-to-text
play reply  ◀─────────────── WAV + text ───     reply (Ollama qwen3-vl:8b-instruct) -> voice (Kokoro)
```

Kid photos and audio never leave the house network and are never written to disk
(unless you run the porch with `--debug` for a living-room test).

## Folders

| | |
|---|---|
| `porch/` | runs on the laptop: camera, detector, mic, speaker, state machine, phone kill switch |
| `brain/` | runs on the workstation: vision, speech-to-text, reply, voice, pre-rendered lines |
| `persona/` | the skeleton's prompt, the vision prompt, all pre-rendered lines, the blocklist |
| `scripts/` | setup + run scripts for both machines, and `check_public.ps1` (run before every push) |

## Porch laptop (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_porch.ps1 -BrainUrl http://<brain-ip>:8080
powershell -ExecutionPolicy Bypass -File scripts\run_porch.ps1
```

Useful commands (`uv run --project porch skeleton-porch <command>`):

| command | what it does |
|---|---|
| `devices` | cameras, mics, speakers; `->` marks what config picked |
| `doctor` | camera fps + brightness, detector speed, mic levels, speaker, brain, power |
| `preview` | camera with person boxes and the watch zone, to aim the camera |
| `selftest-turn` | one full turn through the brain with a live photo; prints timings |
| `run --debug --preview` | the real loop, plus captures + what it heard (living-room tests only) |
| `stats` | reply latency from `porch/logs/turns.jsonl` |
| `record-voice` | record a voice reference for cloning (Phase 2) |

Kill switch: **F12** on the laptop, or the phone page at `http://<laptop-ip>:8081`
(mute, unmute, test line, volume). The Wi-Fi must be a **Private** network for the phone page.

## Brain (Linux workstation)

```bash
git clone https://github.com/Along777/along.git ~/ai/along && cd ~/ai/along/ai-skeleton
bash scripts/setup_brain.sh            # idempotent: uv, Python, Ollama model, voice models, service, doctor
systemctl --user start skeleton-brain  # or run in a terminal: bash scripts/run_brain.sh
journalctl --user -u skeleton-brain -f
```

Useful commands (`uv run --project brain skeleton-brain <command>`): `doctor` (every stage with
timings), `eval` (40 scripted porch moments checked against the persona rules), `serve --fake`
(no models; for porch development), `make-fillers`.

Per-machine settings go in `porch/config.local.toml` and `brain/config.local.toml` (gitignored);
they override only the keys you write.

## Porch ↔ brain API

| | |
|---|---|
| `GET /health` | ready/warming/error, engines, Ollama models loaded, p50/p90 brain time |
| `POST /turn` | multipart: `meta` JSON (`turn_id`, `group_size`, `turn_index`, `costume_hint`), optional `photo` (JPEG ≤1 MB), optional `clip` (WAV 16 kHz mono ≤6 s). Returns `reply_text`, `source` (`llm` or `canned_*`), `description`, `heard`, `audio.b64` (WAV), `timings_ms`. 503 = busy or warming. |
| `GET /fillers`, `GET /fillers/{id}.wav` | the pre-rendered greet / thinking / fallback lines the porch caches |
| `POST /speak {"text"}` | any line in the skeleton voice (tests, volume walk) |

## Tests

```bash
uv run --directory brain pytest -q     # text filter + the HTTP contract with fake engines
uv run --directory porch pytest -q     # state machine + voice activity detector
```
