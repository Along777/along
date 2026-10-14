"""The HTTP contract, end to end, with fake engines (no models, runs anywhere)."""
import base64
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from skeleton_brain import config as config_mod
from skeleton_brain.audiofx import from_wav, to_wav
from skeleton_brain.server import TEST_IMAGE, create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    cfg = config_mod.load()
    app = create_app(cfg, fake=True, cache_root=tmp_path_factory.mktemp("fillers"))  # not the real cache
    with TestClient(app) as c:
        state = app.state.skeleton
        for _ in range(200):
            if state.ready or state.error:
                break
            time.sleep(0.05)
        assert state.ready, state.error
        yield c


def speech_wav(seconds=1.0, sr=16000):
    t = np.arange(int(sr * seconds)) / sr
    return to_wav(0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32), sr)


def test_health(client):
    h = client.get("/health").json()
    assert h["ready"] and h["ok"] and h["fake"]
    assert h["fillers_version"]


def test_turn_with_photo_and_clip(client):
    photo = TEST_IMAGE.read_bytes() if TEST_IMAGE.is_file() else b"\xff\xd8\xff\xe0" + b"0" * 100
    r = client.post("/turn", data={"meta": json.dumps({"turn_id": "t1", "group_size": 2})},
                    files={"photo": ("p.jpg", photo, "image/jpeg"), "clip": ("c.wav", speech_wav(), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["turn_id"] == "t1"
    assert body["heard"] is True
    assert body["transcript"] is None  # only returned in debug mode
    assert "candy" in body["reply_text"].lower()
    audio, sr = from_wav(base64.b64decode(body["audio"]["b64"]))
    assert sr == body["audio"]["sample_rate"] and audio.size > 0
    assert set(body["timings_ms"]) >= {"vision", "stt", "llm", "tts", "total"}


def test_turn_photo_only_and_follow_up_hint(client):
    r = client.post("/turn", data={"meta": json.dumps({"turn_index": 1, "costume_hint": "a pirate"})})
    assert r.status_code == 200
    body = r.json()
    assert body["heard"] is False and body["description"] == "a pirate"


def test_no_photo_means_no_costume_talk():
    from skeleton_brain.pipeline import TurnMeta, build_user_message

    msg = build_user_message("", "hello, how are you?", TurnMeta())
    assert "Costumes:" not in msg and "cannot see" in msg and "hello, how are you?" in msg
    assert build_user_message("a pirate", "", TurnMeta()).startswith("Costumes: a pirate")


def test_group_memory_reaches_the_prompt():
    from skeleton_brain.pipeline import GroupMemory, TurnMeta, build_user_message

    mem = GroupMemory()
    mem.add("g1", "trick or treat", "Here is your candy, mortal.")
    mem.add("g1", "do you like my shirt", "Very practical.")
    mem.add("g1", "what?", "Nothing, friend.")
    assert len(mem.get("g1")) == 2 and mem.get("g2") == [] and mem.get("") == []
    msg = build_user_message("a pirate", "that makes no sense", TurnMeta(turn_index=3), mem.get("g1"))
    assert "Very practical." in msg and "Do not repeat" in msg and "Here is your candy" not in msg


def test_rejects_bad_inputs(client):
    assert client.post("/turn", data={"meta": "{not json"}).status_code == 400
    r = client.post("/turn", files={"photo": ("p.jpg", b"GIF89a....", "image/jpeg")})
    assert r.json()["error"] == "bad_photo"
    r = client.post("/turn", files={"clip": ("c.wav", b"RIFF....", "audio/wav")})
    assert r.json()["error"] == "bad_clip"
    r = client.post("/turn", files={"clip": ("c.wav", speech_wav(seconds=8), "audio/wav")})
    assert r.json()["error"] == "too_large"


def test_fillers_manifest_and_download(client):
    mf = client.get("/fillers").json()
    cats = {it["category"] for it in mf["items"]}
    assert {"greet", "thinking", "fallback", "test"} <= cats
    first = mf["items"][0]
    r = client.get(f"/fillers/{first['id']}.wav")
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    assert client.get("/fillers/nope.wav").status_code == 404


def test_warm(client):
    photo = TEST_IMAGE.read_bytes() if TEST_IMAGE.is_file() else b"\xff\xd8\xff\xe0" + b"0" * 100
    r = client.post("/warm", files={"photo": ("w.jpg", photo, "image/jpeg")})
    assert r.status_code == 200 and r.json()["ok"]
    assert client.post("/warm", files={"photo": ("w.jpg", b"GIF89a", "image/jpeg")}).status_code == 400


def test_speak(client):
    r = client.post("/speak", json={"text": "Testing, testing."})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
