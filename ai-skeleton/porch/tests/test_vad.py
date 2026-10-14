import numpy as np

from skeleton_porch.audio import EnergyVad
from skeleton_porch.config import Vad

SR = 16000
CFG = Vad()  # 30 ms frames, start 60 ms, end 700 ms, max 5 s, preroll 300 ms
F = SR * CFG.frame_ms // 1000


def tone(seconds, dbfs, f=220):
    t = np.arange(int(SR * seconds)) / SR
    amp = 10 ** (dbfs / 20) * np.sqrt(2)
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def noise(seconds, dbfs, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(SR * seconds)) * 10 ** (dbfs / 20)).astype(np.float32)


def feed(vad, x, t0=0.0):
    events = []
    for i in range(0, len(x) - F + 1, F):
        events += vad.feed(x[i:i + F], t0 + i / SR)
    return events


def test_silence_has_no_events():
    vad = EnergyVad(CFG)
    assert feed(vad, noise(3, -65)) == []


def test_utterance_start_and_end_with_preroll():
    vad = EnergyVad(CFG)
    x = np.concatenate([noise(1.0, -65), tone(1.0, -20) + noise(1.0, -65, 1), noise(1.5, -65, 2)])
    ev = feed(vad, x)
    kinds = [e.kind for e in ev]
    assert kinds == ["start", "end"]
    assert 0.9 < ev[0].t < 1.2
    dur = len(ev[1].audio) / SR
    assert 1.2 <= dur <= 1.7, dur  # speech + pre-roll + a short tail, not the whole silence


def test_click_is_not_speech():
    vad = EnergyVad(CFG)
    x = np.concatenate([noise(1.5, -65), tone(0.06, -10), noise(1.0, -65, 3)])
    assert feed(vad, x) == []


def test_short_bang_becomes_noise_not_an_utterance():
    vad = EnergyVad(CFG)
    x = np.concatenate([noise(1.5, -65), tone(0.15, -10), noise(1.0, -65, 4)])
    assert [e.kind for e in feed(vad, x)] == ["start", "noise"]


def test_long_talk_is_cut_at_max():
    vad = EnergyVad(CFG)
    ev = feed(vad, np.concatenate([noise(1.5, -65), tone(7.0, -20)]))
    ends = [e for e in ev if e.kind == "end"]
    assert ends and len(ends[0].audio) / SR <= CFG.max_utterance_s + CFG.preroll_ms / 1000 + 0.05


def test_threshold_adapts_to_steady_noise():
    vad = EnergyVad(CFG)
    assert feed(vad, noise(4, -42)) == []  # a noisy street is not speech
    assert vad.threshold_db > -40
    ev = feed(vad, np.concatenate([tone(1.0, -15) + noise(1.0, -42, 5), noise(1.5, -42, 6)]))
    assert [e.kind for e in ev] == ["start", "end"]


def test_reset_drops_partial_utterance():
    vad = EnergyVad(CFG)
    feed(vad, np.concatenate([noise(1.5, -65), tone(0.5, -20)]))
    assert vad.in_speech
    vad.reset()
    assert not vad.in_speech and not vad.pre
