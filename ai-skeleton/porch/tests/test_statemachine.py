import numpy as np

from skeleton_porch.config import Behavior
from skeleton_porch.statemachine import (CancelTurn, Flush, Fsm, Obs, PlayReply, S, Say, SubmitTurn,
                                         TurnOutcome)

CFG = Behavior()  # arrive 0.5, rearm 3, cooldown 8, thinking 0.8, max_turns 4, still_here 60
UTT = np.ones(16000, np.float32)


class Rig:
    """Drive the FSM through time like the app does (20 Hz ticks)."""

    def __init__(self, **kw):
        self.fsm = Fsm(CFG, no_speech_s=3.0, turn_timeout_s=7.0)
        self.t = 0.0
        self.people = 0
        self.busy = False
        self.brain_ok = True
        self.camera_ok = True
        self.muted = False
        self.log: list = []

    def tick(self, dt=0.05, **kw):
        self.t += dt
        o = Obs(now=self.t, people=self.people, camera_ok=self.camera_ok, audio_busy=self.busy,
                brain_ok=self.brain_ok, muted=self.muted, **kw)
        acts = self.fsm.tick(o)
        self.log += acts
        return acts

    def run(self, seconds, **kw):
        acts = []
        for _ in range(int(round(seconds / 0.05))):
            acts += self.tick(**kw)
        return acts

    def to_listen(self):
        self.people = 2
        acts = self.run(0.6)
        assert any(isinstance(a, Say) and a.category == "greet" for a in acts)
        assert self.fsm.state == S.GREET
        self.busy = True
        self.run(1.0)
        assert self.fsm.state == S.GREET  # still playing
        self.busy = False
        self.tick()
        assert self.fsm.state == S.LISTEN

    def submit(self):
        acts = self.tick(utterance=UTT)
        sub = [a for a in acts if isinstance(a, SubmitTurn)]
        assert len(sub) == 1 and self.fsm.state == S.THINK
        return sub[0]


def test_greets_only_after_person_stays():
    r = Rig()
    r.people = 1
    assert not r.run(0.3)
    assert r.fsm.state == S.IDLE
    acts = r.run(0.3)
    assert [a.category for a in acts if isinstance(a, Say)] == ["greet"]


def test_flicker_does_not_greet():
    r = Rig()
    for _ in range(10):
        r.people = 1
        r.run(0.2)
        r.people = 0
        r.run(0.1)
    assert r.fsm.state == S.IDLE and not r.log


def test_full_turn_with_thinking_filler_then_follow_up():
    r = Rig()
    r.to_listen()
    sub = r.submit()
    assert sub.with_photo and sub.turn_index == 0 and sub.group_size == 2 and sub.clip is not None
    acts = r.run(CFG.thinking_after_s + 0.1)
    assert [a.category for a in acts if isinstance(a, Say)] == ["thinking"]
    assert not [a for a in r.run(1.0) if isinstance(a, Say)]  # only once
    acts = r.tick(turn_done=TurnOutcome(sub.turn_id, True, UTT, 24000, "a pirate", "Arr."))
    assert any(isinstance(a, PlayReply) for a in acts) and r.fsm.state == S.SPEAK
    r.busy = True
    r.run(1.0)
    r.busy = False
    r.tick()
    assert r.fsm.state == S.LISTEN  # follow-up window
    sub2 = r.submit()
    assert not sub2.with_photo and sub2.turn_index == 1 and sub2.costume_hint == "a pirate"


def test_silence_gets_costume_comment_on_first_turn():
    r = Rig()
    r.to_listen()
    acts = r.run(3.1)
    sub = [a for a in acts if isinstance(a, SubmitTurn)]
    assert len(sub) == 1 and sub[0].with_photo and sub[0].clip is None


def test_silent_follow_up_ends_in_cooldown_then_linger_then_idle():
    r = Rig()
    r.to_listen()
    sub = r.submit()
    r.tick(turn_done=TurnOutcome(sub.turn_id, True, UTT, 24000, "unclear", "Hello."))
    r.tick()  # audio not busy -> follow-up listen
    assert r.fsm.state == S.LISTEN
    r.run(3.1)
    assert r.fsm.state == S.COOLDOWN
    r.run(8.1)
    assert r.fsm.state == S.LINGER
    r.people = 0
    r.run(3.2)
    assert r.fsm.state == S.IDLE and r.fsm.armed


def test_timeout_plays_fallback_and_cancels():
    r = Rig()
    r.to_listen()
    sub = r.submit()
    acts = r.run(7.1)
    assert any(isinstance(a, CancelTurn) and a.turn_id == sub.turn_id for a in acts)
    assert any(isinstance(a, Say) and a.category == "fallback" for a in acts)
    late = r.tick(turn_done=TurnOutcome(sub.turn_id, True, UTT))
    assert not any(isinstance(a, PlayReply) for a in late)


def test_stale_result_is_ignored():
    r = Rig()
    r.to_listen()
    sub = r.submit()
    acts = r.tick(turn_done=TurnOutcome(sub.turn_id + 99, True, UTT))
    assert not any(isinstance(a, PlayReply) for a in acts) and r.fsm.state == S.THINK


def test_failed_turn_plays_fallback():
    r = Rig()
    r.to_listen()
    sub = r.submit()
    acts = r.tick(turn_done=TurnOutcome(sub.turn_id, False))
    assert any(isinstance(a, Say) and a.category == "fallback" for a in acts)


def test_brain_down_greets_then_fallback_without_posting():
    r = Rig()
    r.brain_ok = False
    r.people = 1
    r.run(0.6)
    r.busy = False
    acts = r.run(0.2)
    assert any(isinstance(a, Say) and a.category == "fallback" for a in acts)
    assert not any(isinstance(a, SubmitTurn) for a in r.log)


def test_mute_flushes_and_cancels_then_unmute_rearms():
    r = Rig()
    r.to_listen()
    sub = r.submit()
    r.muted = True
    acts = r.tick()
    assert any(isinstance(a, Flush) for a in acts)
    assert any(isinstance(a, CancelTurn) and a.turn_id == sub.turn_id for a in acts)
    assert r.fsm.state == S.MUTED
    assert not r.run(5.0)  # silent while muted
    r.muted = False
    acts = r.run(0.6)
    assert any(isinstance(a, Say) and a.category == "greet" for a in acts)  # people still there: greets again


def test_noise_burst_does_not_block_the_no_speech_path():
    r = Rig()
    r.to_listen()
    r.tick(speech_started=True)
    r.tick(speech_dropped=True)  # it was a car door
    acts = r.run(3.1)
    sub = [a for a in acts if isinstance(a, SubmitTurn)]
    assert len(sub) == 1 and sub[0].clip is None  # still gets its costume comment


def test_max_turns_then_quiet():
    r = Rig()
    r.to_listen()
    for i in range(CFG.max_turns):
        if r.fsm.state != S.LISTEN:
            r.tick(speech_started=True)
        sub = r.submit()
        r.tick(turn_done=TurnOutcome(sub.turn_id, True, UTT, 24000, "a ghost", "Boo."))
        r.tick()
    assert r.fsm.state == S.COOLDOWN
    r.run(8.1)
    assert r.fsm.state == S.LINGER
    r.tick(speech_started=True)
    assert r.fsm.state == S.LINGER  # out of turns: ignores speech


def test_new_wave_gets_greeted():
    r = Rig()
    r.to_listen()
    r.run(3.1)  # nobody spoke -> costume comment turn
    sub = [a for a in r.log if isinstance(a, SubmitTurn)][-1]
    r.tick(turn_done=TurnOutcome(sub.turn_id, True, UTT))
    r.tick()
    r.run(3.1)
    r.run(8.1)
    assert r.fsm.state == S.LINGER
    r.people = 4
    assert not any(isinstance(a, Say) for a in r.run(1.0))  # a short flicker is not a new wave
    acts = r.run(0.7)
    assert any(isinstance(a, Say) and a.category == "greet" for a in acts)


def test_still_here_after_a_minute():
    r = Rig()
    r.to_listen()
    r.run(3.1)
    sub = [a for a in r.log if isinstance(a, SubmitTurn)][-1]
    r.tick(turn_done=TurnOutcome(sub.turn_id, True, UTT))
    r.tick()
    acts = r.run(65)
    assert any(isinstance(a, Say) and a.category == "still_here" for a in acts)


def test_poke_greets_even_with_nobody_in_view():
    r = Rig()
    acts = r.tick(poke=True)
    assert [a.category for a in acts if isinstance(a, Say)] == ["greet"]
    r.tick()  # greeting done (rig audio not busy) -> listen
    assert r.fsm.state == S.LISTEN
    sub = r.submit()
    assert sub.with_photo and sub.group_size == 1


def test_camera_down_is_voice_triggered():
    r = Rig()
    r.camera_ok = False
    r.tick(speech_started=True)
    assert r.fsm.state == S.LISTEN
    sub = r.submit()
    assert not sub.with_photo and sub.clip is not None
