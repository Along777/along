"""Porch behaviour as a pure state machine: tick(observations) -> actions.

No I/O and no clock of its own (time comes in through Obs.now), so every rule
here is unit-tested in tests/test_statemachine.py. The app executes the actions.

IDLE -> GREET -> LISTEN -> THINK -> SPEAK -> (LISTEN follow-up | COOLDOWN) -> LINGER -> IDLE
MUTED overrides everything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import Behavior


class S(str, Enum):
    IDLE = "idle"
    GREET = "greet"
    LISTEN = "listen"
    THINK = "think"
    SPEAK = "speak"
    COOLDOWN = "cooldown"
    LINGER = "linger"
    MUTED = "muted"


@dataclass
class TurnOutcome:
    turn_id: int
    ok: bool
    audio: Any = None          # np.ndarray mono float32
    sr: int = 24000
    description: str | None = None
    reply_text: str = ""
    source: str = ""
    timings: dict = field(default_factory=dict)


@dataclass
class Obs:
    now: float
    people: int = 0                 # smoothed count in the watch zone
    camera_ok: bool = True
    audio_busy: bool = False        # anything playing, incl. the tail margin
    speech_started: bool = False    # VAD saw speech begin since the last tick
    speech_dropped: bool = False    # ...but it turned out to be a bang or a gust, not speech
    utterance: Any = None           # VAD finished an utterance (16 kHz float32) since the last tick
    turn_done: TurnOutcome | None = None
    brain_ok: bool = True
    muted: bool = False
    poke: bool = False              # dashboard "Greet me now": act as if a group just arrived


# -- actions --------------------------------------------------------------------------
@dataclass
class Say:
    category: str                   # greet | thinking | fallback | no_hear | still_here | test


@dataclass
class PlayReply:
    outcome: TurnOutcome


@dataclass
class SubmitTurn:
    turn_id: int
    turn_index: int
    group_size: int
    with_photo: bool
    photo_since: float              # pick the best frame seen since this time
    clip: Any                       # np.ndarray or None
    costume_hint: str | None


@dataclass
class CancelTurn:
    turn_id: int


@dataclass
class Flush:
    pass


class Fsm:
    def __init__(self, cfg: Behavior, no_speech_s: float, turn_timeout_s: float):
        self.cfg = cfg
        self.no_speech_s = no_speech_s
        self.turn_timeout_s = turn_timeout_s
        self.state = S.IDLE
        self.since = 0.0
        self.armed = True
        self.present_since: float | None = None
        self.empty_since: float | None = None
        self.wave_since: float | None = None
        self.turns = 0
        self.group_size = 0
        self.group_started = 0.0
        self.costume: str | None = None
        self.last_activity = 0.0
        self.listen_started = 0.0
        self.speech_seen = False
        self.thinking_played = False
        self.pending: int | None = None
        self.next_id = 1

    # -- helpers --------------------------------------------------------------------
    def _go(self, st: S, now: float) -> None:
        self.state, self.since = st, now

    def _listen(self, now: float, speech_seen: bool = False) -> None:
        self._go(S.LISTEN, now)
        self.listen_started = now
        self.speech_seen = speech_seen

    def _new_group(self, now: float, people: int) -> None:
        self.armed = False
        self.turns = 0
        self.group_size = people
        self.group_started = now
        self.costume = None
        self.wave_since = None

    def _submit(self, o: Obs, clip, with_photo: bool) -> SubmitTurn:
        tid = self.next_id
        self.next_id += 1
        self.pending = tid
        self.thinking_played = False
        self._go(S.THINK, o.now)
        return SubmitTurn(tid, self.turns, self.group_size or o.people, with_photo, self.group_started,
                          clip, self.costume)

    # -- the rules ------------------------------------------------------------------
    def tick(self, o: Obs) -> list:
        cfg = self.cfg
        acts: list = []

        if o.people > 0:
            self.empty_since = None
            if self.present_since is None:
                self.present_since = o.now
        else:
            self.present_since = None
            if self.empty_since is None:
                self.empty_since = o.now
            if not self.armed and o.now - self.empty_since >= cfg.empty_rearm_s:
                self.armed = True  # the group has left
                self.turns = 0
                self.costume = None

        if o.muted:
            if self.state != S.MUTED:
                acts.append(Flush())
                if self.pending is not None:
                    acts.append(CancelTurn(self.pending))
                    self.pending = None
                self._go(S.MUTED, o.now)
            return acts
        if self.state == S.MUTED:
            self._go(S.IDLE, o.now)
            self.armed = True

        st = self.state
        if o.poke and st in (S.IDLE, S.LINGER):
            self._new_group(o.now, max(1, o.people))
            acts.append(Say("greet"))
            self._go(S.GREET, o.now)
            return acts
        if st == S.IDLE:
            if o.camera_ok:
                if self.armed and self.present_since is not None and o.now - self.present_since >= cfg.arrive_s:
                    self._new_group(o.now, o.people)
                    acts.append(Say("greet"))
                    self._go(S.GREET, o.now)
            elif o.speech_started and o.brain_ok:  # camera down: voice-triggered turns
                self._new_group(o.now, 0)
                self._listen(o.now, speech_seen=True)

        elif st == S.GREET:
            if not o.audio_busy:
                if o.brain_ok:
                    self._listen(o.now)
                else:  # brain down: canned line only
                    acts.append(Say("fallback"))
                    self.turns = cfg.max_turns
                    self._go(S.SPEAK, o.now)

        elif st == S.LISTEN:
            if o.speech_started:
                self.speech_seen = True
            if o.speech_dropped and o.utterance is None:
                self.speech_seen = False  # it was noise: keep waiting (the no-speech clock still runs)
            if o.utterance is not None and len(o.utterance):
                acts.append(self._submit(o, o.utterance, with_photo=self.turns == 0 and o.camera_ok))
            elif not self.speech_seen and o.now - self.listen_started >= self.no_speech_s:
                if self.turns == 0 and o.camera_ok and o.people > 0:
                    acts.append(self._submit(o, None, with_photo=True))  # comment on the costume
                else:
                    self._go(S.COOLDOWN, o.now)

        elif st == S.THINK:
            done = o.turn_done if (o.turn_done and o.turn_done.turn_id == self.pending) else None
            if done is not None:
                self.pending = None
                self.turns += 1
                if done.ok:
                    acts.append(PlayReply(done))
                    self.costume = self.costume or (done.description if done.description not in
                                                    (None, "", "unclear", "nobody") else None)
                else:
                    acts.append(Say("fallback"))
                self._go(S.SPEAK, o.now)
            elif o.now - self.since >= self.turn_timeout_s:
                acts.append(CancelTurn(self.pending))
                self.pending = None
                self.turns += 1
                acts.append(Say("fallback"))
                self._go(S.SPEAK, o.now)
            elif not self.thinking_played and o.now - self.since >= cfg.thinking_after_s:
                self.thinking_played = True
                acts.append(Say("thinking"))

        elif st == S.SPEAK:
            if not o.audio_busy:
                self.last_activity = o.now
                if o.people > 0 and self.turns < cfg.max_turns and o.brain_ok:
                    self._listen(o.now)  # follow-up window: they often answer right back
                else:
                    self._go(S.COOLDOWN, o.now)

        elif st == S.COOLDOWN:
            if o.now - self.since >= cfg.cooldown_s:
                # IDLE only once the group has really left (armed); a flicker to 0 people is not leaving
                self._go(S.IDLE if self.armed else S.LINGER, o.now)

        elif st == S.LINGER:
            if o.people >= self.group_size + cfg.new_wave_extra and o.camera_ok:
                self.wave_since = self.wave_since or o.now
            else:
                self.wave_since = None
            if self.armed:
                self._go(S.IDLE, o.now)
            elif self.wave_since is not None and o.now - self.wave_since >= cfg.new_wave_s:
                self._new_group(o.now, o.people)  # a bigger wave arrived: greet them
                acts.append(Say("greet"))
                self._go(S.GREET, o.now)
            elif o.speech_started and self.turns < cfg.max_turns and o.brain_ok:
                self._listen(o.now, speech_seen=True)
            elif o.people > 0 and o.now - self.last_activity >= cfg.still_here_s:
                self.turns = 0
                acts.append(Say("still_here"))
                self._go(S.SPEAK, o.now)

        return acts
