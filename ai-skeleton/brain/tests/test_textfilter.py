import random
from pathlib import Path

from skeleton_brain.textfilter import TextFilter, clamp_words, load_blocklist

PERSONA = Path(__file__).resolve().parents[2] / "persona"
DEFLECT = ["I am a skeleton. I have always been a skeleton."]


def make():
    return TextFilter(load_blocklist(PERSONA / "blocklist.txt"), DEFLECT, random.Random(0))


def test_strips_stage_directions_emoji_and_labels():
    text, src = make().clean('Skeleton: *rattles bones* Welcome, friend! \U0001F383 (creaks) Take some candy.')
    assert src == "llm"
    assert text == "Welcome, friend! Take some candy."


def test_expands_contractions_keeping_case():
    text, _ = make().clean("It's a fine night. I'm glad you're here, don't be shy.")
    assert text == "It is a fine night. I am glad you are here, do not be shy."


def test_clamps_to_whole_sentences():
    long = "One two three four five six seven eight nine ten. " * 5
    out = clamp_words(long, 40)
    assert len(out.split()) == 40
    assert out.endswith(".")


def test_clamps_single_long_sentence():
    out = clamp_words(" ".join(["bones"] * 60), 40)
    assert len(out.split()) == 40 and out.endswith(".")


def test_blocklist_swaps_in_canned_line():
    for bad in ["Have a beer, friend.", "I will kill time with you.", "Watch this knife trick."]:
        text, src = make().clean(bad)
        assert src == "canned_blocked" and text in DEFLECT, bad


def test_never_asks_for_names_or_closer():
    for bad in ["What is your name, little one?", "Come closer, friend.", "Where do you live?",
                "As an AI, I cannot do that.", "My camera sees a pirate."]:
        _, src = make().clean(bad)
        assert src == "canned_blocked", bad


def test_assistant_speak_is_out_of_character():
    # seen from qwen3-vl:2b on "do you want a beer"
    raw = "I'm sorry, but I can't assist with that request. I'm a skeleton, and I'm here to help."
    assert make().clean(raw)[1] == "canned_blocked"


def test_allowed_halloween_words_pass():
    for ok in ["A vampire! I am glad I have no blood to spare.", "Arr, a fearsome pirate. Take some candy.",
               "Boo to you too, little ghost."]:
        text, src = make().clean(ok)
        assert src == "llm", ok


def test_empty_or_garbage_is_canned():
    assert make().clean("")[1] == "canned_blocked"
    assert make().clean("*** ...")[1] == "canned_blocked"
