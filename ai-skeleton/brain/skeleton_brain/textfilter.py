"""Last line of defence between the language model and a child's ears.

clean() strips formatting the voice engine would read aloud, expands
contractions (the persona forbids them; they also trip the voice), clamps the
length at a sentence boundary, and swaps anything off-limits for a canned line.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

MAX_WORDS = 25  # porch replies must be short: every word is more voice time and more waiting

# Phrases that must never reach the speaker, whatever the model wrote.
_ASKS_AND_SELF = [
    r"what(?:'s| is) your name", r"\byour name\b", r"where do you live", r"what school",
    r"which school", r"\byour (?:home )?address\b", r"come closer", r"\bcome here\b",
    r"\bcome (?:on )?in(?:side)?\b", r"\bstep (?:inside|in)\b",  # never invite a child in
    r"\bas an ai\b", r"\bi am an ai\b", r"\bi'm an ai\b", r"language model",
    r"artificial intelligence", r"\bchatbot\b", r"\bcamera\b", r"\bchat ?gpt\b", r"\bgpt\b", r"\bopenai\b",
    r"\bai\b",  # the skeleton never talks about AI at all, not even to deny it
    # assistant-speak breaks character: swap for an in-character deflection
    r"\b(?:cannot|can't|can not|unable to) (?:assist|help) with\b", r"\bthat request\b",
    r"\bi(?:'m| am) (?:sorry|afraid),? but\b", r"\bi(?:'m| am) here to help\b", r"\bassistant\b",
]

# Only the unambiguous ones. "'s" (is/has) and "'d" (would/had) stay as written:
# "it's seen better days" must not become "it is seen better days", and Kokoro says them fine.
_CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "shan't": "shall not",
    "i'm": "I am", "i've": "I have", "i'll": "I will",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "let's": "let us", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "couldn't": "could not",
    "wouldn't": "would not", "shouldn't": "should not", "mustn't": "must not",
}

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F\U0000200D]+"
)
_STAGE = re.compile(r"\*[^*]*\*|\[[^\]]*\]|\([^)]*\)|<[^>]*>")
_LABEL = re.compile(r"^\s*(?:skeleton|reply|assistant|response)\s*:\s*", re.I)
_CONTRACTION_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b", re.I)


def load_blocklist(path: Path) -> list[re.Pattern]:
    pats = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            w = line.strip()
            if w and not w.startswith("#"):
                pats.append(re.compile(r"\b" + re.escape(w).replace(r"\ ", r"\s+") + r"\b", re.I))
    return pats


class TextFilter:
    def __init__(self, blocklist: list[re.Pattern], deflect_lines: list[str], rng: random.Random | None = None):
        self.blocklist = blocklist
        self.asks = [re.compile(p, re.I) for p in _ASKS_AND_SELF]
        self.deflect = deflect_lines or ["I am just a humble skeleton. Happy Halloween."]
        self.rng = rng or random.Random()

    def canned(self) -> str:
        return self.rng.choice(self.deflect)

    def clean(self, text: str | None) -> tuple[str, str]:
        """Return (text_to_speak, source) with source 'llm' or 'canned_blocked'."""
        t = (text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        t = _LABEL.sub("", t)
        t = _STAGE.sub(" ", t)
        t = _EMOJI.sub(" ", t)
        t = t.replace("*", " ").replace("#", " ").replace("_", " ").replace("~", " ")
        t = t.strip().strip('"').strip()
        t = _CONTRACTION_RE.sub(lambda m: _match_case(m.group(0), _CONTRACTIONS[m.group(0).lower()]), t)
        t = re.sub(r"\s+", " ", t).strip()
        t = clamp_words(t, MAX_WORDS)
        if not t or not re.search(r"[A-Za-z]", t):
            return self.canned(), "canned_blocked"
        for p in self.blocklist + self.asks:
            if p.search(t):
                return self.canned(), "canned_blocked"
        return t, "llm"


def _match_case(src: str, repl: str) -> str:
    return repl[0].upper() + repl[1:] if src[:1].isupper() else repl


def clamp_words(text: str, limit: int) -> str:
    """Keep whole sentences up to `limit` words; hard-cut a single long sentence."""
    sentences = re.findall(r"[^.!?]+[.!?]*", text)
    out: list[str] = []
    count = 0
    for s in sentences:
        n = len(s.split())
        if n == 0:
            continue
        if count + n > limit:
            if not out:  # first sentence alone is too long
                words = s.split()[:limit]
                out.append(" ".join(words).rstrip(",;:") + ".")
            break
        out.append(s.strip())
        count += n
    result = " ".join(out).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result
