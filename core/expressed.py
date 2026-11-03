"""Expressed-uncertainty engine — lexical only.

We detect what the speaker SAID, never what they internally believed.

Deliberately NOT used: pause duration, speech rate, prosody, emotion.
Filled-pause behaviour has documented dual status and varies by genre and by
native-speaker status (Kosmala & Crible, Language and Speech 65(1), 2022), so
a timing-based detector would systematically flag L2 speakers and people who
stutter. The lexicon carries the signal on its own.
"""
from __future__ import annotations

from core.claim import ExpressedVerdict, Slot, Turn

# Ordered longest-first so "not sure" wins over "sure".
MARKERS: tuple[str, ...] = (
    "if i remember",
    "something like",
    "i'm not sure",
    "im not sure",
    "not sure",
    "pretty sure",
    "i believe",
    "i think",
    "i guess",
    "i reckon",
    "probably",
    "roughly",
    "or so",
    "maybe",
    "around",
    "about",
    "i mean",
)

# Words that cancel a hedge reading when they sit next to the slot.
ASSERTIONS: frozenset[str] = frozenset(
    {"definitely", "certainly", "exactly", "confirmed", "positive", "sure"}
)

# A hedge scopes over the clause it introduces: "I think the torque was 45"
# puts five words between the marker and the slot. Look back to the start of
# the clause, bounded, rather than a fixed short window.
LOOKBACK = 8      # words before the slot, stopping at a clause boundary
LOOKAHEAD = 4     # words after the slot
REPAIR_LOOKBACK = 4

_BOUNDARY = frozenset({".", "!", "?", ";"})


def _norm(tokens) -> str:
    return " ".join(t.text.lower().strip(".,;:!?") for t in tokens)


def _clause_before(words, slot_start: int):
    """Words from the start of the current clause up to the slot."""
    lo = max(0, slot_start - LOOKBACK)
    for j in range(slot_start - 1, lo - 1, -1):
        if any(words[j].text.rstrip().endswith(p) for p in _BOUNDARY):
            lo = j + 1
            break
    return words[lo:slot_start]


def _detect_self_repair(turn: Turn, slot: Slot) -> bool:
    """'the B... the C-line' / '55... no, 45'.

    Looks for a discarded token of the same shape shortly before the slot,
    optionally with a repair cue between them.
    """
    words = turn.words
    start = max(0, slot.start_idx - REPAIR_LOOKBACK)
    prior = words[start : slot.start_idx]
    if not prior:
        return False

    prior_text = " ".join(w.text.lower().strip(".,;:!?") for w in prior)
    if any(cue in prior_text for cue in ("no,", " no ", "sorry", "rather", "i mean")):
        return True

    target = "".join(c for c in slot.as_heard.lower() if c.isalnum())
    if not target:
        return False
    for w in prior:
        tok = "".join(c for c in w.text.lower() if c.isalnum())
        if not tok or tok == target:
            continue
        # same shape (both numeric, or both identifier-like) and same length
        if len(tok) == len(target) and tok.isdigit() == target.isdigit():
            return True
    return False


def assess_expressed(slot: Slot, turn: Turn) -> ExpressedVerdict:
    words = turn.words

    before = _clause_before(words, slot.start_idx)
    after = words[slot.end_idx : slot.end_idx + LOOKAHEAD]

    # An explicit assertion adjacent to the slot cancels a hedge reading.
    adjacent = _norm(before[-2:] if before else ()) + " " + _norm(after[:2] if after else ())
    asserted = any(a in adjacent.split() for a in ASSERTIONS)

    marker = None
    marker_span = None

    for window in (before, after):
        if marker is not None:
            break
        text = _norm(window)
        for m in MARKERS:
            if m in text:
                marker = m
                if window:
                    marker_span = (window[0].start, window[-1].end)
                break

    repair = _detect_self_repair(turn, slot)

    hedged = (marker is not None or repair) and not asserted

    return ExpressedVerdict(
        marker=marker,
        marker_span=marker_span,
        self_repair=repair,
        verdict="hedged" if hedged else "plain",
    )
