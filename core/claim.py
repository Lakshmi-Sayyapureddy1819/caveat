"""Frozen Claim schema.

`trust` is the only field the policy engine may set, and the only way to
attach one is `Claim.decided()`. There is no public setter, no default and
no constructor path that produces a committed value without a TrustState.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------------------
# AssemblyAI wire types. These mirror the streaming API exactly.
# Do not add fields the API does not send.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start: int          # ms
    end: int            # ms
    confidence: float   # 0.0 - 1.0
    word_is_final: bool
    speaker: str | None = None

    @staticmethod
    def from_api(d: Mapping[str, Any]) -> "Word":
        return Word(
            text=d["text"],
            start=int(d["start"]),
            end=int(d["end"]),
            confidence=float(d["confidence"]),
            word_is_final=bool(d["word_is_final"]),
            speaker=d.get("speaker"),
        )


@dataclass(frozen=True, slots=True)
class Turn:
    turn_order: int
    end_of_turn: bool
    turn_is_formatted: bool
    transcript: str
    end_of_turn_confidence: float
    words: tuple[Word, ...]
    # Optional fields the API sends. `utterance` is populated only when
    # end_of_turn is true. On universal-3-5-pro `turn_is_formatted` tracks
    # `end_of_turn` — there is no separate formatted/unformatted pair, so
    # analysis and display read the same text.
    utterance: str | None = None
    speaker_label: str | None = None
    language_code: str | None = None
    language_confidence: float | None = None

    @staticmethod
    def from_api(d: Mapping[str, Any]) -> "Turn":
        lc = d.get("language_confidence")
        return Turn(
            turn_order=int(d["turn_order"]),
            end_of_turn=bool(d["end_of_turn"]),
            turn_is_formatted=bool(d.get("turn_is_formatted", False)),
            transcript=d.get("transcript", ""),
            end_of_turn_confidence=float(d.get("end_of_turn_confidence", 0.0)),
            words=tuple(Word.from_api(w) for w in d.get("words", ())),
            utterance=d.get("utterance"),
            speaker_label=d.get("speaker_label"),
            language_code=d.get("language_code"),
            language_confidence=(float(lc) if lc is not None else None),
        )


# --------------------------------------------------------------------------
# Claim parts
# --------------------------------------------------------------------------


class TrustState(str, Enum):
    TRUSTED = "TRUSTED"
    VERIFY = "VERIFY"
    UNCONFIRMED = "UNCONFIRMED"
    CONFIRMED = "CONFIRMED"
    CORRECTED = "CORRECTED"


@dataclass(frozen=True, slots=True)
class Slot:
    key: str
    cls: str
    value: Any
    unit: str | None
    as_heard: str
    criticality: str
    start_idx: int      # index into Turn.words, inclusive
    end_idx: int        # exclusive


@dataclass(frozen=True, slots=True)
class MachineVerdict:
    min_word_confidence: float
    gate: float
    in_keyterm_set: bool
    confusable_with: tuple[str, ...]
    verdict: str        # "certain" | "uncertain"

    @property
    def uncertain(self) -> bool:
        return self.verdict == "uncertain"


@dataclass(frozen=True, slots=True)
class ExpressedVerdict:
    marker: str | None
    marker_span: tuple[int, int] | None   # ms
    self_repair: bool
    verdict: str        # "plain" | "hedged"

    @property
    def hedged(self) -> bool:
        return self.verdict == "hedged"


@dataclass(frozen=True, slots=True)
class TrustDecision:
    state: TrustState
    reason: str
    policy_rev: str


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    session_id: str
    turn_order: int
    created_at: str
    slot: Slot
    machine: MachineVerdict
    expressed: ExpressedVerdict
    audio_span: tuple[int, int]
    version: int = 1
    trust: TrustDecision | None = None
    evidence_ref: str | None = None

    # -- the only way to attach a trust state -----------------------------
    def decided(self, decision: TrustDecision) -> "Claim":
        if not isinstance(decision, TrustDecision):
            raise TypeError("trust must be a TrustDecision produced by the policy engine")
        return replace(self, trust=decision)

    def with_evidence(self, evidence_ref: str) -> "Claim":
        return replace(self, evidence_ref=evidence_ref)

    def corrected(self, new_value: Any, new_as_heard: str) -> "Claim":
        """A correction produces a NEW version with no trust state.
        It must be re-run through the policy engine."""
        return replace(
            self,
            slot=replace(self.slot, value=new_value, as_heard=new_as_heard),
            version=self.version + 1,
            trust=None,
        )

    # -- trust propagation: the only serialisation -------------------------
    def to_wire(self, said: str) -> dict[str, Any]:
        """Wire form for any downstream reader.

        There is deliberately no method that returns a bare value. Every path
        out of a Claim carries the trust state with it.
        """
        if self.trust is None:
            raise UndecidedClaimError(
                f"claim {self.claim_id} has no trust state; it cannot leave the system"
            )
        return {
            "value": self.slot.value,
            "unit": self.slot.unit,
            "trust": self.trust.state.value,
            "reason": self.trust.reason,
            "said": said,
            "evidence_ref": self.evidence_ref,
        }


class UndecidedClaimError(RuntimeError):
    """Raised when a claim without a trust state is about to escape."""


# --------------------------------------------------------------------------
# Deterministic slot spanning.
#
# In rt/ a bounded LLM call proposes spans. This regex path handles the
# simple numeric and identifier cases, and is what makes core testable with
# zero network. It labels spans; it decides nothing.
# --------------------------------------------------------------------------

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_IDENT_RE = re.compile(r"^[A-Za-z]-?\d{2,4}$")
_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")


def _clean(tok: str) -> str:
    return tok.strip().strip(".,;:!?").strip()


def normalise_number(tokens: Sequence[str]) -> float | int | None:
    """'forty-five' | '45' | 'forty five' -> 45. Deterministic, no model."""
    joined = " ".join(_clean(t).lower() for t in tokens).replace("-", " ")
    if _NUM_RE.match(joined.replace(" ", "")):
        raw = joined.replace(" ", "")
        return float(raw) if "." in raw else int(raw)
    total, current, seen = 0, 0, False
    for part in joined.split():
        if part not in _NUMBER_WORDS:
            return None
        current += _NUMBER_WORDS[part]
        seen = True
    total += current
    return total if seen else None


def find_slots(turn: Turn, vocab) -> list[tuple[str, int, int, Any, str]]:
    """Return (slot_key, start_idx, end_idx, value, as_heard).

    Keyword-anchored: a slot's keyterms must appear in the turn for that slot
    to be considered, which keeps the deterministic path conservative.
    """
    words = turn.words
    lowered = [_clean(w.text).lower() for w in words]
    text = " ".join(lowered)
    out: list[tuple[str, int, int, Any, str]] = []
    claimed: set[int] = set()

    for key, spec in vocab.slots.items():
        if spec.cls not in ("quantity", "identifier"):
            continue
        if spec.keyterms and not any(kt.lower() in text for kt in spec.keyterms):
            continue

        last_ident = None
        for i, w in enumerate(words):
            if i in claimed:
                continue
            tok = _clean(w.text)

            if spec.cls == "identifier":
                if _IDENT_RE.match(tok):
                    # A self-correction ("A-140 ... sorry, A-104") leaves the
                    # discarded value earlier in the turn. The final value wins,
                    # so keep scanning and take the last candidate.
                    cand = (key, i, i + 1, tok.upper(), tok)
                    claimed.add(i)
                    last_ident = cand
                    continue

            elif spec.cls == "quantity":
                # try a two-word number ("forty five") then a single token
                for width in (2, 1):
                    if i + width > len(words):
                        continue
                    if any(j in claimed for j in range(i, i + width)):
                        continue
                    span = [words[j].text for j in range(i, i + width)]
                    val = normalise_number(span)
                    if val is not None:
                        as_heard = " ".join(_clean(s) for s in span)
                        out.append((key, i, i + width, val, as_heard))
                        claimed.update(range(i, i + width))
                        break
                else:
                    continue
                break

        if last_ident is not None:
            out.append(last_ident)

    return out
