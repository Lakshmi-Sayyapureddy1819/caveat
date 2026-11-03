"""Machine-uncertainty engine — deterministic, no model call, no I/O.

Three signals, per the frozen spec:
  1. minimum word confidence across the slot span, against a per-class gate
  2. identifier not present in the vocabulary's known values
  3. a known confusable twin raises the gate

The AssemblyAI streaming API returns exactly ONE hypothesis. `confusable_with`
comes from the vocabulary, never from the recogniser.
"""
from __future__ import annotations

from core.claim import MachineVerdict, Slot, Turn


def assess_machine(slot: Slot, turn: Turn, vocab) -> MachineVerdict:
    spec = vocab.spec(slot.key)
    span = turn.words[slot.start_idx : slot.end_idx]
    if not span:
        raise ValueError(f"slot {slot.key!r} spans no words")

    # Weakest link, never the mean: averaging hides the one garbled token.
    min_conf = min(w.confidence for w in span)

    twins = spec.twins_of(slot.as_heard) or spec.twins_of(str(slot.value))

    gate = spec.gate
    if twins:
        gate = min(vocab.confusable_gate_cap, gate + vocab.confusable_boost)

    in_keyterm_set = True
    if spec.cls == "identifier" and spec.known_values:
        in_keyterm_set = str(slot.value).upper() in {
            v.upper() for v in spec.known_values
        }

    uncertain = (min_conf < gate) or (not in_keyterm_set)

    return MachineVerdict(
        min_word_confidence=round(min_conf, 4),
        gate=round(gate, 4),
        in_keyterm_set=in_keyterm_set,
        confusable_with=tuple(twins),
        verdict="uncertain" if uncertain else "certain",
    )
