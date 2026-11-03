"""Turn -> Claim -> assessed -> decided -> evidence.

Zero network, zero I/O. `now` and `claim_id` are injected so the pipeline
stays deterministic and testable; nothing here reads a clock.
"""
from __future__ import annotations

from typing import Callable, Iterator

from core.claim import Claim, Slot, Turn, find_slots
from core.expressed import assess_expressed
from core.ledger import EventType as E, Ledger
from core.machine import assess_machine
from core.policy import decide
from core.states import transition


def process_turn(
    turn: Turn,
    vocab,
    ledger: Ledger,
    session_id: str,
    now: str,
    next_id: Callable[[], str],
) -> list[Claim]:
    """Run one final Turn through the full core pipeline."""
    if not turn.end_of_turn:
        return []          # never gate on a partial turn

    claims: list[Claim] = []

    for key, s_idx, e_idx, value, as_heard in find_slots(turn, vocab):
        spec = vocab.spec(key)
        slot = Slot(
            key=key, cls=spec.cls, value=value, unit=spec.unit,
            as_heard=as_heard, criticality=spec.criticality,
            start_idx=s_idx, end_idx=e_idx,
        )
        claim_id = next_id()
        span = (turn.words[s_idx].start, turn.words[e_idx - 1].end)

        ledger.append(claim_id, E.ClaimExtracted, {
            "slot": key, "value": value, "as_heard": as_heard,
            "transcript": turn.transcript, "audio_span": list(span),
            "turn_order": turn.turn_order,
        }, now)

        machine = assess_machine(slot, turn, vocab)
        expressed = assess_expressed(slot, turn)

        ledger.append(claim_id, E.UncertaintyAssessed, {
            "machine": {
                "min_word_confidence": machine.min_word_confidence,
                "gate": machine.gate,
                "in_keyterm_set": machine.in_keyterm_set,
                "confusable_with": list(machine.confusable_with),
                "verdict": machine.verdict,
            },
            "expressed": {
                "marker": expressed.marker,
                "self_repair": expressed.self_repair,
                "verdict": expressed.verdict,
            },
        }, now)

        decision = decide(machine, expressed, slot.criticality, vocab.policy_rev)
        transition(None, decision.state)          # rejects any illegal birth state

        ev = ledger.append(claim_id, E.TrustDecided, {
            "state": decision.state.value,
            "reason": decision.reason,
            "policy_rev": decision.policy_rev,
        }, now)

        claim = Claim(
            claim_id=claim_id, session_id=session_id, turn_order=turn.turn_order,
            created_at=now, slot=slot, machine=machine, expressed=expressed,
            audio_span=span,
        ).decided(decision).with_evidence(ev.event_id)

        claims.append(claim)

    return claims


def counter(prefix: str = "clm") -> Callable[[], str]:
    n = 0
    def _next() -> str:
        nonlocal n
        n += 1
        return f"{prefix}_{n:04d}"
    return _next
