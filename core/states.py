"""Trust state machine.

Five externally meaningful states. DETECTED / EXTRACTED / ASSESSED are
pipeline stages, not trust states, and are deliberately absent.

Hard rule: there is NO UNCONFIRMED -> TRUSTED edge. A claim is TRUSTED at
birth or it never is. A correction re-enters the policy engine.
"""
from __future__ import annotations

from core.claim import TrustState


class IllegalTransition(RuntimeError):
    def __init__(self, frm, to):
        super().__init__(f"illegal trust transition: {frm} -> {to}")
        self.frm, self.to = frm, to


# None = a newly extracted claim with no decision yet.
_ALLOWED: dict[TrustState | None, frozenset[TrustState]] = {
    None: frozenset({TrustState.TRUSTED, TrustState.VERIFY, TrustState.UNCONFIRMED}),
    TrustState.VERIFY: frozenset({TrustState.CONFIRMED, TrustState.CORRECTED}),
    TrustState.UNCONFIRMED: frozenset({TrustState.CONFIRMED}),
    TrustState.TRUSTED: frozenset(),
    TrustState.CONFIRMED: frozenset(),
    TrustState.CORRECTED: frozenset(),
}


def can_transition(frm: TrustState | None, to: TrustState) -> bool:
    return to in _ALLOWED.get(frm, frozenset())


def transition(frm: TrustState | None, to: TrustState) -> TrustState:
    if not can_transition(frm, to):
        raise IllegalTransition(frm, to)
    return to


def is_terminal(state: TrustState) -> bool:
    return not _ALLOWED.get(state, frozenset())
