"""Trust policy — a PURE function.

No LLM. No database. No network. No clock. No randomness.
Same inputs always produce the same output.

The strings TRUSTED / VERIFY / UNCONFIRMED appear in no model output schema
anywhere in this system. There is no channel from a language model to this
decision.
"""
from __future__ import annotations

from core.claim import ExpressedVerdict, MachineVerdict, TrustDecision, TrustState


def decide(
    machine: MachineVerdict,
    expressed: ExpressedVerdict,
    criticality: str,
    policy_rev: str = "v1",
) -> TrustDecision:
    # Low-criticality values are never gated, however they were heard.
    if criticality == "low":
        return TrustDecision(TrustState.TRUSTED, "low_criticality", policy_rev)

    # Machine doubt outranks everything: we may not have the right value at all,
    # and marking a value UNCONFIRMED is meaningless if we misheard it.
    if machine.uncertain:
        return TrustDecision(TrustState.VERIFY, "machine_uncertain", policy_rev)

    # The value is probably right, but the speaker qualified it. Record that
    # honestly rather than challenging someone who already told us they were
    # unsure — they cannot give a better answer.
    if expressed.hedged:
        return TrustDecision(TrustState.UNCONFIRMED, "speaker_hedged", policy_rev)

    return TrustDecision(TrustState.TRUSTED, "clear", policy_rev)
