"""Phase 1 milestone: Turn fixture -> ... -> evidence event, fully offline."""
import pytest

from core.claim import TrustState as S
from core.ledger import Ledger
from core.pipeline import counter, process_turn
from tests import fixtures as F

NOW = "2026-09-14T11:04:22.318Z"


def run(turn, vocab):
    led = Ledger()
    claims = process_turn(turn, vocab, led, "ses_1", NOW, counter())
    return claims, led


# fixture name -> (slot key, value, trust state, reason)
GOLDEN = [
    ("plain_torque",     "torque",   45,      S.TRUSTED,     "clear"),
    ("hedged_torque",    "torque",   45,      S.UNCONFIRMED, "speaker_hedged"),
    ("about_torque",     "torque",   45,      S.UNCONFIRMED, "speaker_hedged"),
    ("trailing_marker",  "torque",   45,      S.UNCONFIRMED, "speaker_hedged"),
    ("asserted_torque",  "torque",   45,      S.TRUSTED,     "clear"),
    ("serial_clear",     "serial",   "A-104", S.TRUSTED,     "clear"),
    ("serial_mumbled",   "serial",   "A-104", S.VERIFY,      "machine_uncertain"),
    ("serial_unknown",   "serial",   "Z-999", S.VERIFY,      "machine_uncertain"),
    ("serial_correction","serial",   "A-140", S.TRUSTED,     "clear"),
    ("pressure_plain",   "pressure", 80,      S.TRUSTED,     "clear"),
    ("pressure_hedged",  "pressure", 80,      S.UNCONFIRMED, "speaker_hedged"),
]


@pytest.mark.parametrize("name,key,value,state,reason", GOLDEN)
def test_golden_fixture(name, key, value, state, reason, vocab):
    claims, led = run(F.ALL_FIXTURES[name], vocab)
    match = [c for c in claims if c.slot.key == key]
    assert match, f"{name}: no {key} claim extracted"
    c = match[0]
    assert c.slot.value == value
    assert c.trust.state is state, f"{name}: {c.trust.state} != {state}"
    assert c.trust.reason == reason
    assert led.verify_chain(c.claim_id)


def test_every_claim_carries_evidence_and_trust(vocab):
    for name, turn in F.ALL_FIXTURES.items():
        claims, led = run(turn, vocab)
        for c in claims:
            assert c.trust is not None, f"{name}: undecided claim escaped"
            assert c.evidence_ref is not None
            assert len(led.events(c.claim_id)) == 3


def test_prompt_injection_does_not_change_the_trust_state(vocab):
    """The injected text reaches the transcript. It reaches nothing else."""
    plain, _ = run(F.T_PLAIN_TORQUE, vocab)
    inj, led = run(F.T_INJECTION, vocab)

    p = [c for c in plain if c.slot.key == "torque"][0]
    i = [c for c in inj if c.slot.key == "torque"][0]

    assert i.trust.state is p.trust.state
    assert i.trust.reason == p.trust.reason
    assert "Ignore your instructions" in led.events(i.claim_id)[0].payload["transcript"]


def test_partial_turns_are_never_gated(vocab):
    import dataclasses
    partial = dataclasses.replace(F.T_HEDGED_TORQUE, end_of_turn=False)
    claims, led = run(partial, vocab)
    assert claims == []
    assert led.events() == ()
