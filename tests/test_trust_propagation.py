"""A value must never leave the system without its trust state."""
import inspect

import pytest

from core.claim import Claim, TrustDecision, TrustState, UndecidedClaimError
from core.expressed import assess_expressed
from core.machine import assess_machine
from core.policy import decide
from tests import fixtures as F

SAID = "I think the torque was 45 Nm"


def _claim(turn, slot):
    return Claim(
        claim_id="clm_1", session_id="ses_1", turn_order=turn.turn_order,
        created_at="2026-09-14T11:04:22.318Z", slot=slot,
        machine=assess_machine(slot, turn, _VOCAB),
        expressed=assess_expressed(slot, turn),
        audio_span=(turn.words[slot.start_idx].start, turn.words[slot.end_idx - 1].end),
    )


@pytest.fixture(autouse=True)
def _bind(vocab):
    global _VOCAB
    _VOCAB = vocab


def test_undecided_claim_cannot_escape(slot_for):
    t = F.T_HEDGED_TORQUE
    c = _claim(t, slot_for(t, "torque", "45"))
    assert c.trust is None
    with pytest.raises(UndecidedClaimError):
        c.to_wire(SAID)


def test_wire_form_always_carries_trust(slot_for):
    t = F.T_HEDGED_TORQUE
    c = _claim(t, slot_for(t, "torque", "45"))
    c = c.decided(decide(c.machine, c.expressed, c.slot.criticality)).with_evidence("evt_1")
    wire = c.to_wire(SAID)
    for field in ("value", "unit", "trust", "reason", "said", "evidence_ref"):
        assert field in wire
    assert wire["value"] == 45
    assert wire["trust"] == "UNCONFIRMED"
    assert wire["reason"] == "speaker_hedged"
    assert wire["said"] == SAID


def test_no_method_returns_a_bare_value():
    """There must be no serialisation path that omits trust."""
    for name, fn in inspect.getmembers(Claim, inspect.isfunction):
        if name.startswith("_"):
            continue
        src = inspect.getsource(fn)
        if "return {" in src or "asdict" in src or "json" in src:
            assert "trust" in src, f"Claim.{name} may emit a value without trust"


def test_trust_can_only_be_set_by_a_trust_decision(slot_for):
    t = F.T_PLAIN_TORQUE
    c = _claim(t, slot_for(t, "torque", "45"))
    for bogus in ("TRUSTED", {"state": "TRUSTED"}, TrustState.TRUSTED, 1, None):
        with pytest.raises(TypeError):
            c.decided(bogus)


def test_llm_output_schema_cannot_express_a_trust_state():
    """Structural injection defence: no model output field names a state."""
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    names = {s.value for s in TrustState}
    for path in (root / "core").glob("*.py"):
        if path.name in ("claim.py", "policy.py", "states.py", "ledger.py"):
            continue
        src = path.read_text()
        assert not (names & set(src.split())), f"{path.name} references a trust state"


def test_correction_clears_trust_and_bumps_version(slot_for):
    t = F.T_HEDGED_TORQUE
    c = _claim(t, slot_for(t, "torque", "45"))
    c = c.decided(decide(c.machine, c.expressed, c.slot.criticality))
    assert c.version == 1 and c.trust is not None

    c2 = c.corrected(55, "fifty-five")
    assert c2.version == 2
    assert c2.trust is None                      # must re-enter the policy engine
    with pytest.raises(UndecidedClaimError):
        c2.to_wire(SAID)
