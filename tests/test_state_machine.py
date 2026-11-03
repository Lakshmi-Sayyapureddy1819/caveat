import pytest

from core.claim import TrustState as S
from core.states import IllegalTransition, can_transition, is_terminal, transition


@pytest.mark.parametrize("to", [S.TRUSTED, S.VERIFY, S.UNCONFIRMED])
def test_new_claim_may_reach_any_initial_state(to):
    assert transition(None, to) is to


def test_verify_may_confirm_or_correct():
    assert transition(S.VERIFY, S.CONFIRMED) is S.CONFIRMED
    assert transition(S.VERIFY, S.CORRECTED) is S.CORRECTED


def test_unconfirmed_to_trusted_is_forbidden():
    """The hard rule: a claim is TRUSTED at birth or never."""
    assert not can_transition(S.UNCONFIRMED, S.TRUSTED)
    with pytest.raises(IllegalTransition):
        transition(S.UNCONFIRMED, S.TRUSTED)


@pytest.mark.parametrize("frm", [S.VERIFY, S.CONFIRMED, S.CORRECTED, S.UNCONFIRMED])
def test_nothing_may_become_trusted_after_birth(frm):
    with pytest.raises(IllegalTransition):
        transition(frm, S.TRUSTED)


def test_unconfirmed_may_be_explicitly_confirmed():
    assert transition(S.UNCONFIRMED, S.CONFIRMED) is S.CONFIRMED


def test_trusted_is_terminal():
    assert is_terminal(S.TRUSTED)
    for to in S:
        assert not can_transition(S.TRUSTED, to)


def test_pipeline_stages_are_not_trust_states():
    names = {s.value for s in S}
    assert names == {"TRUSTED", "VERIFY", "UNCONFIRMED", "CONFIRMED", "CORRECTED"}
    for stage in ("DETECTED", "EXTRACTED", "ASSESSED"):
        assert stage not in names
