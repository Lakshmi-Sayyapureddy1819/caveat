import pytest

from core.ledger import GENESIS, EventType as E, Ledger, compute_hash

TS = "2026-09-14T11:04:22.318Z"


def _seed(led, cid="clm_1"):
    led.append(cid, E.ClaimExtracted,
               {"slot": "torque", "value": 45, "as_heard": "forty-five",
                "transcript": "I think the torque was 45 Nm"}, TS)
    led.append(cid, E.UncertaintyAssessed,
               {"machine": {"min_word_confidence": 0.94, "verdict": "certain"},
                "expressed": {"marker": "i think", "verdict": "hedged"}}, TS)
    led.append(cid, E.TrustDecided,
               {"state": "UNCONFIRMED", "reason": "speaker_hedged",
                "policy_rev": "v1"}, TS)
    return cid


def test_append_and_read_back():
    led = Ledger()
    cid = _seed(led)
    evs = led.events(cid)
    assert len(evs) == 3
    assert [e.type for e in evs] == [E.ClaimExtracted, E.UncertaintyAssessed, E.TrustDecided]


def test_chain_starts_at_genesis_and_links():
    led = Ledger()
    cid = _seed(led)
    evs = led.events(cid)
    assert evs[0].prev_hash == GENESIS
    for prev, cur in zip(evs, evs[1:]):
        assert cur.prev_hash == prev.hash
    assert led.verify_chain(cid)
    assert led.head(cid) == evs[-1].hash


def test_tampering_breaks_the_chain():
    led = Ledger()
    cid = _seed(led)
    victim = led.events(cid)[0]
    forged = compute_hash(victim.claim_id, victim.type, victim.timestamp,
                          victim.actor, {"value": 999}, victim.prev_hash)
    assert forged != victim.hash          # a changed payload changes the hash
    led._events[0] = type(victim)(
        **{**{f: getattr(victim, f) for f in victim.__slots__},
           "payload": {"value": 999}})
    assert led.verify_chain(cid) is False


def test_model_reasoning_is_rejected():
    led = Ledger()
    for key in ("reasoning", "chain_of_thought", "scratchpad"):
        with pytest.raises(ValueError, match="reasoning"):
            led.append("clm_x", E.ClaimExtracted, {key: "because..."}, TS)


def test_actor_must_be_system_or_human():
    led = Ledger()
    with pytest.raises(ValueError):
        led.append("clm_x", E.ClaimExtracted, {}, TS, actor="llm")


def test_fold_derives_current_state_including_correction():
    led = Ledger()
    cid = _seed(led)
    led.append(cid, E.VerificationAsked,
               {"question_id": "q1", "question_text": "45 or 55?"}, TS)
    led.append(cid, E.ClaimAnswered,
               {"question_id": "q1", "outcome": "corrected", "new_value": 55},
               TS, actor="human")
    led.append(cid, E.ClaimCommitted,
               {"final_value": 55, "final_state": "CONFIRMED"}, TS)
    state = led.fold(cid)
    assert state["value"] == 55
    assert state["trust"] == "CONFIRMED"
    assert state["committed"] is True
    assert state["events"] == 6
    assert led.verify_chain(cid)


def test_chains_are_per_claim_and_independent():
    led = Ledger()
    _seed(led, "clm_a")
    _seed(led, "clm_b")
    assert led.events("clm_a")[0].prev_hash == GENESIS
    assert led.events("clm_b")[0].prev_hash == GENESIS
    assert led.head("clm_a") != led.head("clm_b")
    assert led.verify_chain("clm_a") and led.verify_chain("clm_b")
