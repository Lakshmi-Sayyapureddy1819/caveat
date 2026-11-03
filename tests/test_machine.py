from dataclasses import replace

import pytest
from core.machine import assess_machine
from tests import fixtures as F


def test_high_confidence_is_certain(vocab, slot_for):
    t = F.T_PLAIN_TORQUE
    v = assess_machine(slot_for(t, "torque", "45"), t, vocab)
    assert v.verdict == "certain"
    assert v.min_word_confidence == 0.94
    assert v.gate == pytest.approx(0.91)          # 0.85 + 0.06 confusable boost


def test_low_confidence_is_uncertain(vocab, slot_for):
    t = F.T_SERIAL_MUMBLED
    v = assess_machine(slot_for(t, "serial", "A-104"), t, vocab)
    assert v.verdict == "uncertain"
    assert v.min_word_confidence == 0.62


def test_confusable_value_raises_the_gate(vocab, slot_for):
    t = F.T_PLAIN_TORQUE
    v = assess_machine(slot_for(t, "torque", "45"), t, vocab)
    assert v.confusable_with == ("55",)
    assert v.gate > vocab.spec("torque").gate


def test_unknown_identifier_is_uncertain_despite_high_confidence(vocab, slot_for):
    t = F.T_SERIAL_UNKNOWN
    v = assess_machine(slot_for(t, "serial", "Z-999"), t, vocab)
    assert v.min_word_confidence == 0.97
    assert v.in_keyterm_set is False
    assert v.verdict == "uncertain"


def test_gate_cap_is_respected(vocab, slot_for):
    t = F.T_SERIAL_CLEAR
    v = assess_machine(slot_for(t, "serial", "A-104"), t, vocab)
    assert v.gate <= vocab.confusable_gate_cap


def test_minimum_not_mean(vocab, slot_for):
    """A slot is as weak as its weakest word."""
    t = F.make_turn([("The", .99), ("torque", .99), ("was", .99),
                     ("45", .40), ("Nm", .99)])
    s = slot_for(t, "torque", "45")
    s = replace(s, end_idx=s.start_idx + 2)     # span "45 Nm"
    v = assess_machine(s, t, vocab)
    assert v.min_word_confidence == 0.40      # not the 0.695 mean
    assert v.verdict == "uncertain"
