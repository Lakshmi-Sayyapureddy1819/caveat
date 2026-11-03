from core.expressed import assess_expressed
from tests import fixtures as F


def test_i_think_before_slot(slot_for):
    t = F.T_HEDGED_TORQUE
    v = assess_expressed(slot_for(t, "torque", "45"), t)
    assert v.verdict == "hedged"
    assert v.marker == "i think"
    assert v.marker_span is not None


def test_trailing_i_think_after_slot(slot_for):
    """'45 Nm, I think' — the case a naive lookback-only implementation misses."""
    t = F.T_TRAILING_MARKER
    v = assess_expressed(slot_for(t, "torque", "45"), t)
    assert v.verdict == "hedged"
    assert v.marker == "i think"


def test_i_believe(slot_for):
    t = F.T_PRESSURE_HEDGED
    v = assess_expressed(slot_for(t, "pressure", "80"), t)
    assert v.verdict == "hedged"
    assert v.marker == "i believe"


def test_about_is_a_hedge(slot_for):
    t = F.T_ABOUT_TORQUE
    v = assess_expressed(slot_for(t, "torque", "45"), t)
    assert v.verdict == "hedged"
    assert v.marker == "about"


def test_self_repair(slot_for):
    t = F.T_SELF_REPAIR
    v = assess_expressed(slot_for(t, "serial", "A-104"), t)
    assert v.self_repair is True
    assert v.verdict == "hedged"


def test_plain_statement_is_not_hedged(slot_for):
    t = F.T_PLAIN_TORQUE
    v = assess_expressed(slot_for(t, "torque", "45"), t)
    assert v.verdict == "plain"
    assert v.marker is None
    assert v.self_repair is False


def test_assertion_is_not_hedged(slot_for):
    """'definitely 45' must NOT be flagged. A pause alone is never a trigger."""
    t = F.T_ASSERTED_TORQUE
    v = assess_expressed(slot_for(t, "torque", "45"), t)
    assert v.verdict == "plain"


def test_no_timing_signal_is_used():
    """Guard the scientific claim: no prosody, rate or pause inference."""
    import inspect
    from core import expressed
    import ast
    tree = ast.parse(inspect.getsource(expressed))
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)):
        tree.body.pop(0)                       # drop the module docstring
    body = ast.unparse(tree)                   # comments and docstrings gone
    for banned in ("speech_rate", "syllables", "pause_ms", "rate_drop", "prosody"):
        assert banned not in body


def test_self_repair_extracts_the_corrected_value_not_the_discarded_one(vocab):
    """'A-140 ... sorry, A-104' must yield A-104. The final value wins."""
    from core.claim import find_slots
    slots = find_slots(F.T_SELF_REPAIR, vocab)
    serial = [s for s in slots if s[0] == "serial"]
    assert serial, "no serial extracted"
    assert serial[0][3] == "A-104"
