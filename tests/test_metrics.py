"""The metrics must be right before any real data touches them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from eval.metrics import Outcomes, auc, compare_signals, prf, reliability, wilson


def test_auc_perfect_separation():
    assert auc([0.9, 0.8, 0.3, 0.2], [True, True, False, False]) == 1.0


def test_auc_perfect_inversion():
    assert auc([0.2, 0.3, 0.8, 0.9], [True, True, False, False]) == 0.0


def test_auc_partial_overlap():
    # pos {0.9, 0.3}, neg {0.8, 0.2}; 3 of 4 pairs ordered correctly
    assert auc([0.9, 0.8, 0.3, 0.2], [True, False, True, False]) == 0.75


def test_auc_all_ties_is_chance():
    assert auc([0.5] * 6, [True, False, True, False, True, False]) == 0.5


def test_auc_undefined_with_one_class():
    assert auc([0.9, 0.8], [True, True]) is None
    assert auc([0.9, 0.8], [False, False]) is None


def test_compare_signals_picks_the_discriminating_one():
    """'min' separates; 'mean' is noise. The function must say so."""
    rows = [
        {"min": 0.95, "mean": 0.5, "median": 0.5, "critical": 0.95, "correct": True},
        {"min": 0.93, "mean": 0.9, "median": 0.5, "critical": 0.93, "correct": True},
        {"min": 0.40, "mean": 0.7, "median": 0.5, "critical": 0.40, "correct": False},
        {"min": 0.35, "mean": 0.6, "median": 0.5, "critical": 0.35, "correct": False},
    ]
    out = compare_signals(rows)
    assert out["min"]["auc"] == 1.0
    assert out["median"]["auc"] == 0.5          # constant -> chance
    assert out["_best"] in ("min", "critical")
    assert out["min"]["separation"] == pytest.approx(0.565, abs=1e-3)


def test_reliability_ece_zero_when_perfectly_calibrated():
    scores = [0.95] * 20 + [0.55] * 20
    correct = [True] * 19 + [False] + [True] * 11 + [False] * 9
    r = reliability(scores, correct)
    assert r["n"] == 40
    assert r["ece"] < 0.02


def test_reliability_flags_overconfidence():
    """Claims 0.95, is right half the time."""
    r = reliability([0.95] * 20, [True] * 10 + [False] * 10)
    assert r["ece"] == pytest.approx(0.45, abs=0.01)


def test_wilson_is_wide_at_small_n():
    lo, hi = wilson(14, 17)
    assert lo < 0.824 < hi
    assert hi - lo > 0.30           # honest about n=17


def test_unconfirmed_wrong_is_not_a_silent_error():
    o = Outcomes()
    o.add(state="UNCONFIRMED", correct=False, hedged=True)
    r = o.report()
    assert r["silent_errors"] == 0
    assert r["caught_errors"] == 1


def test_trusted_wrong_is_a_silent_error():
    o = Outcomes()
    o.add(state="TRUSTED", correct=False, hedged=False)
    assert o.report()["silent_errors"] == 1


def test_confirmed_error_is_tracked_separately():
    o = Outcomes()
    o.add(state="CONFIRMED", correct=False, hedged=False)
    r = o.report()
    assert r["confirmed_errors"] == 1
    assert r["silent_errors"] == 0


def test_challenging_a_hedged_value_is_not_a_false_verification():
    o = Outcomes()
    o.add(state="VERIFY", correct=True, hedged=True)
    assert o.report()["false_verifications"] == 0


def test_prf():
    r = prf(tp=11, fp=1, fn=1)
    assert r["precision"] == pytest.approx(0.9167, abs=1e-3)
    assert r["recall"] == pytest.approx(0.9167, abs=1e-3)


def test_best_is_none_when_nothing_beats_chance():
    """Two high-confidence wrong values: no signal discriminates. Say so."""
    rows = [
        {"min": 0.97, "mean": 0.97, "median": 0.97, "critical": 0.97, "correct": False},
        {"min": 0.95, "mean": 0.95, "median": 0.95, "critical": 0.95, "correct": True},
        {"min": 0.96, "mean": 0.96, "median": 0.96, "critical": 0.96, "correct": False},
        {"min": 0.94, "mean": 0.94, "median": 0.94, "critical": 0.94, "correct": True},
    ]
    out = compare_signals(rows)
    assert out["_best"] is None
    assert out["_discriminating"] is False


def test_best_is_named_when_a_signal_clearly_works():
    rows = [
        {"min": 0.97, "mean": 0.5, "median": 0.5, "critical": 0.97, "correct": True},
        {"min": 0.95, "mean": 0.5, "median": 0.5, "critical": 0.95, "correct": True},
        {"min": 0.42, "mean": 0.5, "median": 0.5, "critical": 0.42, "correct": False},
        {"min": 0.38, "mean": 0.5, "median": 0.5, "critical": 0.38, "correct": False},
    ]
    out = compare_signals(rows)
    assert out["_best"] in ("min", "critical")
    assert out["_discriminating"] is True
