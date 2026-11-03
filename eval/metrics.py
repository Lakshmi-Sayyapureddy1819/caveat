"""Metrics for the Phase 2 experiment. No dependencies beyond the stdlib.

Definitions are the frozen ones and are kept strictly separate:

  SILENT ERROR      committed as TRUSTED and the value differs from truth.
  CONFIRMED ERROR   a human explicitly confirmed a wrong value.
  FALSE VERIFICATION challenged a value that was already correct and unhedged.

An UNCONFIRMED wrong value is NOT a silent error — the uncertainty was preserved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, median


# ---------------------------------------------------------------- discrimination
def auc(scores: list[float], positive: list[bool]) -> float | None:
    """Probability a randomly chosen correct token scores above an incorrect one.

    Mann-Whitney U with tie correction. 0.5 = no discrimination at all.
    Returns None when one class is empty (AUC undefined).
    """
    pos = [s for s, p in zip(scores, positive) if p]
    neg = [s for s, p in zip(scores, positive) if not p]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rsum = sum(r for r, p in zip(ranks, positive) if p)
    n1, n2 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2) / (n1 * n2)


def compare_signals(rows: list[dict]) -> dict[str, dict]:
    """Which confidence statistic actually predicts critical-token correctness?

    rows: {"min":float,"mean":float,"median":float,"critical":float,"correct":bool}
    The specification says use the minimum. This function is what decides it.
    """
    correct = [r["correct"] for r in rows]
    out = {}
    for name in ("min", "mean", "median", "critical"):
        vals = [r[name] for r in rows]
        a = auc(vals, correct)
        ok = [v for v, c in zip(vals, correct) if c]
        bad = [v for v, c in zip(vals, correct) if not c]
        out[name] = {
            "auc": None if a is None else round(a, 4),
            "mean_when_correct": round(mean(ok), 4) if ok else None,
            "mean_when_wrong": round(mean(bad), 4) if bad else None,
            "separation": (round(mean(ok) - mean(bad), 4) if ok and bad else None),
            "n_correct": len(ok), "n_wrong": len(bad),
        }
    ranked = [(k, v["auc"]) for k, v in out.items() if v["auc"] is not None]
    if not ranked:
        out["_best"] = None
    else:
        name, best = max(ranked, key=lambda kv: kv[1])
        # An AUC at or below chance means nothing discriminates. Saying so is
        # the finding; crowning the least-bad signal would hide it.
        out["_best"] = name if best > 0.55 else None
        out["_best_auc"] = round(best, 4)
        out["_discriminating"] = best > 0.55
    return out


# ---------------------------------------------------------------- calibration
def reliability(scores: list[float], correct: list[bool], bins: int = 10) -> dict:
    buckets = [{"lo": i / bins, "hi": (i + 1) / bins, "n": 0, "hits": 0,
                "conf_sum": 0.0} for i in range(bins)]
    for s, c in zip(scores, correct):
        idx = min(bins - 1, max(0, int(s * bins)))
        b = buckets[idx]
        b["n"] += 1
        b["hits"] += int(c)
        b["conf_sum"] += s
    total = len(scores)
    ece = 0.0
    for b in buckets:
        if b["n"]:
            b["accuracy"] = b["hits"] / b["n"]
            b["avg_confidence"] = b["conf_sum"] / b["n"]
            ece += (b["n"] / total) * abs(b["accuracy"] - b["avg_confidence"])
        else:
            b["accuracy"] = None
            b["avg_confidence"] = None
    return {"bins": buckets, "ece": round(ece, 4), "n": total}


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Interval that stays honest at small n."""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


# ---------------------------------------------------------------- outcomes
@dataclass
class Outcomes:
    n: int = 0
    silent_errors: int = 0
    confirmed_errors: int = 0
    caught_errors: int = 0          # wrong AND (VERIFY or UNCONFIRMED)
    false_verifications: int = 0    # VERIFY on a correct, unhedged value
    verifications: int = 0
    unconfirmed: int = 0
    trusted: int = 0
    detail: list[dict] = field(default_factory=list)

    def add(self, *, state: str, correct: bool, hedged: bool, label: str = "") -> None:
        self.n += 1
        if state == "TRUSTED":
            self.trusted += 1
            if not correct:
                self.silent_errors += 1
        elif state == "VERIFY":
            self.verifications += 1
            if correct and not hedged:
                self.false_verifications += 1
            if not correct:
                self.caught_errors += 1
        elif state == "UNCONFIRMED":
            self.unconfirmed += 1
            if not correct:
                self.caught_errors += 1
        elif state == "CONFIRMED" and not correct:
            self.confirmed_errors += 1
        self.detail.append({"label": label, "state": state,
                            "correct": correct, "hedged": hedged})

    def report(self) -> dict:
        clear = sum(1 for d in self.detail if d["correct"] and not d["hedged"])
        wrong = sum(1 for d in self.detail if not d["correct"])
        return {
            "n": self.n,
            "wrong_values_present": wrong,
            "silent_errors": self.silent_errors,
            "caught_errors": self.caught_errors,
            "confirmed_errors": self.confirmed_errors,
            "verifications": self.verifications,
            "false_verifications": self.false_verifications,
            "false_verification_rate": (
                round(self.false_verifications / clear, 4) if clear else None),
            "false_verification_ci": wilson(self.false_verifications, clear),
            "unconfirmed": self.unconfirmed,
            "trusted": self.trusted,
        }


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f = (2 * p * r / (p + r)) if (p and r) else None
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": None if p is None else round(p, 4),
            "recall": None if r is None else round(r, 4),
            "f1": None if f is None else round(f, 4),
            "precision_ci": wilson(tp, tp + fp), "recall_ci": wilson(tp, tp + fn)}
