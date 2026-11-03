#!/usr/bin/env python3
"""Offline replay: captured Turn JSON -> Caveat arms -> metrics.

Once tools/record_turns.py has captured the fixtures, every threshold and
ablation experiment runs from disk. No microphone, no AssemblyAI call, no key.
That is what makes the experiment repeatable and affordable.

    python3 eval/replay.py --fixtures eval/fixtures/A4 --labels eval/labels.csv
    python3 eval/replay.py --fixtures eval/fixtures/A4 --labels ... --json out.json
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.claim import Slot, Turn, find_slots
from core.expressed import assess_expressed
from core.ledger import Ledger
from core.machine import assess_machine
from core.pipeline import counter
from core.policy import decide
from eval.metrics import Outcomes, compare_signals, prf, reliability
from vocab.schema import load_vocabulary

# --------------------------------------------------------------------------
# Ablation arms. Each is a decision policy over the SAME assessed claim, so
# the only thing that varies is which signals the gate is allowed to see.
# --------------------------------------------------------------------------
ARMS = ("BASELINE", "A_machine", "B_expressed", "C_both", "D_criticality", "E_full")

# Competing architectures, as decision policies. This is not a benchmark of
# anyone's implementation — it runs their published decision rule on our data.
POLICIES = ("P0_always_commit", "P1_confirm_all_critical", "P2_action_gate", "P3_caveat")


def arm_decide(arm: str, machine, expressed, criticality: str) -> str:
    if arm == "BASELINE":
        return "TRUSTED"
    if arm == "A_machine":
        return "VERIFY" if machine.uncertain else "TRUSTED"
    if arm == "B_expressed":
        return "UNCONFIRMED" if expressed.hedged else "TRUSTED"
    if arm == "C_both":
        if machine.uncertain:
            return "VERIFY"
        return "UNCONFIRMED" if expressed.hedged else "TRUSTED"
    # D and E both apply criticality; E adds the keyterm loop, which changes
    # recognition upstream, not the gate. They differ only across arms A1..A4.
    return decide(machine, expressed, criticality).state.value


def policy_decide(pol: str, machine, expressed, criticality: str,
                  triggers_action: bool) -> str:
    if pol == "P0_always_commit":
        return "TRUSTED"
    if pol == "P1_confirm_all_critical":
        return "VERIFY" if criticality != "low" else "TRUSTED"
    if pol == "P2_action_gate":
        # Gates only speech that would execute something. In a records domain
        # there is nothing to gate — which is the point of the comparison.
        return "VERIFY" if (triggers_action and machine.uncertain) else "TRUSTED"
    return decide(machine, expressed, criticality).state.value


def load_labels(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["fixture"]] = row
    return out


def final_turn(payload: dict) -> Turn | None:
    finals = [m for m in payload.get("messages", [])
              if m.get("type") == "Turn" and m.get("end_of_turn")]
    return Turn.from_api(finals[-1]) if finals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--vocab", default="vocab/maintenance.yaml")
    ap.add_argument("--json", help="write the full report as JSON")
    args = ap.parse_args()

    vocab = load_vocabulary(args.vocab)
    labels = load_labels(Path(args.labels))
    files = sorted(Path(args.fixtures).glob("*.json"))
    if not files:
        sys.exit(f"no fixtures in {args.fixtures} — run tools/record_turns.py first")

    signal_rows: list[dict] = []
    calib_scores: list[float] = []
    calib_correct: list[bool] = []
    arm_out = {a: Outcomes() for a in ARMS}
    pol_out = {p: Outcomes() for p in POLICIES}
    hedge_tp = hedge_fp = hedge_fn = 0
    recognition = {"n": 0, "wrong": 0}
    skipped: list[str] = []

    for path in files:
        payload = json.loads(path.read_text())
        turn = final_turn(payload)
        if turn is None:
            skipped.append(f"{path.name}: no final turn")
            continue
        lab = labels.get(path.stem)
        if lab is None:
            skipped.append(f"{path.name}: no label row")
            continue

        slots = find_slots(turn, vocab)
        want_key = lab["slot"]
        match = [s for s in slots if s[0] == want_key]
        if not match:
            skipped.append(f"{path.name}: slot {want_key!r} not extracted")
            continue
        key, s_idx, e_idx, value, as_heard = match[0]

        spec = vocab.spec(key)
        slot = Slot(key=key, cls=spec.cls, value=value, unit=spec.unit,
                    as_heard=as_heard, criticality=spec.criticality,
                    start_idx=s_idx, end_idx=e_idx)

        machine = assess_machine(slot, turn, vocab)
        expressed = assess_expressed(slot, turn)

        correct = str(value).upper() == str(lab["ground_truth"]).upper()
        hedged_truth = lab.get("hedged", "no").strip().lower() == "yes"
        triggers_action = lab.get("triggers_action", "no").strip().lower() == "yes"

        recognition["n"] += 1
        recognition["wrong"] += (not correct)

        # --- STEP 5: which confidence statistic actually discriminates? ---
        confs = [w.confidence for w in turn.words]
        crit = [w.confidence for w in turn.words[s_idx:e_idx]]
        signal_rows.append({
            "min": min(confs), "mean": mean(confs), "median": median(confs),
            "critical": min(crit), "correct": correct, "label": path.stem,
        })
        calib_scores.append(min(crit))
        calib_correct.append(correct)

        # --- hedge detection against the scripted ground truth ---
        if expressed.hedged and hedged_truth:
            hedge_tp += 1
        elif expressed.hedged and not hedged_truth:
            hedge_fp += 1
        elif not expressed.hedged and hedged_truth:
            hedge_fn += 1

        for a in ARMS:
            arm_out[a].add(state=arm_decide(a, machine, expressed, slot.criticality),
                           correct=correct, hedged=hedged_truth, label=path.stem)
        for p in POLICIES:
            pol_out[p].add(state=policy_decide(p, machine, expressed,
                                               slot.criticality, triggers_action),
                           correct=correct, hedged=hedged_truth, label=path.stem)

    report = {
        "fixtures": str(args.fixtures),
        "evaluated": recognition["n"],
        "skipped": skipped,
        "recognition": {
            "critical_value_errors": recognition["wrong"],
            "critical_value_error_rate": (
                round(recognition["wrong"] / recognition["n"], 4)
                if recognition["n"] else None),
        },
        "confidence_signal_comparison": compare_signals(signal_rows) if signal_rows else {},
        "calibration": reliability(calib_scores, calib_correct) if calib_scores else {},
        "hedge_detection": prf(hedge_tp, hedge_fp, hedge_fn),
        "ablation": {a: arm_out[a].report() for a in ARMS},
        "policies": {p: pol_out[p].report() for p in POLICIES},
    }

    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwritten to {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
