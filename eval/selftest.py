#!/usr/bin/env python3
"""Harness self-test. Converts the Phase 1 golden fixtures into the capture
format and runs replay.py over them.

THIS MEASURES NOTHING ABOUT ASSEMBLYAI. The confidences are hand-written
values from tests/fixtures.py. Its only purpose is to prove the capture ->
replay -> metrics path works before real audio exists.
"""
import json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict
from tests import fixtures as F

OUT = Path("eval/fixtures/SELFTEST")
OUT.mkdir(parents=True, exist_ok=True)

# fixture -> (slot, ground truth, hedged, triggers_action)
LABELS = {
    "plain_torque":      ("torque",   "45",    "no",  "no"),
    "hedged_torque":     ("torque",   "45",    "yes", "no"),
    "about_torque":      ("torque",   "45",    "yes", "no"),
    "trailing_marker":   ("torque",   "45",    "yes", "no"),
    "asserted_torque":   ("torque",   "45",    "no",  "no"),
    "serial_clear":      ("serial",   "A-104", "no",  "no"),
    "serial_mumbled":    ("serial",   "A-104", "no",  "no"),
    "serial_unknown":    ("serial",   "A-104", "no",  "no"),   # heard Z-999 -> WRONG
    "serial_correction": ("serial",   "A-140", "no",  "no"),
    "pressure_plain":    ("pressure", "80",    "no",  "no"),
    "pressure_hedged":   ("pressure", "80",    "yes", "no"),
    "self_repair":       ("serial",   "A-104", "yes", "no"),
    "injection":         ("torque",   "45",    "no",  "no"),
}

rows = ["fixture,slot,ground_truth,hedged,triggers_action"]
for name, (slot, truth, hedged, act) in LABELS.items():
    turn = F.ALL_FIXTURES[name]
    msg = {
        "type": "Turn",
        "turn_order": turn.turn_order,
        "end_of_turn": True,
        "turn_is_formatted": turn.turn_is_formatted,
        "transcript": turn.transcript,
        "utterance": turn.transcript,
        "end_of_turn_confidence": turn.end_of_turn_confidence,
        "words": [asdict(w) for w in turn.words],
    }
    (OUT / f"{name}.json").write_text(json.dumps(
        {"meta": {"arm": "SELFTEST", "label": name,
                  "WARNING": "synthetic confidences — not AssemblyAI output"},
         "messages": [msg]}, indent=2))
    rows.append(f"{name},{slot},{truth},{hedged},{act}")

Path("eval/labels_selftest.csv").write_text("\n".join(rows) + "\n")
print(f"wrote {len(LABELS)} synthetic captures to {OUT}\n")

subprocess.run([sys.executable, "eval/replay.py",
                "--fixtures", str(OUT),
                "--labels", "eval/labels_selftest.csv"], check=True)
