#!/usr/bin/env python3
"""Print the schema AssemblyAI actually sent for one raw capture, and apply
the Phase 2 decision gate on machine (word) confidence spread.

Usage:
    python3 tools/inspect_turn.py eval/fixtures/<ARM>/<file>.json

Exit codes (see RUN_PHASE2.md decision gate):
    0   word confidence varies (spread >= 0.05)      -> proceed to A2/A3/A4
    1   spread < 0.05                                 -> stop, report the numbers
    2   confidence is constant, or no Turn ever arrived -> stop, no signal
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SPREAD_GATE = 0.05


def schema_of(value):
    """Shape only — keys and types, no values. What was actually sent."""
    if isinstance(value, dict):
        return {k: schema_of(v) for k, v in value.items()}
    if isinstance(value, list):
        return [schema_of(value[0])] if value else []
    return type(value).__name__


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: inspect_turn.py <fixture.json>")

    path = Path(sys.argv[1])
    if not path.is_file():
        sys.exit(f"no such file: {path}")

    payload = json.loads(path.read_text())
    messages = payload.get("messages", [])

    print(f"file: {path}")
    print(f"arm/label: {payload.get('meta', {}).get('arm')} / {payload.get('meta', {}).get('label')!r}")
    print(f"messages received: {len(messages)}")
    for msg in messages:
        print(f"  - type={msg.get('type')!r} schema={json.dumps(schema_of(msg))}")

    turns = [m for m in messages if m.get("type") == "Turn"]
    final_turns = [t for t in turns if t.get("end_of_turn")]

    if not turns:
        print("\nNO_DATA: no Turn message ever arrived — nothing to assess.")
        sys.exit(2)

    words = []
    for t in final_turns or turns:
        words.extend(t.get("words", []))

    confidences = [w["confidence"] for w in words if "confidence" in w]

    if not confidences:
        print("\nNO_DATA: Turn message(s) present but no word confidences found.")
        sys.exit(2)

    lo, hi = min(confidences), max(confidences)
    spread = hi - lo

    print(f"\nword confidences: n={len(confidences)} min={lo:.4f} max={hi:.4f} spread={spread:.4f}")

    if spread == 0:
        print("CONSTANT: every word has identical confidence — machine axis has no signal.")
        sys.exit(2)
    if spread < SPREAD_GATE:
        print(f"LOW_SPREAD: spread {spread:.4f} < gate {SPREAD_GATE} — report and stop.")
        sys.exit(1)

    print(f"VARIES: spread {spread:.4f} >= gate {SPREAD_GATE} — proceed to A2, A3, A4.")
    sys.exit(0)


if __name__ == "__main__":
    main()
