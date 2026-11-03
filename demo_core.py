#!/usr/bin/env python3
"""Milestone demo — the frozen spec's first end-to-end target, fully offline.

  python3 demo_core.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.ledger import Ledger
from core.pipeline import counter, process_turn
from vocab.schema import load_vocabulary
from tests import fixtures as F

B, D, R = "\033[1m", "\033[2m", "\033[0m"
COL = {"TRUSTED": "\033[32m", "UNCONFIRMED": "\033[35m",
       "VERIFY": "\033[33m", "CONFIRMED": "\033[32m", "CORRECTED": "\033[36m"}
NOW = "2026-09-14T11:04:22.318Z"

vocab = load_vocabulary(Path(__file__).parent / "vocab" / "maintenance.yaml")
ledger = Ledger()
next_id = counter()


def why(c):
    st = c.trust.state.value
    print(f"\n  {B}{c.slot.key.upper()} = {c.slot.value} {c.slot.unit or ''}{R}"
          f"   {COL[st]}{B}{st}{R}")
    m, e = c.machine, c.expressed
    mark = "✓" if not m.uncertain else "⚠"
    print(f"    Machine      {mark}  min conf {m.min_word_confidence:.2f} vs gate {m.gate:.2f}"
          + (f"   twin: {m.confusable_with[0]}" if m.confusable_with else ""))
    if not m.in_keyterm_set:
        print(f"                 ⚠  '{c.slot.value}' is not in the parts list")
    mark = "⚠" if e.hedged else "✓"
    detail = f'said "{e.marker}"' if e.marker else ("self-correction" if e.self_repair else "no hedge")
    print(f"    Speaker      {mark}  {detail}")
    print(f"    Criticality     {c.slot.criticality}  (from maintenance.yaml)")
    print(f"    {D}Decision: {st} — {c.trust.reason} — policy {c.trust.policy_rev}{R}")


def run(title, turn):
    print(f"\n{B}{'─'*66}{R}\n{B}{title}{R}")
    print(f'  {D}"{turn.transcript}"{R}')
    cs = process_turn(turn, vocab, ledger, "ses_1", NOW, next_id)
    for c in cs:
        why(c)
    return cs


# ── DEMO 1 — expressed uncertainty ─────────────────────────────────────
c1 = run("DEMO 1  ·  the speaker hedged", F.T_HEDGED_TORQUE)[0]

print(f"\n{B}  Later, a downstream reader asks: \"What was the torque?\"{R}")
w = c1.to_wire(said="I think the torque was 45 Nm")
print(f'  → "{w["value"]} {w["unit"]}, but it was recorded as {w["trust"]} '
      f'because the\n     technician said {chr(34)}{c1.expressed.marker}{chr(34)}."')
print(f"  {D}wire: {w}{R}")

# ── DEMO 2 — machine uncertainty + correction ──────────────────────────
c2 = run("DEMO 2  ·  the machine was unsure", F.T_SERIAL_MUMBLED)[0]
print(f"\n  {B}Agent asks:{R} \"Did you say A-104 or A-140?\"")
print(f"  {D}keyterms to inject on UpdateConfiguration: "
      f"{[c2.slot.as_heard] + list(c2.machine.confusable_with)}{R}")

c3 = run("        ·  the human corrects it", F.T_SERIAL_CORRECTION)[0]
print(f"\n  {B}A-104 → A-140{R}, re-gated through the policy engine → "
      f"{COL[c3.trust.state.value]}{c3.trust.state.value}{R}")

# ── DEMO 3 — prompt injection ──────────────────────────────────────────
c4 = run("DEMO 3  ·  spoken prompt injection", F.T_INJECTION)[0]
print(f"\n  The injected text is in the transcript and in the evidence record.")
print(f"  The trust state is {COL[c4.trust.state.value]}{c4.trust.state.value}{R} — "
      f"identical to the clean utterance.")
print(f"  {D}No model output field in this system can name a trust state.{R}")

# ── evidence ───────────────────────────────────────────────────────────
print(f"\n{B}{'─'*66}{R}\n{B}EVIDENCE LEDGER  ·  claim {c1.claim_id}{R}\n")
for ev in ledger.events(c1.claim_id):
    print(f"  {ev.type.value:<20} {D}{ev.hash[:12]}…  prev {ev.prev_hash[:8]}…{R}")
    for k, val in ev.payload.items():
        print(f"      {D}{k}: {val}{R}")
print(f"\n  chain verified: {ledger.verify_chain(c1.claim_id)}")
print(f"  total events across all claims: {len(ledger.events())}\n")
