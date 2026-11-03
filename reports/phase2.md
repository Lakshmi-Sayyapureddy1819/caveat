# Phase 2 — BLOCKED

**Date:** 5 September 2026
**Status:** no measurements taken. No AssemblyAI call was made.

---

## 1. Blocker

Phase 2 is blocked because no real AssemblyAI API key is available, and two
further blockers would prevent the experiment even if a key were supplied.

| Requirement | Status | Evidence |
|---|---|---|
| `ASSEMBLYAI_API_KEY` | **absent** | not in environment; no `.env` anywhere in the tree |
| Microphone | **absent** | `/dev/snd` does not exist; PortAudio not installed |
| Network reach to AssemblyAI | **denied by policy** | gateway answers `403` to `CONNECT` for `api.assemblyai.com:443`, `streaming.assemblyai.com:443` and `www.assemblyai.com:443` |
| Bridge to a local machine | **not usable** | present, but no folder connected; its shell follows the same egress allowlist, and it is an isolated VM without desktop audio |

Per the Phase 2 instruction, **no synthetic confidence values have been
substituted for measurements**, and no numbers appear in this report that
were not produced by code running on data.

The experiment must be run on a machine with a microphone, a real key, and
unrestricted network access to `streaming.assemblyai.com`.

---

## 2. What was completed — API verification (Step 1)

Verified against the current official streaming API reference. Three findings
change the Phase 1 code:

| Finding | Consequence |
|---|---|
| **`format_turns` is Universal-only.** On `universal-3-5-pro`, `turn_is_formatted` tracks `end_of_turn`. | There is no formatted/unformatted pair on Pro. The frozen spec's "analyse the unformatted turn, display the formatted one" does not apply — there is one text. `format_turns` removed from the capture tool. |
| **`utterance`, `speaker_label`, `language_code`, `language_confidence`** are real Turn fields. | Added to `core.claim.Turn`. `utterance` is populated only when `end_of_turn` is true. |
| **`keyterms_prompt` accepts at most 100 terms**; `prompt` and `agent_context` are capped at 1750 characters and are Pro-only. | Enforced in `tools/record_turns.py`; the arms are sized within these limits. |

Confirmed as previously assumed: per-word `confidence`, `start`, `end`,
`word_is_final`; `UpdateConfiguration` can change `prompt`, `keyterms_prompt`
and `agent_context` mid-session; `previous_context_n_turns` (default 5, range
0–100) controls context carryover; the client terminate message is
`{"type":"Terminate"}`. **No alternative hypotheses are returned** — one
hypothesis only, so `confusable_with` must come from the vocabulary.

---

## 3. What was completed — the experiment harness

Everything needed to run Phase 2 exists and is tested. The moment a key and a
microphone are available, four commands produce the dataset.

```
tools/record_turns.py     mic -> Universal-3.5 Pro -> raw Turn JSON, verbatim
tools/diagnostic_20.txt   the 20 diagnostic utterances
eval/replay.py            captured JSON -> Caveat arms -> metrics, offline
eval/metrics.py           AUC, reliability/ECE, Wilson intervals, outcome counts
eval/selftest.py          proves the capture->replay->metrics path works
```

The four AssemblyAI arms are kept strictly separate, as instructed:
`A1` base · `A2` + `prompt` · `A3` + `keyterms_prompt` · `A4` + `agent_context`.

**The critical property holds:** once Turn JSON is captured, every threshold
and ablation experiment replays from disk with no microphone and no further
AssemblyAI call.

### Step 5 is answered by data, not by the specification

`eval/metrics.compare_signals` computes rank-based AUC for **minimum**, **mean**,
**median** and **critical-token** confidence against actual correctness, and
reports which discriminates. It returns `_best: None` when nothing beats
chance, so a null result is visible rather than hidden behind a least-bad
winner. If minimum confidence loses, `core/machine.py` changes.

---

## 4. Two real bugs found and fixed

Both surfaced from running the harness, not from reading the code.

**Self-correction extracted the discarded value.** `find_slots` returned the
*first* matching identifier, so `"The serial is A-140... sorry, A-104"` yielded
`A-140`. The frozen spec requires the final value to win. Fixed; regression
test added.

**The signal comparison crowned a chance-level winner.** With every signal at
or below AUC 0.5, `_best` still named one. It now returns `None` and sets
`_discriminating: False`. Two tests added.

---

## 5. Test status

```
87 passed          (70 Phase 1 + 14 metrics + 3 regression)
core/ imports      dataclasses, enum, hashlib, json, re, typing — nothing else
                   pipeline verified to run with socket.socket disabled
secrets            no key-shaped literals in the tree; key read from the
                   environment only; .env in .gitignore
```

---

## 6. Sections that cannot be filled

Dataset · recognition results · confidence-vs-correctness · calibration ·
Caveat results · ablation · keyterm loop · latency — **all require real audio.**

The `eval/selftest.py` run produces numbers, but they come from hand-written
confidences in `tests/fixtures.py`. They demonstrate that the harness computes
and reports correctly. **They say nothing whatsoever about AssemblyAI and must
never be quoted as results.**

---

## 7. Decision

**YELLOW — blocked, not failed.**

Not GREEN: no evidence exists yet.
Not RED: nothing has been falsified; the API exposes every field the design
depends on, which was the main technical risk.

The gates in `vocab/maintenance.yaml` (0.85 quantity, 0.88 identifier, +0.06
confusable boost) are **placeholders calibrated against invented fixture
values.** They must be re-derived from the A1–A4 captures before any result is
reported.

---

## 8. To unblock

On a machine with a microphone, network access and a key:

```bash
pip install websocket-client sounddevice numpy pyyaml pytest
export ASSEMBLYAI_API_KEY=...

python3 -m pytest tests/ -q                                   # expect 87 passed
python3 ../caveat-day1/confidence_test.py                      # the 2-hour probe

for arm in A1 A2 A3 A4; do
  python3 tools/record_turns.py --arm $arm --script tools/diagnostic_20.txt
done

python3 eval/replay.py --fixtures eval/fixtures/A4 --labels eval/labels.csv
```

Then this report is rewritten with real numbers.
