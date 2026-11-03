# Caveat — core

A voice data integrity layer. Decides whether a spoken value is safe to write
down, and records why.

Two independent kinds of doubt, routed to different behaviours:

| | trigger | outcome |
|---|---|---|
| **Machine uncertainty** | min word confidence below the slot's gate, or an identifier outside the vocabulary | `VERIFY` — the agent asks |
| **Expressed uncertainty** | the speaker used an epistemic marker ("I think", "about") or self-corrected | `UNCONFIRMED` — recorded honestly, no challenge |

## Phase 1 status — complete

```
Turn fixture → claim extraction → machine uncertainty → expressed uncertainty
             → deterministic policy → trust state → evidence event
```

Runs with no microphone, no WebSocket, no AssemblyAI call, no LLM call,
no database and no frontend.

```bash
pip install pytest pyyaml
python3 -m pytest tests/ -q      # 70 passed
python3 demo_core.py             # the three milestone demos
```

## Guarantees, enforced by tests

- `core/` imports nothing that touches a network or a filesystem — verified by
  an AST import check, and the pipeline runs with `socket.socket` disabled.
- `policy.decide` is pure: same inputs, same output, 50/50 runs identical.
- The strings `TRUSTED` / `VERIFY` / `UNCONFIRMED` appear in no model output
  schema. A spoken prompt injection cannot promote a claim because there is no
  channel from a model to the trust state.
- There is no `UNCONFIRMED → TRUSTED` edge. A claim is trusted at birth or never.
- A correction clears the trust state and must re-enter the policy engine.
- `Claim.to_wire()` is the only way out and always carries `trust`, `reason`,
  `said` and `evidence_ref`. An undecided claim raises rather than escaping.
- The evidence ledger is append-only and hash-chained per claim; tampering with
  a payload breaks `verify_chain`. Model reasoning is rejected at write time.

## Expressed uncertainty — what we deliberately do not do

No prosody, pause duration, speech rate, emotion or lie detection. Filled-pause
behaviour has documented dual status and varies by genre and by native-speaker
status (Kosmala & Crible, *Language and Speech* 65(1), 2022), so a timing-based
detector would systematically flag L2 speakers and people who stutter. We detect
what the speaker **said**, not what they internally believed. A test asserts no
timing signal is used.

## Layout

```
vocab/     maintenance.yaml   policy: slot classes, gates, criticality, confusables
           schema.py          loader + validation, frozen and immutable
core/      claim.py           AssemblyAI Turn/Word types, Claim schema, spanning
           machine.py         machine-uncertainty engine
           expressed.py       expressed-uncertainty engine
           policy.py          the pure decision function
           states.py          trust state machine
           ledger.py          append-only hash-chained events
           pipeline.py        wiring, dependency-injected clock and ids
tests/     fixtures.py        golden Turns in the exact AssemblyAI shape
```

## Not built, deliberately

Tool gateway, multi-agent orchestration, MCP, Kafka, Redis, dashboard, RAG,
emotion detection, prosody-based uncertainty. Caveat writes records; it does
not execute actions.

## Next — Phase 2

AssemblyAI `universal-3-5-pro` v3 streaming, browser mic, bounded claim
extraction, `UpdateConfiguration` keyterm injection on `VERIFY`, read API.
