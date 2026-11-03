"""Append-only evidence ledger.

Six event types. Hash-chained per claim. The current claim state is a fold
over its events.

Never stored: LLM chain-of-thought. Stored: transcript, extracted span,
evidence, decision, policy version, correction, final state.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

GENESIS = "0" * 64


class EventType(str, Enum):
    ClaimExtracted = "ClaimExtracted"
    UncertaintyAssessed = "UncertaintyAssessed"
    TrustDecided = "TrustDecided"
    VerificationAsked = "VerificationAsked"
    ClaimAnswered = "ClaimAnswered"
    ClaimCommitted = "ClaimCommitted"


_FORBIDDEN_KEYS = frozenset(
    {"reasoning", "chain_of_thought", "thoughts", "rationale", "scratchpad"}
)


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    claim_id: str
    type: EventType
    timestamp: str
    actor: str            # "system" | "human"
    payload: Mapping[str, Any]
    prev_hash: str
    hash: str


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(claim_id: str, etype: EventType, timestamp: str,
                 actor: str, payload: Mapping[str, Any], prev_hash: str) -> str:
    body = _canonical(
        {
            "claim_id": claim_id,
            "type": etype.value,
            "timestamp": timestamp,
            "actor": actor,
            "payload": payload,
            "prev_hash": prev_hash,
        }
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class Ledger:
    """In-memory append-only log. Persistence is a later concern."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._heads: dict[str, str] = {}
        self._seq = 0

    # -- write ------------------------------------------------------------
    def append(self, claim_id: str, etype: EventType, payload: Mapping[str, Any],
               timestamp: str, actor: str = "system") -> Event:
        bad = _FORBIDDEN_KEYS & set(payload)
        if bad:
            raise ValueError(f"model reasoning must not be stored: {sorted(bad)}")
        if actor not in ("system", "human"):
            raise ValueError(f"actor must be 'system' or 'human', got {actor!r}")

        prev = self._heads.get(claim_id, GENESIS)
        h = compute_hash(claim_id, etype, timestamp, actor, payload, prev)
        self._seq += 1
        ev = Event(
            event_id=f"evt_{self._seq:06d}",
            claim_id=claim_id,
            type=etype,
            timestamp=timestamp,
            actor=actor,
            payload=dict(payload),
            prev_hash=prev,
            hash=h,
        )
        self._events.append(ev)
        self._heads[claim_id] = h
        return ev

    # -- read -------------------------------------------------------------
    def events(self, claim_id: str | None = None) -> tuple[Event, ...]:
        if claim_id is None:
            return tuple(self._events)
        return tuple(e for e in self._events if e.claim_id == claim_id)

    def head(self, claim_id: str) -> str:
        return self._heads.get(claim_id, GENESIS)

    def verify_chain(self, claim_id: str) -> bool:
        prev = GENESIS
        for ev in self.events(claim_id):
            if ev.prev_hash != prev:
                return False
            expect = compute_hash(
                ev.claim_id, ev.type, ev.timestamp, ev.actor, ev.payload, ev.prev_hash
            )
            if ev.hash != expect:
                return False
            prev = ev.hash
        return True

    def fold(self, claim_id: str) -> dict[str, Any]:
        """Derive current claim state from its events."""
        state: dict[str, Any] = {"claim_id": claim_id, "events": 0}
        for ev in self.events(claim_id):
            state["events"] += 1
            if ev.type is EventType.ClaimExtracted:
                state.update(
                    slot=ev.payload.get("slot"),
                    value=ev.payload.get("value"),
                    as_heard=ev.payload.get("as_heard"),
                    transcript=ev.payload.get("transcript"),
                )
            elif ev.type is EventType.TrustDecided:
                state.update(
                    trust=ev.payload.get("state"),
                    reason=ev.payload.get("reason"),
                    policy_rev=ev.payload.get("policy_rev"),
                )
            elif ev.type is EventType.ClaimAnswered:
                if ev.payload.get("outcome") == "corrected":
                    state["value"] = ev.payload.get("new_value")
                state["outcome"] = ev.payload.get("outcome")
            elif ev.type is EventType.ClaimCommitted:
                state.update(
                    value=ev.payload.get("final_value"),
                    trust=ev.payload.get("final_state"),
                    committed=True,
                )
        return state
