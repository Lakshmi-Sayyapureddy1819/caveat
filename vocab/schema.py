"""Vocabulary loading and validation.

The vocabulary is POLICY: slot classes, criticality and gates live here and
are set by a human. Nothing at runtime may mutate a loaded Vocabulary — the
dataclasses are frozen and the mappings are exposed as read-only proxies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import yaml

SLOT_CLASSES = frozenset({"quantity", "identifier", "categorical", "free_text"})
CRITICALITIES = frozenset({"low", "medium", "high"})


class VocabError(ValueError):
    """Raised when a vocabulary file is malformed."""


@dataclass(frozen=True, slots=True)
class SlotSpec:
    key: str
    cls: str
    criticality: str
    gate: float
    unit: str | None
    keyterms: tuple[str, ...]
    known_values: frozenset[str]
    confusables: Mapping[str, tuple[str, ...]]

    def twins_of(self, as_heard: str) -> tuple[str, ...]:
        """Confusable values for a heard token. Case-insensitive on identifiers."""
        if as_heard in self.confusables:
            return self.confusables[as_heard]
        upper = as_heard.upper()
        for k, v in self.confusables.items():
            if k.upper() == upper:
                return v
        return ()


@dataclass(frozen=True, slots=True)
class Vocabulary:
    name: str
    policy_rev: str
    slots: Mapping[str, SlotSpec]
    confusable_boost: float
    confusable_gate_cap: float

    def spec(self, slot_key: str) -> SlotSpec:
        try:
            return self.slots[slot_key]
        except KeyError:
            raise VocabError(f"unknown slot key: {slot_key!r}") from None

    def all_keyterms(self) -> tuple[str, ...]:
        seen: list[str] = []
        for spec in self.slots.values():
            for t in spec.keyterms:
                if t not in seen:
                    seen.append(t)
        return tuple(seen)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise VocabError(msg)


def load_vocabulary(path: str | Path) -> Vocabulary:
    raw = yaml.safe_load(Path(path).read_text())
    _require(isinstance(raw, dict), "vocabulary root must be a mapping")
    _require("slots" in raw and isinstance(raw["slots"], dict), "missing 'slots' mapping")

    defaults = raw.get("defaults") or {}
    default_gate = float(defaults.get("gate", 0.85))
    default_crit = str(defaults.get("criticality", "high"))
    boost = float(defaults.get("confusable_boost", 0.06))
    cap = float(defaults.get("confusable_gate_cap", 0.98))

    specs: dict[str, SlotSpec] = {}
    for key, body in raw["slots"].items():
        _require(isinstance(body, dict), f"slot {key!r} must be a mapping")
        cls = str(body.get("class", "")).strip()
        crit = str(body.get("criticality", default_crit)).strip()
        gate = float(body.get("gate", default_gate))

        _require(cls in SLOT_CLASSES, f"slot {key!r}: bad class {cls!r}")
        _require(crit in CRITICALITIES, f"slot {key!r}: bad criticality {crit!r}")
        _require(0.0 <= gate <= 1.0, f"slot {key!r}: gate {gate} out of [0,1]")

        conf_raw = body.get("confusables") or {}
        _require(isinstance(conf_raw, dict), f"slot {key!r}: confusables must be a mapping")
        confusables = MappingProxyType(
            {str(k): tuple(str(x) for x in v) for k, v in conf_raw.items()}
        )

        specs[key] = SlotSpec(
            key=key,
            cls=cls,
            criticality=crit,
            gate=gate,
            unit=(str(body["unit"]) if body.get("unit") is not None else None),
            keyterms=tuple(str(t) for t in (body.get("keyterms") or ())),
            known_values=frozenset(str(v) for v in (body.get("known_values") or ())),
            confusables=confusables,
        )

    _require(bool(specs), "vocabulary defines no slots")

    return Vocabulary(
        name=str(raw.get("name", "unnamed")),
        policy_rev=str(raw.get("policy_rev", "v0")),
        slots=MappingProxyType(specs),
        confusable_boost=boost,
        confusable_gate_cap=cap,
    )
