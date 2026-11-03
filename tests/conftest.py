import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.claim import Slot                      # noqa: E402
from vocab.schema import load_vocabulary          # noqa: E402


@pytest.fixture(scope="session")
def vocab():
    return load_vocabulary(Path(__file__).resolve().parents[1] / "vocab" / "maintenance.yaml")


@pytest.fixture
def slot_for(vocab):
    """Build a Slot from a turn by locating a token, using the vocab spec."""
    def _make(turn, key, token):
        spec = vocab.spec(key)
        texts = [w.text.strip(".,").upper() for w in turn.words]
        idx = texts.index(token.upper())
        from core.claim import normalise_number
        value = normalise_number([token]) if spec.cls == "quantity" else token.upper()
        return Slot(
            key=key, cls=spec.cls, value=value, unit=spec.unit,
            as_heard=token, criticality=spec.criticality,
            start_idx=idx, end_idx=idx + 1,
        )
    return _make
