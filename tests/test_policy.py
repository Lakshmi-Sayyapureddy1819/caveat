"""Every combination of machine x speaker x criticality, plus purity."""
import itertools

import pytest

from core.claim import ExpressedVerdict, MachineVerdict, TrustState
from core.policy import decide


def mach(uncertain: bool) -> MachineVerdict:
    return MachineVerdict(
        min_word_confidence=0.60 if uncertain else 0.95,
        gate=0.85, in_keyterm_set=not uncertain, confusable_with=(),
        verdict="uncertain" if uncertain else "certain",
    )


def expr(hedged: bool) -> ExpressedVerdict:
    return ExpressedVerdict(
        marker="i think" if hedged else None, marker_span=None,
        self_repair=False, verdict="hedged" if hedged else "plain",
    )


# machine_uncertain, speaker_hedged, criticality -> expected state, reason
TABLE = [
    (False, False, "high",   TrustState.TRUSTED,     "clear"),
    (False, True,  "high",   TrustState.UNCONFIRMED, "speaker_hedged"),
    (True,  False, "high",   TrustState.VERIFY,      "machine_uncertain"),
    (True,  True,  "high",   TrustState.VERIFY,      "machine_uncertain"),
    (False, False, "medium", TrustState.TRUSTED,     "clear"),
    (False, True,  "medium", TrustState.UNCONFIRMED, "speaker_hedged"),
    (True,  False, "medium", TrustState.VERIFY,      "machine_uncertain"),
    (True,  True,  "medium", TrustState.VERIFY,      "machine_uncertain"),
    (False, False, "low",    TrustState.TRUSTED,     "low_criticality"),
    (False, True,  "low",    TrustState.TRUSTED,     "low_criticality"),
    (True,  False, "low",    TrustState.TRUSTED,     "low_criticality"),
    (True,  True,  "low",    TrustState.TRUSTED,     "low_criticality"),
]


@pytest.mark.parametrize("mu,sh,crit,state,reason", TABLE)
def test_policy_table(mu, sh, crit, state, reason):
    d = decide(mach(mu), expr(sh), crit)
    assert d.state is state
    assert d.reason == reason


def test_machine_doubt_outranks_hedging():
    """If we may have the wrong value, asking beats flagging."""
    assert decide(mach(True), expr(True), "high").state is TrustState.VERIFY


def test_hedged_never_becomes_verify():
    """Challenging someone who already said they were unsure is pointless."""
    assert decide(mach(False), expr(True), "high").state is TrustState.UNCONFIRMED


def test_is_pure_and_deterministic():
    for mu, sh, crit in itertools.product([True, False], [True, False],
                                          ["low", "medium", "high"]):
        runs = {decide(mach(mu), expr(sh), crit) for _ in range(50)}
        assert len(runs) == 1


def test_policy_module_has_no_io_or_model_imports():
    """Purity enforced structurally: inspect the import graph, not the prose."""
    import ast
    import inspect

    from core import policy

    tree = ast.parse(inspect.getsource(policy))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    banned = {"os", "sys", "socket", "random", "time", "datetime", "sqlite3",
              "requests", "httpx", "openai", "anthropic", "aiohttp", "urllib",
              "pathlib", "subprocess"}
    assert not (imported & banned), f"policy imports {sorted(imported & banned)}"
    assert imported <= {"__future__", "core"}, f"unexpected imports: {sorted(imported)}"


def test_policy_rev_is_recorded():
    assert decide(mach(False), expr(False), "high", policy_rev="v9").policy_rev == "v9"
