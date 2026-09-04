"""the protocol must cover the contract, and agree with it.

this exists because it drifted once already: six of ten checks registered, two of
the missing four REQUIRED, and one authority disagreeing. none of it was visible
from the record, because an unregistered check does not appear at all - it is not
`not_run`, it is absent. a missing check that says nothing is the defect the whole
record contract exists to prevent, reappearing one level up.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from omicstra.protocols import ProtocolOrderError, Step, build_protocol
from omicstra.protocols.eda import EDA_STEPS
from omicstra.protocols.inventory import INVENTORY_STEPS

CONTRACT = json.loads(
    (Path(__file__).resolve().parents[1] / "configs" / "eda_contract.json").read_text()
)
CHECKS = {c["id"]: c for c in CONTRACT["checks"]}
REGISTERED = {s.id: s for s in EDA_STEPS}


def test_every_contract_check_is_registered():
    """an unregistered check is INVISIBLE - not not_run, absent."""
    missing = sorted(set(CHECKS) - set(REGISTERED))
    assert not missing, (
        f"{len(missing)} contract check(s) unregistered: {missing}. a check that is "
        f"not registered produces no record at all, so the gate cannot judge it and "
        f"nobody can see that it was skipped."
    )


def test_no_registered_step_is_absent_from_the_contract():
    """the reverse: a step nobody declared has no criterion to be judged against."""
    extra = sorted(set(REGISTERED) - set(CHECKS))
    assert not extra, f"registered but undeclared: {extra}"


@pytest.mark.parametrize("cid", sorted(CHECKS))
def test_authority_agrees_with_the_contract(cid):
    """authority decides whether a bar derived elsewhere ESCALATES on a new cohort.

    declaring a cohort_calibrated check as universal makes it judge a new cohort
    against another cohort's threshold without asking - silently.
    """
    want = CHECKS[cid].get("authority")
    got = REGISTERED[cid].authority
    assert got == want, f"{cid}: contract says {want!r}, step declares {got!r}"


@pytest.mark.parametrize("cid", [c for c, v in CHECKS.items() if v.get("applies_when")])
def test_conditional_checks_have_an_applicability_predicate(cid):
    """`applies_when` in the contract means the step must be able to say
    not_applicable. without a predicate it would report not_run instead, which
    means 'nobody looked' rather than 'it cannot apply here'."""
    assert REGISTERED[cid].applicable_when is not None, (
        f"{cid} declares applies_when={CHECKS[cid]['applies_when']!r} but the step "
        f"has no applicable_when predicate")


def test_required_checks_are_all_present():
    req = {c for c, v in CHECKS.items() if v.get("required")}
    assert req <= set(REGISTERED), f"required and unregistered: {sorted(req - set(REGISTERED))}"


# --- the driver's own invariants --------------------------------------------
def test_protocol_refuses_a_bad_order():
    """an invalid order is a wrong METHOD, so it must fail before data is touched."""
    with pytest.raises(ProtocolOrderError):
        build_protocol(
            [Step(id="moran", fn=lambda c: None, requires=frozenset({"weights"})),
             Step(id="knn", fn=lambda c: None, produces=frozenset({"weights"}))],
            "bad")


def test_protocol_accepts_a_good_order():
    assert build_protocol(
        [Step(id="knn", fn=lambda c: None, produces=frozenset({"weights"})),
         Step(id="moran", fn=lambda c: None, requires=frozenset({"weights"}))],
        "good") is not None


@pytest.mark.parametrize("steps,name", [(EDA_STEPS, "eda"), (INVENTORY_STEPS, "inventory")])
def test_shipped_protocols_build(steps, name):
    """both real protocols must satisfy their own order assertion."""
    assert build_protocol(steps, name) is not None


# --- the two bugs that shipped, now covered ---------------------------------
def test_inventory_halts_on_any_failure_not_only_unbound_roles():
    """a failed inventory must stop eda.

    the halt originally read only `bind.unbound_required`. when `platform`
    failed, `bind` was not_run, its unbound list was empty, and the run
    proceeded - letting eda measure data inventory could not describe.
    """
    import inspect
    from omicstra import eda_graph
    src = inspect.getsource(eda_graph.inventory)
    assert 'status") == "fail"' in src or "status') == 'fail'" in src, (
        "inventory must halt on any failed step, not only on unbound roles")
    assert "unbound or failed" in src, "the halt must be the union of both conditions"


@pytest.mark.parametrize("proj", ["projects/tnbc-92"])
def test_platform_json_has_the_flat_samples_map(proj):
    """platform.json must map sample_id -> platform key.

    it was first written with sample_ids nested inside each platform. the
    inventory step requires the flat map, so every sample read as undeclared
    and the whole cohort halted for a schema reason, not a data one.
    """
    p = Path(__file__).resolve().parents[1] / proj / "platform.json"
    if not p.exists():
        pytest.skip(f"{proj} has no platform.json")
    d = json.loads(p.read_text())
    assert "samples" in d, "platform.json needs a flat samples map"
    assert isinstance(d["samples"], dict) and d["samples"], "samples must be non-empty"
    undefined = set(d["samples"].values()) - set(d.get("platforms", {}))
    assert not undefined, f"samples name undefined platforms: {sorted(undefined)}"
    for key, spec in d.get("platforms", {}).items():
        assert "sample_ids" not in spec, (
            f"{key} still nests sample_ids - that is the superseded shape")


def test_eda_graph_has_an_inventory_node_before_profile():
    """inventory produces the join key, platform and raw matrix that eda reads."""
    from omicstra.eda_graph import build_eda_graph
    nodes = set(build_eda_graph().get_graph().nodes)
    assert {"inventory", "profile", "gate"} <= nodes, f"nodes: {sorted(nodes)}"


def test_step_registry_is_gone_from_the_graph():
    """the registry lives in protocols/. a graph that also owns one is two things."""
    from omicstra import eda_graph
    assert not hasattr(eda_graph, "STEP_REGISTRY")
