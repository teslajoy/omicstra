"""the arm is a property of the QUESTION crossed with the cohort.

it read `bool(ev.get("tasks"))`, so a cohort holding evidence for one task of
seven sent all seven to ask - and the six without evidence came back "this
cohort records no evidence for X", a refusal standing where a routing decision
belongs. "that needs compute, and here are the gates" is the true answer.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from omicstra.graph import _arm_for, discover
from omicstra.settings import settings

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "projects" / "tnbc-92"


@pytest.fixture
def partial(tmp_path, monkeypatch):
    """a cohort evaluated on ONE task of seven - the case that was mis-armed."""
    if not (FIXTURE / "routing_evidence.json").is_file():
        pytest.skip("fixture cohort absent")
    for f in ("project.json", "cohort.json", "platform.json", "routing_evidence.json"):
        shutil.copy(FIXTURE / f, tmp_path / f)
    ev = json.loads((tmp_path / "routing_evidence.json").read_text())
    keep = "cross_modal_retrieval"
    ev["tasks"] = {keep: ev["tasks"][keep]}
    (tmp_path / "routing_evidence.json").write_text(json.dumps(ev))
    monkeypatch.setattr(settings, "project_dir", tmp_path)
    return tmp_path


# --- the three arms ---------------------------------------------------------
def test_an_evidenced_task_goes_to_ask(partial):
    o = discover({"task_id": "cross_modal_retrieval"})
    assert o["route"] == "ask" and o["has_evidence_for_task"] is True


def test_a_declared_task_without_evidence_goes_to_COMPUTE_not_refuse(partial):
    """THE rule. work outstanding is not a refusal."""
    for t in ("pathway_transfer", "tissue_state_grouping", "morphology_decode"):
        o = discover({"task_id": t})
        assert o["route"] == "compute", f"{t} routed {o['route']}"
        assert o["has_evidence_for_task"] is False
        assert "not run it" in o["route_reason"]


def test_an_undeclared_task_is_the_only_real_refusal(partial):
    """a declared task with no evidence is work not yet done; an undeclared one
    is a question the system has no method for, and no compute produces one."""
    o = discover({"task_id": "not_a_task_family"})
    assert o["route"] == "refuse"
    assert "no method to run" in o["route_reason"]


def test_no_task_named_falls_back_to_the_cohort_level_question(partial):
    o = discover({})
    assert o["route"] == "ask"
    assert "no task named" in o["route_reason"]


def test_every_arm_carries_a_reason(partial):
    for t in ("cross_modal_retrieval", "pathway_transfer", "nope", None):
        assert discover({"task_id": t})["route_reason"], f"{t} armed without a reason"


def test_the_arm_helper_is_pure():
    """resolvable without a cohort on disk, so the rule is testable alone."""
    assert _arm_for("a", {"a": {}}, None)[0] == "ask"
    assert _arm_for(None, {"a": {}}, None)[0] == "ask"
    assert _arm_for(None, {}, None)[0] == "compute"


def test_refuse_is_a_declared_arm():
    """not an error state - a question with no method behind it has an answer.

    read by ITERATING the vocabulary, not through `typing.get_args`. that is the
    whole reason the arm stopped being a `Literal`: a Literal's members are
    visible to a type checker and to nothing at runtime, so no lint could ask
    whether a consumer handles all of them, and no boundary could refuse a value
    outside them.
    """
    from omicstra.graph import Arm
    from omicstra.labels import Route

    assert Arm is Route
    assert {m.value for m in Arm} == {"ask", "compute", "refuse"}


def test_an_unknown_route_is_refused_rather_than_run_as_compute():
    """`state.get("arm", "compute")` answered a wiring fault with a guess - and
    with the one guess that runs the most work."""
    import pytest

    from omicstra.graph import _arm
    from omicstra.labels import UnknownLabel

    assert _arm({"route": "ask"}) == "ask"
    with pytest.raises(UnknownLabel, match="not a Route"):
        _arm({"route": "evaluate"})    # a HANDLE label in the routing field
    with pytest.raises(UnknownLabel, match="not a Route"):
        _arm({})


# --- the ledger counts arms per task ----------------------------------------
def test_the_ledger_reports_the_arm_for_every_task(partial):
    from omicstra.routing import decision_record

    d = decision_record()
    assert d["arms"]["ask"] == 1
    assert d["arms"]["compute"] == 6
    assert d["answerable_now"] == ["cross_modal_retrieval"]
    assert len(d["needs_compute"]) == 6
    assert set(d["by_task_arm"].values()) <= {"ask", "compute"}


def test_a_fully_evaluated_cohort_needs_no_compute():
    if not (FIXTURE / "routing_evidence.json").is_file():
        pytest.skip("fixture cohort absent")
    from omicstra.routing import decision_record

    d = decision_record(project_id="tnbc-92")
    assert d["arms"]["compute"] == 0
    assert len(d["answerable_now"]) == 7


def test_a_cohort_level_count_cannot_answer_this(partial):
    """the shape of the defect, kept as a test: `has_evidence` is True while six
    of seven questions are unanswerable, so the cohort-level flag hides exactly
    what a caller needs to know."""
    o = discover({"task_id": "pathway_transfer"})
    assert o["has_evidence"] is True          # the cohort has SOME evidence
    assert o["has_evidence_for_task"] is False   # and none for this question
    assert o["route"] == "compute"


# --- the rename: one key, one meaning ---------------------------------------
def test_discover_no_longer_writes_the_colliding_key():
    """the read alias is the harmless half. a value `discover` still WROTE would
    keep landing in every subgraph's `arm` by name, which is the defect - the
    encode state declares `arm` meaning he|st and would have received
    `compute`."""
    from omicstra.graph import discover

    out = discover({"task_id": "cross_modal_retrieval"})
    assert "route" in out and out["route_reason"]
    assert "arm" not in out, "the parent must not hand a routing label to a subgraph"
    assert "arm_reason" not in out


def test_the_two_vocabularies_no_longer_share_a_field_name():
    """`arm` means exactly one thing in the codebase now: the handle arm.

    the parent declares neither `arm` nor a read alias for it - an earlier
    version of this docstring said it kept the name to read an old checkpoint,
    which stopped being true when the alias came out. the subgraphs keep `arm`
    and must not carry `route`, which is the half asserted here; that the parent
    carries neither is asserted in test_graph_wiring.py.
    """
    from omicstra.graphs.align import AlignState
    from omicstra.graphs.encode import EncodeState
    from omicstra.graphs.evaluate import EvaluateState

    for state in (EncodeState, AlignState, EvaluateState):
        assert "arm" in state.__annotations__
        assert "route" not in state.__annotations__, \
            f"{state.__name__} must not carry a routing label"
