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
    assert o["arm"] == "ask" and o["has_evidence_for_task"] is True


def test_a_declared_task_without_evidence_goes_to_COMPUTE_not_refuse(partial):
    """THE rule. work outstanding is not a refusal."""
    for t in ("pathway_transfer", "tissue_state_grouping", "morphology_decode"):
        o = discover({"task_id": t})
        assert o["arm"] == "compute", f"{t} armed {o['arm']}"
        assert o["has_evidence_for_task"] is False
        assert "not run it" in o["arm_reason"]


def test_an_undeclared_task_is_the_only_real_refusal(partial):
    """a declared task with no evidence is work not yet done; an undeclared one
    is a question the system has no method for, and no compute produces one."""
    o = discover({"task_id": "not_a_task_family"})
    assert o["arm"] == "refuse"
    assert "no method to run" in o["arm_reason"]


def test_no_task_named_falls_back_to_the_cohort_level_question(partial):
    o = discover({})
    assert o["arm"] == "ask"
    assert "no task named" in o["arm_reason"]


def test_every_arm_carries_a_reason(partial):
    for t in ("cross_modal_retrieval", "pathway_transfer", "nope", None):
        assert discover({"task_id": t})["arm_reason"], f"{t} armed without a reason"


def test_the_arm_helper_is_pure():
    """resolvable without a cohort on disk, so the rule is testable alone."""
    assert _arm_for("a", {"a": {}}, None)[0] == "ask"
    assert _arm_for(None, {"a": {}}, None)[0] == "ask"
    assert _arm_for(None, {}, None)[0] == "compute"


def test_refuse_is_a_declared_arm():
    """not an error state - a question with no method behind it has an answer."""
    from omicstra.graph import Arm
    import typing

    assert set(typing.get_args(Arm)) == {"ask", "compute", "refuse"}


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
    assert o["arm"] == "compute"
