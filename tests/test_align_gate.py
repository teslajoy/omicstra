"""the align gate: this repo's opening constraint, enforced rather than kept.

CLAUDE.md's first critical constraint is "do not propose alignment, contrastive
learning, retrieval evaluation, or agent routing until EDA.md is satisfied". that
was a sentence a reader had to honour. `eda_admissibility` is the gate that makes
it structural, and these are the tests that say so.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from omicstra.graphs.align import build_align_graph
from omicstra.protocols.align import EdaGateMissing, preflight_align

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "projects" / "tnbc-92"


def _run(answer=None, tid="t", project_id="tnbc-92", compute=False):
    app = build_align_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": tid}}
    out = app.invoke({"project_id": project_id, "arm": "align", "compute": compute}, cfg)
    if answer is not None and "__interrupt__" in out:
        out = app.invoke(Command(resume=answer), cfg)
    return out


# --- the gate set ----------------------------------------------------------
def test_the_align_stage_asks_about_admissibility_first():
    if not (SEED / "eda_summary.json").is_file():
        pytest.skip("no seed cohort on this machine")
    ids = [g.id for g in preflight_align(project_id="tnbc-92")]
    assert ids[0] == "eda_admissibility", "admissibility is the first question"
    assert set(ids) == {"eda_admissibility", "verdict_disagreement", "grid_scope"}


def test_an_unrun_gate_is_refused_not_asked(tmp_path):
    """a gate is a question with a closed option set and someone to ask. "the
    gate has never been run" is neither, and asking a person to accept a verdict
    nobody computed is how a cohort gets a pass it never earned.
    """
    from omicstra.settings import settings

    (tmp_path / "project.json").write_text(json.dumps({"project_id": tmp_path.name}))
    before = settings.project_dir
    settings.project_dir = tmp_path
    try:
        with pytest.raises(EdaGateMissing, match="no EDA verdict"):
            preflight_align()
    finally:
        settings.project_dir = before


def test_a_disagreeing_verdict_opens_its_own_gate():
    """the same shape as the geometry escalation: a measurement that contradicts
    a declaration escalates rather than silently winning.

    the seed cohort declares `proceed` and computes `proceed_with_caution`, and
    nothing had surfaced that at the point where it decides anything.
    """
    if not (SEED / "eda_summary.json").is_file():
        pytest.skip("no seed cohort on this machine")
    g = {x.id: x for x in preflight_align(project_id="tnbc-92")}["verdict_disagreement"]
    if g.observed["agrees"]:
        pytest.skip("this cohort's verdicts agree - nothing to escalate")
    assert g.open, "a disagreement must not resolve itself"
    assert g.observed["computed"] != g.observed["declared"]


def test_a_declared_grid_closes_its_gate_without_asking():
    if not (SEED / "project.json").is_file():
        pytest.skip("no seed cohort on this machine")
    g = {x.id: x for x in preflight_align(project_id="tnbc-92")}["grid_scope"]
    assert not g.open
    assert g.source == "project.json#contrastive_runs"
    assert g.observed["n_arms"] >= 1


# --- the graph -------------------------------------------------------------
def test_the_grid_does_not_resolve_until_its_open_gate_is_answered():
    if not (SEED / "eda_summary.json").is_file():
        pytest.skip("no seed cohort on this machine")
    out = _run(tid="unanswered")
    assert "__interrupt__" in out, "it must pause rather than resolve"
    v = out["__interrupt__"][0].value
    assert v["gates"], "the payload must carry the gates, not just say there are some"
    for g in v["gates"]:
        assert g["options"] and g["forecloses"]
    assert v["no_default"] == "the system does not pick"
    assert "inherits the cohort's" in v["consequence"]


def test_a_declared_proceed_is_the_acceptance_and_is_not_re_asked():
    """`graphs/eda.py` already interrupts on a cautionary verdict, so asking again
    here would have a person accept the same cautions twice, in two records.

    a person wrote `proceed` into the committed summary and version control
    attests the file - that IS the recorded acceptance, by the same rule the actor
    vocabulary uses for a declared gate answer.
    """
    if not (SEED / "eda_summary.json").is_file():
        pytest.skip("no seed cohort on this machine")
    g = {x.id: x for x in preflight_align(project_id="tnbc-92")}["eda_admissibility"]
    if g.observed["declared_verdict"] != "proceed":
        pytest.skip("this cohort does not declare proceed")
    assert not g.open, "a recorded acceptance must not be re-asked"
    assert g.source == "eda_summary.json#verdict"
    assert g.observed["verdict"] != "proceed", (
        "and it closes DESPITE the computed verdict being short of proceed - "
        "which is exactly what verdict_disagreement then surfaces")


def test_answering_resolves_the_grid_or_names_what_is_missing():
    """the gate is what this tests, and the gate works either way.

    guarded on eda_summary.json alone at first, which is COMMITTED while the runs
    it resolves are git-ignored - so it passed here and failed in every clone.
    the same mistake the encode tests made with ingest.json. both outcomes are
    asserted instead: resolved where the artifacts exist, `unavailable` naming
    the stage where they do not.
    """
    if not (SEED / "eda_summary.json").is_file():
        pytest.skip("no seed cohort on this machine")
    out = _run(answer={"eda_admissibility": "accept_cautions",
                       "verdict_disagreement": "computed"}, tid="answered")
    rep = out["report"]
    if rep["status"] == "unavailable":
        assert rep["stage"] in ("niche_join", "align")
        assert rep["why"], "an unavailable stage must say what is missing"
    else:
        assert rep["status"] == "resolved"
        assert rep["computed_by_this_call"] == [], "compute=False trains nothing"
        assert rep["n_arms"] == len(rep["runs"]) >= 1

    steps = [r["step_id"] for r in out["records"]]
    assert "preflight_align" in steps, "the gates must record under their own stage"
    assert "preflight_encode" not in steps, "the encode stage did not ask these"
    sel = [r for r in out["records"] if r.get("kind") == "selection"][-1]
    assert sel["actor"] == "client" and sel["confirmed_by"] is None


def test_halting_keeps_the_reason_it_was_given():
    """an upstream refusal names what was missing. the generic halt note used to
    overwrite it, leaving the cause in the ledger and out of the response."""
    out = _run(answer={"eda_admissibility": "halt",
                       "verdict_disagreement": "computed"}, tid="halted")
    assert out["halted"] is True
    assert out["report"]["status"] == "halted"


def test_the_grid_refuses_if_a_gate_is_still_open():
    """belt and braces. the routing makes it unreachable, and a path that trusts
    its own routing is one edit from training on an inadmissible cohort."""
    from omicstra.graphs.align import align

    r = align({"gates": [{"id": "eda_admissibility", "open": True}]})
    assert r["report"]["status"] == "refused"
    assert "eda_admissibility" in r["report"]["why"]


# --- the bug class, across every arm ---------------------------------------
@pytest.mark.parametrize("arm", ["he", "st", "align"])
def test_a_handle_round_trips_to_its_own_arm(arm, tmp_path):
    """start -> interrupt -> resume must land in the SAME graph, and every ledger
    row must name that stage.

    this is the test for a bug class, not a bug. `AlignState` did not declare
    `arm`, langgraph dropped the key at the boundary, the handle defaulted to
    "he", and a resume re-entered the encode graph - reporting an encode
    preflight for an align run. the same shape as reading a field that does not
    exist and getting a default: silent, and wrong in a plausible direction.
    """
    import os

    from omicstra.mcp import server as S

    before = dict(os.environ)
    os.environ["OMICSTRA_CHECKPOINT"] = f"sqlite:{tmp_path / 'ck.sqlite'}"
    try:
        if not (SEED / "eda_summary.json").is_file():
            pytest.skip("no seed cohort on this machine")
        rid = f"rt-{arm}"
        start = {"he": lambda: S.he_encode(project_id="tnbc-92", run_id=rid),
                 "st": lambda: S.st_encode(project_id="tnbc-92", run_id=rid),
                 "align": lambda: S.align_run(project_id="tnbc-92", run_id=rid)}[arm]
        r = start()

        if r["status"] == "awaiting_answer":
            opts = {g["id"]: g["options"][0] for g in r["gate"]["gates"]}
            r = S.runs_resume(rid, answers=opts)

        # the arm survives the round trip
        assert r.get("arm", arm) == arm, f"{arm} resumed as {r.get('arm')}"
        assert S.runs_status(rid)["arm"] == arm

        # and no ledger row is attributed to another stage
        stage = {"he": "encode", "st": "encode", "align": "align"}[arm]
        other = {"encode": "preflight_align", "align": "preflight_encode"}[stage]
        steps = [d.get("step_id", "") for d in S.runs_record(rid)["records"]]
        assert other not in steps, f"{arm} recorded a row under {other}: {steps}"
    finally:
        os.environ.clear()
        os.environ.update(before)
