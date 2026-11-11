"""WS 7+8 integration tests.

WS 7: proves all five COMPUTE_CHAIN stages are individually wired and that
      execute() chains discover() -> build_evaluate_graph() -> ExecutionResult.

WS 8: proves the pack arm interrupt+resume path is functional and that the
      delivery bundle (provenance, summary, findings) has its required keys.

none of these tests load actual embedding arrays; the evaluate graph returns
"unavailable" when the files are absent, which is still a valid ExecutionResult.
"""
from __future__ import annotations

import pytest

from omicstra.question import AnalysisRequest, ExecutionResult, execute


# ---------------------------------------------------------------------------
# shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def cross_modal_request() -> AnalysisRequest:
    """a valid request for which tnbc-92 has routing evidence."""
    return AnalysisRequest(
        question="Does R4_v3 retrieve cross-modal matches above spatial-NN?",
        objective="test",
        task_id="cross_modal_retrieval",
        hypothesis="H1",
        project_id="tnbc-92",
        run="R4_v3",
    )


# ---------------------------------------------------------------------------
# WS 7 — full chain: discover() -> evaluate graph -> ExecutionResult
# ---------------------------------------------------------------------------

def test_execute_returns_execution_result(cross_modal_request):
    """execute() on tnbc-92 (evidence present) returns an ExecutionResult.

    tnbc-92 has routing evidence for cross_modal_retrieval, so discover()
    returns Route.ASK and build_evaluate_graph(compute=True) is invoked.
    without data files the measure node returns "unavailable", which is still
    a valid terminal report.
    """
    result = execute(cross_modal_request)

    assert isinstance(result, ExecutionResult)
    assert "status" in result.report
    # any terminal status is acceptable; "refused" only if the contract itself
    # is broken, which a separate contract test would catch first
    assert result.report["status"] in {
        "measured", "unavailable", "refused", "compute_required", "not_run"
    }
    assert result.run_id.startswith("question-")


def test_execute_route_is_ask_not_compute(cross_modal_request):
    """tnbc-92 has evidence for cross_modal_retrieval, so the route is ASK.

    a Route.COMPUTE result carries "plan" in the report; Route.ASK does not.
    this asserts that discover() fired the ask arm, not the compute arm.
    """
    result = execute(cross_modal_request)

    # the compute arm sets status="compute_required" and adds "plan"
    assert result.report.get("status") != "compute_required", (
        "discover() returned Route.COMPUTE but tnbc-92 has routing evidence "
        "for cross_modal_retrieval — check load_routing_evidence()"
    )


def test_execute_no_plan_on_ask_route(cross_modal_request):
    """Route.ASK never includes a 4-step compute plan in the report."""
    result = execute(cross_modal_request)
    assert "plan" not in result.report


# ---------------------------------------------------------------------------
# WS 8 — interrupt + resume through the pack arm
# ---------------------------------------------------------------------------

def test_evaluate_graph_interrupt_and_resume():
    """pack arm (compute=False) interrupts on open gates and resumes cleanly.

    tnbc-92 has three open gates (hypothesis_scope, artifact_pinning,
    guard_failure). the test drives the graph through the ask node with
    InMemorySaver, supplies a valid answer, and asserts the final report has a
    terminal status.

    invoke() returns {"__interrupt__": [...]} rather than raising; that is the
    LangGraph HITL contract — same pattern as test_encode_graph.py.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from omicstra.graph import _load_cohort
    from omicstra.graphs.evaluate import build_evaluate_graph

    app = build_evaluate_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "test-interrupt-resume"}}

    state = {
        "project_id": "tnbc-92",
        "cohort": _load_cohort("tnbc-92"),
        "compute": False,
    }

    # first invoke: should pause at the ask node
    out = app.invoke(state, cfg)
    assert "__interrupt__" in out, (
        "gates() found 3 open gates but ask() did not pause — "
        "check that _route_gates returns 'ask' when any gate is open"
    )

    # the interrupt payload carries the gates and instructions
    payload = out["__interrupt__"][0].value
    assert "gates" in payload

    # resume: supply valid answers for all three open gates
    answer = {
        "hypothesis_scope": "declared_scope",
        "artifact_pinning": "evaluate_unpinned",
        "guard_failure": "report_as_diagnostic",
    }
    out = app.invoke(Command(resume=answer), cfg)

    assert "__interrupt__" not in out, "graph paused again after a full answer"
    assert "report" in out
    assert "status" in out["report"]
    assert out["report"]["status"] in {"evaluated", "halted", "refused", "not_run"}


def test_evaluate_graph_halt_on_no_answer():
    """answering False (halt) via the resume command halts cleanly."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from omicstra.graph import _load_cohort
    from omicstra.graphs.evaluate import build_evaluate_graph

    app = build_evaluate_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "test-halt-on-no-answer"}}

    state = {
        "project_id": "tnbc-92",
        "cohort": _load_cohort("tnbc-92"),
        "compute": False,
    }

    out = app.invoke(state, cfg)
    assert "__interrupt__" in out

    # passing False halts the graph (_accept returns approved=False, halted=True)
    out = app.invoke(Command(resume=False), cfg)
    assert "__interrupt__" not in out
    assert out.get("halted") is True


# ---------------------------------------------------------------------------
# WS 8 — delivery bundle: provenance, summary, findings
# ---------------------------------------------------------------------------

PROVENANCE_REQUIRED = {
    "question", "task_id", "project_id", "run", "run_id",
    "hypothesis", "paired_view", "signals", "include_pathway_scores",
    "resolved_params", "missing",
}

SUMMARY_REQUIRED = {
    "question", "hypothesis", "verdicts", "run", "provenance",
}


def test_provenance_has_required_keys(cross_modal_request):
    """provenance() carries every field needed to reproduce and cite the run."""
    result = execute(cross_modal_request)
    prov = result.provenance()

    missing = PROVENANCE_REQUIRED - set(prov)
    assert not missing, f"provenance() missing keys: {sorted(missing)}"


def test_summary_has_required_keys(cross_modal_request):
    """summary() is the delivery bundle — required keys present on any route."""
    result = execute(cross_modal_request)
    s = result.summary()

    missing = SUMMARY_REQUIRED - set(s)
    assert not missing, f"summary() missing keys: {sorted(missing)}"


def test_findings_is_dict(cross_modal_request):
    """findings() returns a dict (possibly empty) regardless of data presence."""
    result = execute(cross_modal_request)
    f = result.findings()

    assert isinstance(f, dict)
