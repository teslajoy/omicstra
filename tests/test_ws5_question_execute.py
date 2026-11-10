"""WS 5: question.execute() routes through discover() before invoking the graph.

three paths:
  Route.COMPUTE -> returns ExecutionResult with report["status"] = "compute_required"
                   and report["plan"] listing the pipeline tools in order
  Route.REFUSE  -> returns ExecutionResult with report["status"] = "refused"
  Route.ASK     -> calls build_evaluate_graph() as before (evidence exists)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omicstra.question import AnalysisRequest, ExecutionResult, execute


def _req(task_id: str = "cross_modal_retrieval",
         project_id: str = "test-proj",
         run: str = "R1_v3") -> AnalysisRequest:
    return AnalysisRequest(
        question="test question",
        objective="test objective",
        task_id=task_id,
        hypothesis="H1",
        project_id=project_id,
        run=run,
    )


def _disc(route: str, reason: str = "mock", cohort: dict | None = None) -> dict:
    return {
        "route": route,
        "route_reason": reason,
        "cohort": cohort or {},
        "has_evidence": route == "ask",
        "has_evidence_for_task": route == "ask",
    }


class TestExecuteRoutesViaDiscover:
    def test_compute_route_returns_compute_required(self):
        """when discover() returns compute, report status is compute_required."""
        with patch("omicstra.graph.discover", return_value=_disc("compute")):
            result = execute(_req())

        assert isinstance(result, ExecutionResult)
        assert result.report["status"] == "compute_required"
        assert result.records == []

    def test_compute_plan_contains_pipeline_tools_in_order(self):
        """plan lists he_encode, st_encode, align_run, evaluate_run in that order."""
        with patch("omicstra.graph.discover", return_value=_disc("compute")):
            result = execute(_req())

        tools = [s["tool"] for s in result.report["plan"]]
        assert "he_encode" in tools
        assert "st_encode" in tools
        assert "align_run" in tools
        assert "evaluate_run" in tools
        assert tools.index("he_encode") < tools.index("align_run")
        assert tools.index("st_encode") < tools.index("align_run")
        assert tools.index("align_run") < tools.index("evaluate_run")

    def test_compute_plan_params_carry_project_id(self):
        """every plan step carries the request's project_id."""
        with patch("omicstra.graph.discover", return_value=_disc("compute")):
            result = execute(_req(project_id="my-cohort"))

        for step in result.report["plan"]:
            assert step["params"].get("project_id") == "my-cohort"

    def test_refuse_route_returns_refused_status(self):
        """when discover() returns refuse, report status is refused (no plan)."""
        with patch("omicstra.graph.discover",
                   return_value=_disc("refuse", "task not in contract")):
            result = execute(_req(task_id="not_a_real_task"))

        assert result.report["status"] == "refused"
        assert result.records == []
        assert "plan" not in result.report

    def test_ask_route_invokes_evaluate_graph(self):
        """when discover() returns ask, build_evaluate_graph is called."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "report": {"status": "measured"},
            "records": [],
        }
        with patch("omicstra.graph.discover",
                   return_value=_disc("ask", cohort={"cohort_id": "tnbc-92"})):
            with patch("omicstra.graphs.evaluate.build_evaluate_graph",
                       return_value=mock_graph):
                result = execute(_req())

        mock_graph.invoke.assert_called_once()
        assert result.report["status"] == "measured"