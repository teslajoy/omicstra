"""natural-language question -> validated analysis request -> execution.

interpretation happens OUTSIDE this module. three clients exist:

    Claude Desktop    the desktop model interprets the question and calls
                      the MCP `ask` tool with structured params
    Claude Code       same - the terminal model interprets, calls MCP
    omicstra ask      a CLI command with a configurable interpreter callable

all three reach the same `validate -> execute -> notebook` path. the MCP tool
receives already-interpreted structured parameters; it does not call an LLM.
`interpret_and_execute` is the CLI's path - it takes an interpreter callable
so a test can hand it a fixed dict and a live CLI can hand it whatever model
the user configured.

everything after interpretation is deterministic: validation against
declarations, parameter resolution from cohort.json, graph invocation,
notebook generation, provenance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from omicstra.contracts.project import ProjectConfig
from omicstra.settings import settings


@dataclass
class AnalysisRequest:
    """what the interpreter produced, after validation."""
    question: str
    objective: str
    task_id: str
    hypothesis: str
    project_id: str
    run: str
    paired_view: str | None = None
    signals: list[str] = field(default_factory=list)
    include_pathway_scores: bool = False
    resolved_params: dict[str, Any] = field(default_factory=dict)
    param_sources: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    interpretation_model: str = ""
    interpretation_source: str = ""
    notes: list[str] = field(default_factory=list)


TASK_FAMILY_TO_HYPOTHESIS = {
    "cross_modal_retrieval": "H1",
    "structural_agreement": "H1",
    "tissue_state_grouping": "H2-A",
    "subject_identity_suppression": "H2-C",
    "pathway_transfer": "H3",
    "pathway_discrimination": "H3-DAG",
    "morphology_decode": "H2-B.2",
}

# which evaluate_params steps each task family requires
TASK_FAMILY_STEPS = {
    "subject_identity_suppression": (
        "label_granularity", "paired_identifiability",
        "structure_and_its_null", "confounder_suppression",
        "held_out_comparison", "hypothesis_verdict"),
    "pathway_transfer": (
        "association_and_specificity", "hypothesis_verdict"),
    "tissue_state_grouping": (
        "label_granularity", "structure_and_its_null",
        "hypothesis_verdict"),
    "cross_modal_retrieval": (
        "paired_identifiability", "held_out_comparison",
        "hypothesis_verdict"),
    "structural_agreement": (
        "paired_identifiability", "hypothesis_verdict"),
    "morphology_decode": (
        "held_out_comparison", "hypothesis_verdict"),
    "pathway_discrimination": (
        "association_and_specificity", "hypothesis_verdict"),
}


def _load_routing_contract() -> dict:
    from omicstra.contracts.routing import load_routing_contract
    return load_routing_contract()


def _task_families(contract: dict) -> dict[str, dict]:
    return {f["id"]: f for f in contract.get("task_families", [])}


def validate(raw: dict, project_id: str | None = None) -> AnalysisRequest:
    """deterministic validation of an interpreted request against declarations.

    refuses with every problem named at once rather than one per call.
    """
    problems: list[str] = []

    question = raw.get("question", "")
    objective = raw.get("objective", "")
    task_id = raw.get("task_id", "")
    run = raw.get("run", "")
    pid = raw.get("project_id") or project_id

    if not pid:
        problems.append("no project_id")
    if not task_id:
        problems.append("no task_id")
    if not run:
        problems.append("no run")

    contract = _load_routing_contract()
    families = _task_families(contract)
    if task_id and task_id not in families:
        problems.append(
            f"task_id {task_id!r} is not a declared task family. "
            f"known: {sorted(families)}")

    hypothesis = ""
    if task_id in families:
        hypothesis = families[task_id].get("hypothesis", "")

    cfg = None
    if pid:
        try:
            cfg = ProjectConfig.load(pid)
        except Exception as e:
            problems.append(f"cannot load project {pid}: {e}")

    if run and cfg:
        all_runs = list(cfg.contrastive_runs or []) + list(
            cfg.classical_runs or [])
        if all_runs and run not in all_runs:
            problems.append(
                f"run {run!r} is not declared. known: {all_runs}")

    paired_view = raw.get("paired_view")
    signals = list(raw.get("signals") or [])
    include_pathway_scores = bool(raw.get("include_pathway_scores"))

    # check that required evaluate_params steps are declared
    param_sources: dict[str, str] = {}
    resolved: dict[str, Any] = {}
    missing: list[str] = []

    if cfg and pid and task_id:
        cohort_path = settings.project_root(pid) / "cohort.json"
        if cohort_path.is_file():
            cohort = json.loads(cohort_path.read_text())
            ep = cohort.get("evaluate_params", {})

            needed = TASK_FAMILY_STEPS.get(task_id, ())
            for step_key in needed:
                if step_key in ep:
                    resolved[step_key] = ep[step_key]
                    param_sources[step_key] = "cohort.json#evaluate_params"
                elif step_key != "hypothesis_verdict":
                    missing.append(
                        f"evaluate_params.{step_key} is required for "
                        f"{task_id} but not declared in cohort.json")
            # hypothesis_verdict needs the hypothesis mapping
            hv = ep.get("hypothesis_verdict", {})
            if hv:
                resolved["hypothesis_verdict"] = hv
                param_sources["hypothesis_verdict"] = "cohort.json#evaluate_params"
        else:
            problems.append(f"no cohort.json at {cohort_path}")

    if problems:
        raise ValidationError(problems)

    return AnalysisRequest(
        question=question,
        objective=objective,
        task_id=task_id,
        hypothesis=hypothesis,
        project_id=pid or "",
        run=run,
        paired_view=paired_view,
        signals=signals,
        include_pathway_scores=include_pathway_scores,
        resolved_params=resolved,
        param_sources=param_sources,
        missing=missing,
        interpretation_model=raw.get("interpretation_model", ""),
        interpretation_source=raw.get("interpretation_source", ""),
        notes=list(raw.get("notes") or []),
    )


class ValidationError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__(f"{len(problems)} validation problem(s): " +
                         "; ".join(problems))


class ExecutionResult:
    """what execution produced, with provenance."""
    def __init__(self, request: AnalysisRequest, report: dict,
                 records: list[dict], run_id: str,
                 notebook_path: str | None = None,
                 trace_path: str | None = None,
                 notebook_error: str | None = None):
        self.request = request
        self.report = report
        self.records = records
        self.run_id = run_id
        self.notebook_path = notebook_path
        self.trace_path = trace_path
        self.notebook_error = notebook_error

    def provenance(self) -> dict:
        return {
            "question": self.request.question,
            "objective": self.request.objective,
            "interpretation_model": self.request.interpretation_model,
            "interpretation_source": self.request.interpretation_source,
            "task_id": self.request.task_id,
            "hypothesis": self.request.hypothesis,
            "project_id": self.request.project_id,
            "run": self.request.run,
            "paired_view": self.request.paired_view,
            "signals": self.request.signals,
            "include_pathway_scores": self.request.include_pathway_scores,
            "resolved_params": self.request.resolved_params,
            "param_sources": self.request.param_sources,
            "missing": self.request.missing,
            "notes": self.request.notes,
            "run_id": self.run_id,
            "notebook": self.notebook_path,
            "trace": self.trace_path,
            "notebook_error": self.notebook_error,
        }

    def findings(self) -> dict[str, Any]:
        """numerical findings from the computed records."""
        out: dict[str, Any] = {}
        for rec in self.records:
            sid = rec.get("step_id")
            if not sid:
                continue
            obs = rec.get("observed")
            if not obs:
                continue
            scalars = {k: v for k, v in obs.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            entry: dict[str, Any] = {
                "status": rec.get("status"),
                "result": rec.get("result", ""),
            }
            if scalars:
                entry["values"] = scalars
            if "per_view" in obs:
                entry["per_view"] = obs["per_view"]
            if "guards" in obs:
                entry["guards"] = obs["guards"]
            if scalars or "per_view" in obs:
                out[sid] = entry
        return out

    def summary(self) -> dict:
        """plain-language answer grounded in computed results."""
        verdicts = (self.report.get("verdicts") or {})
        steps = self.report.get("steps") or {}
        hyp = self.request.hypothesis

        return {
            "question": self.request.question,
            "hypothesis": hyp,
            "verdicts": verdicts,
            "steps": steps,
            "findings": self.findings(),
            "run": self.request.run,
            "n_units": self.report.get("n_units"),
            "n_samples": self.report.get("n_samples"),
            "limitations": self.report.get("label_note") or None,
            "notebook": self.notebook_path,
            "trace": self.trace_path,
            "notebook_error": self.notebook_error,
            "provenance": self.provenance(),
        }


def execute(request: AnalysisRequest) -> ExecutionResult:
    """invoke the existing evaluation graph on a validated request."""
    import uuid

    from omicstra.graph import _load_cohort
    from omicstra.graphs.evaluate import build_evaluate_graph

    cohort = _load_cohort(request.project_id)
    run_id = f"question-{uuid.uuid4().hex[:12]}"

    graph = build_evaluate_graph()
    out = graph.invoke({
        "project_id": request.project_id,
        "arm": "evaluate",
        "cohort": cohort,
        "compute": True,
        "run": request.run,
        "paired_view": request.paired_view,
        "signals": request.signals,
        "include_pathway_scores": request.include_pathway_scores,
    })

    report = out.get("report", {})
    records = out.get("records", [])

    nb_path, tr_path, nb_err = None, None, None
    if report.get("status") == "measured" and records:
        try:
            nb_path, tr_path = _write_notebook(
                request, records, cohort, run_id)
        except Exception as e:
            nb_err = f"{type(e).__name__}: {e}"

    return ExecutionResult(
        request=request,
        report=report,
        records=records,
        run_id=run_id,
        notebook_path=str(nb_path) if nb_path else None,
        trace_path=str(tr_path) if tr_path else None,
        notebook_error=nb_err,
    )


def _write_notebook(request: AnalysisRequest, records: list[dict],
                    cohort: dict, run_id: str):
    """generate the evaluate notebook from execution records."""
    from omicstra.notebook import write

    bundle = {
        "unit": "evaluate",
        "project_id": request.project_id,
        "project_dir": str(settings.project_root(request.project_id)),
        "run": request.run,
        "run_id": run_id,
        "params": cohort.get("evaluate_params", {}),
        "records": [r for r in records if r.get("step_id")],
    }
    return write(bundle, request.project_id)


def interpret_and_execute(
    question: str,
    interpreter: Callable[[str, dict], dict],
    project_id: str | None = None,
) -> ExecutionResult:
    """the full path: question -> interpret -> validate -> execute -> results.

    `interpreter` is the model boundary. it receives (question, context) and
    returns a dict with task_id, run, signals, etc. a test supplies a function
    that returns a fixed dict; a live server supplies one that calls Claude.
    """
    from omicstra.routing import list_task_families

    context = {
        "project_id": project_id,
        "task_families": list_task_families(project_id=project_id),
    }

    raw = interpreter(question, context)
    raw["question"] = question
    if project_id and not raw.get("project_id"):
        raw["project_id"] = project_id

    request = validate(raw, project_id=project_id)
    return execute(request)