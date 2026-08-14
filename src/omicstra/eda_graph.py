"""the EDA subgraph - profile, gate, escalate.

generalizable by construction: the graph knows about no cohort and no step. it
reads WHICH steps to run from the contract, resolves them through a registry, and
takes their params from the cohort's own config. adding a step means registering
a function and naming it in the contract; the graph is untouched.

    contract  ──► which steps, which criteria      package, cohort-free
    registry  ──► name -> callable                  package
    config    ──► this cohort's params              project
    adapter   ──► this cohort's files -> AnnData    project

the one branch: the gate verdict. `stop` halts, `caution` interrupts for a human,
`proceed` continues. that is the only edge in the EDA that changes what happens
next, which is why the rest can be a straight line.
"""
from __future__ import annotations

from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from omicstra import eda_steps
from omicstra.eda import cohort_escalations, load_calibration, load_contract
from omicstra.records import DiagnosticRecord, GateRecord


# --- state -----------------------------------------------------------------
# MUST be module level. under `from __future__ import annotations` a
# function-local TypedDict raises NameError when langgraph resolves node
# annotations against module globals - and only when a conditional edge is
# present, so it surfaces late. verified on 3.11 / 3.13 / 3.14.
class EDAState(TypedDict, total=False):
    project_id: str
    params: dict[str, Any]          # cohort-supplied, per step
    adata_path: str                 # a REFERENCE, never the object
    records: list[dict]
    verdict: str
    violations: list[str]
    cautions: list[str]
    escalations: list[dict]
    approved: bool | None
    halted: bool


# --- step registry ---------------------------------------------------------
# name -> callable(adata, **params) -> DiagnosticRecord
STEP_REGISTRY: dict[str, Callable[..., DiagnosticRecord]] = {
    "count_statistics": eda_steps.count_statistics,
    "marker_expression": eda_steps.marker_expression,
    "spatial_autocorrelation": eda_steps.spatial_autocorrelation,
    "batch_structure": eda_steps.batch_structure,
}


def register_step(name: str, fn: Callable[..., DiagnosticRecord]) -> None:
    """a new modality registers its steps here; the graph does not change."""
    STEP_REGISTRY[name] = fn


# --- nodes -----------------------------------------------------------------
def profile(state: EDAState) -> dict:
    """run every declared step that the registry can resolve and this cohort
    supplies params for. a step with no params is skipped, not guessed at.

    loads from `adata_path` rather than receiving an object: checkpointed state
    must be serialisable, so state carries references and nodes load. this is the
    same discipline stages.py declares - artifact refs, not arrays - and it is
    what makes the graph resumable.
    """
    import anndata as ad
    adata = ad.read_h5ad(state["adata_path"])
    params = state.get("params", {})
    records: list[dict] = list(state.get("records", []))

    for name, kwargs in params.items():
        fn = STEP_REGISTRY.get(name)
        if fn is None:
            records.append(DiagnosticRecord(
                step_id=name, status="not_run",
                result=f"no step registered under {name!r}",
                decision="register the step or remove it from the contract",
            ).model_dump())
            continue
        try:
            records.append(fn(adata, **kwargs).model_dump())
        except Exception as e:
            records.append(DiagnosticRecord(
                step_id=name, status="error",
                result=f"{type(e).__name__}: {e}",
                decision="step failed; the gate cannot judge this check",
            ).model_dump())
    return {"records": records}


def gate(state: EDAState) -> dict:
    """turn the records into a verdict. no new measurement happens here.

    a step whose criterion is `cohort_calibrated` in the contract, on a cohort
    with no calibration record for it, is NOT judged: the measurement stands,
    but the bar it would be judged against was derived somewhere else. that
    escalates instead, through the same interrupt a cautionary verdict takes.

    a step the contract declares no check for is judged on its own status, as
    before - an unregistered step is a missing step, not a missing criterion.
    """
    records = state.get("records", [])
    violations, cautions = [], []

    contract = load_contract()
    calibration = load_calibration(state.get("project_id"))
    escalations = cohort_escalations(contract, calibration, state.get("project_id", ""),
                                     only={r["step_id"] for r in records})
    escalated_steps = {e["check"] for e in escalations}

    for r in records:
        if r["step_id"] in escalated_steps:
            continue
        if r.get("status") == "fail":
            violations.append(f"{r['step_id']}: {r.get('result', 'failed')}")
        elif r.get("status") in ("not_run", "error"):
            cautions.append(f"{r['step_id']}: {r.get('result', 'not measured')}")
        for c in r.get("caveats", []):
            cautions.append(f"{r['step_id']}: {c}")

    for e in escalations:
        cautions.append(f"{e['check']}: escalated - {e['why_not_inherited']}")

    verdict = ("stop" if violations
               else "proceed_with_caution" if cautions
               else "proceed")

    rec = GateRecord(
        step_id="gate_eda", verdict=verdict,
        violations=violations, cautions=cautions,
        reason=("; ".join(violations) if violations
                else f"{len(cautions)} caution(s)" if cautions else "all checks clean"),
        escalated=(verdict == "proceed_with_caution"),
    )
    return {"verdict": verdict, "violations": violations, "cautions": cautions,
            "escalations": escalations, "records": records + [rec.model_dump()]}


def escalate(state: EDAState) -> dict:
    """HITL. pauses hard, checkpoints, resumes at this exact point.

    the system does not pick a default - that is what escalation means.
    """
    escalations = state.get("escalations", [])
    answer = interrupt({
        "question": "The gate returned proceed_with_caution. Accept the cautions and proceed?",
        "verdict": state.get("verdict"),
        "cautions": state.get("cautions", []),
        # a calibrated criterion this cohort never derived is the sharper ask:
        # it names what accepting forecloses, not just what it risks.
        "escalations": escalations,
        "consequence": "proceeding records these as accepted; they will not be raised again"
                       + ("; accepting an escalated check adopts a criterion calibrated on "
                          "another cohort" if escalations else ""),
        "no_default": "the system does not pick",
    })
    if isinstance(answer, bool):
        accepted = answer
    else:
        accepted = str(answer).strip().lower() in ("yes", "y", "accept", "true", "proceed")
    return {"approved": accepted, "halted": not accepted}


def _route(state: EDAState) -> str:
    v = state.get("verdict")
    return "stop" if v == "stop" else "escalate" if v == "proceed_with_caution" else "proceed"


def halt(state: EDAState) -> dict:
    return {"halted": True}


# --- graph -----------------------------------------------------------------
def build_eda_graph(checkpointer=None):
    """the EDA subgraph. cohort-free: everything specific arrives in state."""
    g = StateGraph(EDAState)
    g.add_node("profile", profile)
    g.add_node("gate", gate)
    g.add_node("escalate", escalate)
    g.add_node("halt", halt)

    g.add_edge(START, "profile")
    g.add_edge("profile", "gate")
    g.add_conditional_edges("gate", _route,
                            {"proceed": END, "escalate": "escalate", "stop": "halt"})
    g.add_edge("escalate", END)
    g.add_edge("halt", END)
    return g.compile(checkpointer=checkpointer)