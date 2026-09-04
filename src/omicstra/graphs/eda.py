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

from omicstra.eda import cohort_escalations, load_calibration, load_contract
from omicstra.protocols import build_protocol
from omicstra.protocols.eda import EDA_STEPS
from omicstra.protocols.inventory import INVENTORY_STEPS
from omicstra.records import DiagnosticRecord, GateRecord
from omicstra.settings import settings


# --- state -----------------------------------------------------------------
# MUST be module level. under `from __future__ import annotations` a
# function-local TypedDict raises NameError when langgraph resolves node
# annotations against module globals - and only when a conditional edge is
# present, so it surfaces late. verified on 3.11 / 3.13 / 3.14.
class EDAState(TypedDict, total=False):
    project_id: str
    params: dict[str, Any]          # cohort-supplied, per eda step
    inventory_params: dict[str, Any]
    inventory: dict[str, dict]
    adata_path: str                 # a REFERENCE, never the object
    records: list[dict]
    verdict: str
    violations: list[str]
    cautions: list[str]
    escalations: list[dict]
    approved: bool | None
    halted: bool


# --- nodes -----------------------------------------------------------------
# the step registry lives in protocols/, not here. a graph that also owns a
# registry is two things; this file is the gate and the interrupt, nothing else.
def inventory(state: EDAState) -> dict:
    """"what is this data" - 8 steps, ending in a conformance bind.

    runs FIRST because eda reads three of its answers: the join key, the
    platform per sample, and which matrix is raw. a required role left unbound
    halts here rather than letting eda measure something it cannot describe.
    """
    ctx = {"project_dir": str(settings.project_root(state.get("project_id"))),
           "project_id": state.get("project_id"),
           "params": state.get("inventory_params", {})}
    recs = build_protocol(INVENTORY_STEPS, "inventory").invoke(ctx)
    unbound = (recs.get("bind", {}).get("observed") or {}).get("unbound_required") or []
    failed = [k for k, v in recs.items() if v.get("status") == "fail"]
    # halt on ANY inventory failure, not only on unbound roles. bind reports
    # unbound only when it RAN - if platform failed, bind is not_run and its
    # empty unbound list would wave the run through, letting eda measure data
    # inventory could not describe.
    return {"records": list(state.get("records", [])) + list(recs.values()),
            "inventory": recs,
            "halted": bool(unbound or failed)}


def profile(state: EDAState) -> dict:
    """"is it usable" - the eda protocol, run as one chain.

    loads from `adata_path` rather than receiving an object: checkpointed state
    must be serialisable, so state carries references and nodes load. that is
    also why the protocol takes a path, not an AnnData.
    """
    ctx = {"adata_path": state["adata_path"],
           "project_id": state.get("project_id"),
           "params": state.get("params", {})}
    recs = build_protocol(EDA_STEPS, "eda").invoke(ctx)
    return {"records": list(state.get("records", [])) + list(recs.values())}


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
        # the records MUST travel in the payload. whatever is passed to
        # interrupt() is what the caller receives in `__interrupt__`, and while
        # a subgraph is paused the parent sees none of its state - so without
        # this a client is asked to accept cautions with no sight of the
        # diagnostics that produced them. that is the difference between an
        # informed decision and a rubber stamp, and it is the whole reason the
        # gate asks a person at all.
        "records": state.get("records", []),
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
    g.add_node("inventory", inventory)
    g.add_node("profile", profile)
    g.add_node("gate", gate)
    g.add_node("escalate", escalate)
    g.add_node("halt", halt)

    g.add_edge(START, "inventory")
    # inventory HALTS the run when a required role is unbound - eda cannot
    # measure what inventory could not describe.
    g.add_conditional_edges("inventory",
                            lambda s: "halt" if s.get("halted") else "profile",
                            {"halt": "halt", "profile": "profile"})
    g.add_edge("profile", "gate")
    g.add_conditional_edges("gate", _route,
                            {"proceed": END, "escalate": "escalate", "stop": "halt"})
    g.add_edge("escalate", END)
    g.add_edge("halt", END)
    return g.compile(checkpointer=checkpointer)