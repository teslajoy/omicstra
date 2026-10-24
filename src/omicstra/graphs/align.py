"""LEVEL 1: the align gate. the admissibility constraint, enforced rather than kept.

this repo opens with "do not propose alignment, contrastive learning, retrieval
evaluation, or agent routing until EDA.md is satisfied". that was a convention a
reader had to honour. `eda_admissibility` is the same sentence as a gate: the
cohort's recorded verdict arrives as an interrupt, and the grid does not resolve
until someone accepts its cautions or records a dissent.

the gate set differs from encode's, which is why this is its own graph rather
than a mode: encode asks about weights, geometry and capacity before embedding a
unit; this asks whether the cohort is admissible at all, whether the computed
verdict matches the declared one, and which arms are in scope - questions that
only exist once there is something to align.

a cohort with NO verdict is REFUSED, not asked. a gate is a question with a
closed option set and someone to ask; "the gate has never been run" is neither,
and asking a person to accept a verdict nobody computed is how a cohort gets a
pass it never earned.

shared with the other arms: `ask` and `halt` are generic over a gate list, and
`_accept` validates a human's answer against the same closed option set a
declaration faces.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from omicstra.graphs.encode import _accept, _as_dicts, halt
from omicstra.protocols.align import EdaGateMissing, preflight_align
from omicstra.protocols.encode import gate_record
from omicstra.records import timed_node


class AlignState(TypedDict, total=False):
    project_id: str
    arm: str                   # "align". a key this state does not DECLARE is
                               # dropped at the boundary, so omitting it made the
                               # handle resolve to the default arm and a resume
                               # re-entered the encode graph
    cohort: dict[str, Any]
    compute: bool              # False resolves the grid and reports it, as run_align does
    hypotheses: list[str]
    gates: list[dict]
    answers: dict[str, str]
    records: list[dict]
    approved: bool | None
    halted: bool
    report: dict


def gates(state: AlignState) -> dict:
    """compute the gates and resolve what the verdict and the declarations settle.

    no judgement here. a `proceed` verdict closes the admissibility gate because
    there is nothing to accept; agreement closes the disagreement gate because
    there is no disagreement; a declared grid closes the scope gate.
    """
    try:
        reqs = preflight_align(state.get("cohort"), state.get("project_id"))
    except EdaGateMissing as e:
        return {"halted": True,
                "report": {"status": "refused", "why": str(e),
                           "constraint": "no alignment until the EDA gate is satisfied"},
                "records": [*state.get("records", []),
                            {"step_id": "preflight_align", "kind": "gate",
                             "status": "not_run", "actor": "deterministic",
                             "reason": str(e)}]}
    # the second parameter is ENCODER. passing the cohort name put it in the
    # encoder field of every align record. this stage chooses no encoder, so it
    # says so rather than naming something that is not one.
    rec = gate_record(reqs, "n/a - the align stage selects no encoder",
                      step_id="preflight_align")
    return {"gates": _as_dicts(reqs),
            "records": [*state.get("records", []), rec.model_dump(mode="json")]}


def ask(state: AlignState) -> dict:
    """HITL. the open gates with their evidence and what accepting forecloses."""
    from langgraph.types import interrupt

    open_gates = [g for g in state.get("gates", []) if g["open"]]
    answer = interrupt({
        "question": (f"{len(open_gates)} gate(s) stand between this cohort and its "
                     "grid. the first is its own admissibility."),
        "gates": open_gates,
        "how_to_answer": ("reply with a mapping of gate id -> one of that gate's "
                          'options, e.g. {"eda_admissibility": "accept_cautions"}'),
        "consequence": ("every number the grid produces inherits the cohort's "
                        "admissibility. a dissent travels with the results."),
        "no_default": "the system does not pick",
    })
    out = _accept(state, answer, step_id="preflight_align")
    # `declared_with_reason` overrides a computed verdict with a declared one.
    # the contract requires the reason, so an answer without one is refused here
    # rather than recorded as a bare choice.
    if (out.get("answers") or {}).get("verdict_disagreement") == "declared_with_reason":
        why = (answer or {}).get("reason") if isinstance(answer, dict) else None
        if not (isinstance(why, str) and why.strip()):
            return {"approved": False, "halted": True,
                    "report": {"status": "refused",
                               "why": ("verdict_disagreement=declared_with_reason "
                                       "needs a non-empty `reason`. overriding a "
                                       "computed verdict with a declared one is the "
                                       "one answer here that is not self-justifying.")},
                    "records": [*state.get("records", []),
                                {"step_id": "preflight_align", "kind": "gate",
                                 "status": "error", "actor": "client",
                                 "reason": "declared_with_reason given with no reason"}]}
        out["records"] = [*out.get("records", []),
                          {"step_id": "preflight_align", "kind": "selection",
                           "status": "pass", "actor": "client", "confirmed_by": None,
                           "chosen": {"verdict_disagreement": "declared_with_reason"},
                           "reason": why.strip()}]
    return out


def align(state: AlignState) -> dict:
    """resolve the grid. refuses if anything upstream left a gate open.

    `compute=False` is the default and reports what WOULD run, which is the same
    convention `run_align` already follows: a node that computes by default turns
    every caller - a test, a plan, a dry run - into a training job.
    """
    from omicstra.contracts.project import ProjectConfig
    from omicstra.protocols.align import ComputeUnavailable, run_align, run_niche_join

    still_open = [g for g in state.get("gates", []) if g["open"]]
    if still_open:
        return {"report": {"status": "refused",
                           "why": f"reached the grid with {[g['id'] for g in still_open]} open"}}

    answered = state.get("answers", {})
    cfg = ProjectConfig.load(state.get("project_id"))
    try:
        nj, _ = run_niche_join(cfg, state.get("project_id"))
    except (ComputeUnavailable, FileNotFoundError, KeyError) as e:
        return {"report": {"status": "unavailable", "stage": "niche_join", "why": str(e),
                           "answers": answered}}
    try:
        grid, rec = run_align(cfg, nj, project_id=state.get("project_id"),
                              compute=bool(state.get("compute")))
    except (ComputeUnavailable, FileNotFoundError, KeyError) as e:
        return {"report": {"status": "unavailable", "stage": "align", "why": str(e),
                           "answers": answered}}

    return {"records": [*state.get("records", []), rec.model_dump(mode="json")],
            "report": {"status": "resolved" if not state.get("compute") else "computed",
                       # AlignGrid keys its arms by run_id -> artifact path. a
                       # `.runs` attribute does not exist, and reading one gave a
                       # report that said zero arms for a ten-arm grid.
                       "n_arms": len(grid.run_refs),
                       "runs": sorted(grid.run_refs),
                       "runs_root": grid.runs_root,
                       "computed_by_this_call": list(grid.computed),
                       # NicheJoin is a BaseModel, so the hasattr guard that was
                       # here could never be false. the same idiom one line above
                       # read a field that did not exist and reported zero arms
                       # for a ten-arm grid.
                       "niche_join": nj.model_dump(),
                       "answers": answered,
                       "note": ("the grid is resolved from the cohort's recorded "
                                "artifacts. pass compute=True to train.")}}


def _route_gates(state: AlignState) -> str:
    if state.get("halted"):
        return "halt"
    return "ask" if any(g["open"] for g in state.get("gates", [])) else "align"


def _route_answer(state: AlignState) -> str:
    return "align" if state.get("approved") else "halt"


def build_align_graph(checkpointer=None):
    """gates -> (ask) -> align, with halt on refusal or a missing verdict."""
    g = StateGraph(AlignState)
    g.add_node("gates", timed_node(gates))
    g.add_node("ask", timed_node(ask))
    g.add_node("align", timed_node(align))
    g.add_node("halt", timed_node(halt))

    g.add_edge(START, "gates")
    g.add_conditional_edges("gates", _route_gates,
                            {"ask": "ask", "align": "align", "halt": "halt"})
    g.add_conditional_edges("ask", _route_answer, {"align": "align", "halt": "halt"})
    g.add_edge("align", END)
    g.add_edge("halt", END)
    return g.compile(checkpointer=checkpointer)


__all__ = ["align", "build_align_graph"]
