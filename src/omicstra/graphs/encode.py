"""LEVEL 1 - the encode gate. the only thing here that a protocol cannot do.

`protocols/encode.py` computes what each preflight gate would ask and resolves
the ones a cohort has declared. it never pauses, because a protocol that can
block is a graph wearing the wrong name. this file is the pause.

    protocols/encode.preflight()   what the gates ARE on this cohort
    graphs/encode.py               asks the person, records the answer, resumes

the rule from `decisions.md`: a thing gets its own graph only if it can ask a
human. encode qualifies for exactly one reason - a machine is about to spend
hours of accelerated compute on a values call, and five of those calls have no
correct default.

the fixture runs unattended
---------------------------
tnbc-92 declares all five answers, so `ask` is never entered and the graph runs
start to finish with nobody in the loop. that is not the gates being bypassed -
every one is still computed, still resolved, still recorded with its provenance.
it is what makes the cohort usable as the end-to-end fixture for step 8.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from omicstra.protocols.encode import (
    ComputeRefused,
    GateRequest,
    assert_clear,
    capacity_plan,
    contract,
    gate_record,
    preflight,
)


# --- state -----------------------------------------------------------------
# MUST be module level, for the same reason EDAState is: under
# `from __future__ import annotations` a function-local TypedDict raises
# NameError when langgraph resolves node annotations against module globals,
# and only when a conditional edge is present, so it surfaces late.
class EncodeState(TypedDict, total=False):
    project_id: str
    cohort: dict[str, Any]           # the cohort declaration, already read
    encoder: str
    unit_counts: dict[str, int]      # from the inventory record, never re-measured
    min_scope: int | None
    device: str | None               # declared; never probed here - see preflight()
    plan: dict                       # what the run will cost, or why that is not sayable
    gates: list[dict]                # GateRequest, as plain dicts for the trace
    answers: dict[str, str]          # gate id -> the option a person picked
    records: list[dict]
    approved: bool | None
    halted: bool
    shards: list[dict]
    report: dict


def _as_dicts(reqs: list[GateRequest]) -> list[dict]:
    """GateRequest -> plain dicts. the tracer serialises node state, so what
    travels has to be JSON rather than a dataclass whose repr looks like data."""
    return [{"id": r.id, "question": r.question, "options": list(r.options),
             "forecloses": r.forecloses, "pre_answerable_by": r.pre_answerable_by,
             "answer": r.answer, "source": r.source, "observed": r.observed,
             "open": r.open} for r in reqs]


# --- nodes -----------------------------------------------------------------
def gates(state: EncodeState) -> dict:
    """compute every applicable preflight gate and resolve what is declared.

    no judgement here - this is the protocol's answer, lifted into state so the
    branch below and the interrupt payload read the same thing.
    """
    cohort, encoder = state.get("cohort", {}), state.get("encoder", "virchow2")
    reqs = preflight(cohort, encoder, unit_counts=state.get("unit_counts"),
                     min_scope=state.get("min_scope"), device=state.get("device"))
    rec = gate_record(reqs, encoder)
    # the price, computed whether or not anyone is asked. a run that clears every
    # gate unattended should still leave behind what it expected to cost, because
    # that is the number the next plan is checked against.
    plan = capacity_plan(cohort, encoder, unit_counts=state.get("unit_counts"),
                         device=state.get("device"))
    return {"gates": _as_dicts(reqs), "plan": plan,
            "records": [*state.get("records", []), rec.model_dump(mode="json")]}


def ask(state: EncodeState) -> dict:
    """HITL. pauses hard, checkpoints, resumes at this exact point.

    the payload carries the OPEN gates with their evidence and, for each, what
    accepting forecloses. that last field is load-bearing: a person being asked
    whether to drop 19 subarrays needs to know it changes the cohort, not just
    that a threshold was crossed. the same lesson the eda gate learned - a
    decision without its consequence in front of it is a rubber stamp.
    """
    open_gates = [g for g in state.get("gates", []) if g["open"]]
    answer = interrupt({
        "question": (f"{len(open_gates)} preflight gate(s) have no declared answer. "
                     "Compute does not start until each is decided."),
        "gates": open_gates,
        "how_to_answer": ("reply with a mapping of gate id -> one of that gate's "
                          "options, e.g. {\"platform_floor\": \"drop_below_floor\"}"),
        "consequence": "each answer is recorded with actor=human and travels with the run",
        "pre_answerable": {g["id"]: g["pre_answerable_by"] for g in open_gates
                           if g["pre_answerable_by"]},
        # what it costs, in front of the person deciding whether to spend it.
        # the gates say what is being decided; this says what the decision buys,
        # and "run it here or somewhere else" cannot be answered without it.
        "plan": state.get("plan", {}),
        "no_default": "the system does not pick",
    })
    return _accept(state, answer)


def _accept(state: EncodeState, answer: Any) -> dict:
    """validate a human's answer against the SAME closed option set a declaration
    faces.

    this is the symmetry that makes the contract's rule true. the option set was
    closed on the interrupt path and open on the declaration path until
    `_resolve_option` landed; closing only one of them is not closing it. a
    person picking a value that is not an option is refused here for the reason a
    cohort file is refused there - the ledger must not record a choice nobody was
    offered.
    """
    from omicstra.protocols.encode import _resolve_option

    if answer in (False, None) or (isinstance(answer, str)
                                   and answer.strip().lower() in ("no", "halt", "stop")):
        return {"approved": False, "halted": True}

    by_id = {g["id"]: g for g in contract()["gates"]}
    open_ids = [g["id"] for g in state.get("gates", []) if g["open"]]
    raw = answer if isinstance(answer, dict) else {}

    missing = [g for g in open_ids if g not in raw]
    if missing:
        return {"approved": False, "halted": True,
                "records": [*state.get("records", []),
                            {"step_id": "preflight_encode", "kind": "gate", "status": "not_run",
                             "actor": "human", "verdict": "halt",
                             "reason": f"unanswered: {missing}"}]}

    resolved: dict[str, str] = {}
    for gid, val in raw.items():
        g = by_id.get(gid)
        if g is None:
            return {"approved": False, "halted": True,
                    "records": [*state.get("records", []),
                                {"step_id": "preflight_encode", "kind": "gate",
                                 "status": "error", "actor": "human",
                                 "reason": f"{gid} is not a gate in the contract"}]}
        try:
            resolved[gid] = _resolve_option(val, tuple(g["options"]), gid,
                                            g.get("answer_kind", "choice"))
        except ValueError as e:
            return {"approved": False, "halted": True,
                    "records": [*state.get("records", []),
                                {"step_id": "preflight_encode", "kind": "gate",
                                 "status": "error", "actor": "human", "reason": str(e)}]}

    if any(v == "halt" for v in resolved.values()):
        return {"approved": False, "halted": True, "answers": resolved}

    gates_now = [dict(g, answer=resolved.get(g["id"], g["answer"]),
                      source="human" if g["id"] in resolved else g["source"],
                      open=False if g["id"] in resolved else g["open"])
                 for g in state.get("gates", [])]
    return {"approved": True, "halted": False, "answers": resolved, "gates": gates_now,
            "records": [*state.get("records", []),
                        {"step_id": "preflight_encode", "kind": "selection", "status": "pass",
                         "actor": "human", "chosen": resolved,
                         "note": "answered at the encode gate; recorded, not defaulted"}]}


def encode(state: EncodeState) -> dict:
    """the compute. refuses if anything upstream left a gate open.

    the refusal is belt and braces on purpose: the routing below should make it
    unreachable, and a compute path that trusts its own routing is one edit away
    from spending GPU hours on an unanswered values call.
    """
    reqs = preflight(state.get("cohort", {}), state.get("encoder", "virchow2"),
                     unit_counts=state.get("unit_counts"),
                     min_scope=state.get("min_scope"))
    answered = state.get("answers", {})
    still_open = [r for r in reqs if r.open and r.id not in answered]
    if still_open:
        raise ComputeRefused(f"reached compute with {[r.id for r in still_open]} open")
    if not answered:
        assert_clear(reqs)

    return {"report": {"status": "ready",
                       "plan": state.get("plan", {}),
                       "note": ("gates cleared. the shard run is dispatch's job - "
                                "protocols.encode.encode_he_cohort(), which this node "
                                "calls once a cohort supplies its sample list."),
                       "encoder": state.get("encoder", "virchow2"),
                       "answers": answered}}


def halt(state: EncodeState) -> dict:
    return {"halted": True, "report": {"status": "halted",
                                       "note": "a preflight gate was refused; nothing ran"}}


def _route_gates(state: EncodeState) -> str:
    return "ask" if any(g["open"] for g in state.get("gates", [])) else "encode"


def _route_answer(state: EncodeState) -> str:
    return "encode" if state.get("approved") else "halt"


def build_encode_graph(checkpointer=None):
    """gates -> (ask) -> encode, with halt on refusal."""
    g = StateGraph(EncodeState)
    g.add_node("gates", gates)
    g.add_node("ask", ask)
    g.add_node("encode", encode)
    g.add_node("halt", halt)

    g.add_edge(START, "gates")
    g.add_conditional_edges("gates", _route_gates, {"ask": "ask", "encode": "encode"})
    g.add_conditional_edges("ask", _route_answer, {"encode": "encode", "halt": "halt"})
    g.add_edge("encode", END)
    g.add_edge("halt", END)
    return g.compile(checkpointer=checkpointer)
