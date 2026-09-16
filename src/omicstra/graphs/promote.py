"""LEVEL 1 - the promote gate. the proposal is computed; the prose is asked for.

`protocols/promote.py` reads the scored artifacts and proposes: candidates,
intervals, floor comparisons, guard verdicts, and the outcome the contract's
rules imply. it never pauses and it never writes a sentence. this file is the
pause, and it is where the only human fields in the pack come from.

    protocols/promote.propose()   what the evidence SAYS
    graphs/promote.py             asks the person for what it MEANS, records it

promote qualifies for its own graph under the same rule encode does: a person
has to answer, and there is no correct default. the difference is what is being
asked. encode asks a values call before spending compute; promote asks for a
judgement that the measurement cannot supply - why a tie is a tie, what a
refusal means for the next run, which caveat belongs on which row.

what a person may NOT do here
-----------------------------
change a value, an interval, a margin or an outcome. those are computed, and
`assert_not_authored` runs before the pack is emitted. the gate takes prose and
an approval, nothing else: a pack where a person edited a number is exactly the
artifact the whole chain exists to make impossible.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from omicstra.protocols.promote import assert_not_authored, propose, provenance


class PromoteState(TypedDict, total=False):
    project_id: str
    sources: dict[str, Any]            # the cohort's evidence_sources declaration
    runs_root: str
    runs: list[str]
    proposal: dict                     # computed, never edited downstream
    notes: dict[str, dict[str, str]]   # task_id -> {note|caveat|interpretation}
    approved: bool | None
    pack: dict
    provenance: dict
    halted: bool


# --- nodes -----------------------------------------------------------------
def compute(state: PromoteState) -> dict:
    """the deterministic half, in one call. no person has spoken yet."""
    from pathlib import Path

    p = propose(state["sources"], Path(state["runs_root"]), state["runs"])
    return {"proposal": p}


def _summary(proposal: dict) -> list[dict]:
    """what the person is being shown - one line per row, no prose invented."""
    out = []
    for task_id, t in proposal["tasks"].items():
        if t["outcome"] == "not_available":
            out.append({"task": task_id, "outcome": "not_available", "why": t["why"]})
            continue
        lead = (t.get("winner") or ", ".join(t.get("tied", [])) or "-")
        out.append({"task": task_id, "outcome": t["outcome"], "leads": lead,
                    "margin": t.get("margin"), "n_candidates": len(t["candidates"]),
                    "refusal": t.get("refusal")})
    return out


def ask(state: PromoteState) -> dict:
    """the gate. the payload is the proposal; the answer is prose and a yes.

    a caller that supplies neither leaves the pack unwritten, which is the
    honest state: a pack whose notes were skipped is a pack nobody vouched for.
    """
    answer = interrupt({
        "kind": "promote",
        "project_id": state.get("project_id"),
        "proposal": _summary(state["proposal"]),
        "asks": ("a note or caveat per row where one is warranted, and an approval. "
                 "values, intervals, margins and outcomes are computed and are not "
                 "editable here."),
        "fields": ["note", "caveat", "interpretation"],
    })
    if isinstance(answer, dict):
        return {"notes": answer.get("notes", {}) or {},
                "approved": bool(answer.get("approved"))}
    return {"approved": bool(answer)}


def emit(state: PromoteState) -> dict:
    """merge the two halves, and refuse a pack where a person touched a number."""
    proposal = state["proposal"]
    notes = state.get("notes") or {}
    tasks = {}
    for task_id, t in proposal["tasks"].items():
        row = dict(t)
        for field, text in (notes.get(task_id) or {}).items():
            if field not in ("note", "caveat", "interpretation"):
                raise ValueError(f"{task_id}: {field!r} is not a human field; "
                                 "computed fields are not editable at this gate")
            row[field] = text
            row.setdefault("authored", []).append(field)
        tasks[task_id] = row
    pack = {"evidence_version": proposal.get("evidence_version", 1),
            "project_id": proposal.get("project_id"),
            "runs_root": proposal.get("runs_root"),
            "tasks": tasks,
            "families_not_declared": proposal.get("families_not_declared", [])}
    assert_not_authored(pack)
    return {"pack": pack, "provenance": provenance(pack)}


def halt(state: PromoteState) -> dict:
    """approval withheld. the proposal stands as a record; the pack is not written."""
    return {"halted": True, "pack": {}}


def _after_gate(state: PromoteState) -> str:
    return "emit" if state.get("approved") else "halt"


def build_promote_graph(checkpointer=None):
    g = StateGraph(PromoteState)
    g.add_node("compute", compute)
    g.add_node("ask", ask)
    g.add_node("emit", emit)
    g.add_node("halt", halt)
    g.add_edge(START, "compute")
    g.add_edge("compute", "ask")
    g.add_conditional_edges("ask", _after_gate, {"emit": "emit", "halt": "halt"})
    g.add_edge("emit", END)
    g.add_edge("halt", END)
    return g.compile(checkpointer=checkpointer)
