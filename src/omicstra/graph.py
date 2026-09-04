"""level 0 - the only entry point, and the only place a checkpointer is passed.

three levels, and one rule decides which a thing is:

    a thing gets its own GRAPH only if it can ask a human.
    everything else is a PROTOCOL.

    level 0  this file         routes between subgraphs. owns the checkpointer.
    level 1  eda_graph,        one compiled StateGraph per subgraph, each added
             agents/graph      as ONE node. each holds exactly one interrupt.
    level 2  protocols/        langchain Runnables. fixed order, no state, no
                               interrupt, ever. a transcription of a published
                               method, not a design.

why the checkpointer lives here and nowhere else
------------------------------------------------
`interrupt()` requires one. a subgraph compiled with its own would checkpoint to
a thread the caller cannot resume, so subgraphs compile with `checkpointer=None`
and INHERIT the parent's. an interrupt raised inside `eda` pauses the parent at
the `eda` node; the caller resumes with `Command(resume=...)` against the parent
using the same `thread_id`. so the CLI and the MCP server only ever talk to one
graph, and there is exactly one place that decides durability.

state
-----
subgraphs share keys with the parent BY NAME. a key a subgraph writes that the
parent does not declare stays inside the subgraph. `records` needs a reducer
because more than one node appends to it; scalars do not.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from omicstra.routing import load_routing_evidence

Arm = Literal["ask", "compute"]


# MUST be module level. under `from __future__ import annotations` a
# function-local TypedDict raises NameError when langgraph resolves node
# annotations against module globals, and only when a conditional edge is
# present - so it surfaces late. same constraint as EDAState and RouteState.
class OmicstraState(TypedDict, total=False):
    # inputs
    question: str
    task_id: str | None
    project_id: str | None
    project_dir: str | None
    # discovery
    has_evidence: bool
    arm: Arm
    # --- shared with subgraphs BY NAME ------------------------------------
    # a key the parent does not declare is DROPPED at the boundary, in both
    # directions. so every key a subgraph reads from or writes to the parent
    # must appear here, or it silently vanishes and the subgraph raises a
    # KeyError that looks like a bug in the subgraph.
    adata_path: str          # produced by inventory A7, consumed by eda.profile
    params: dict             # per-step params, cohort-supplied
    # the reducer is why more than one node may append. without it the second
    # writer OVERWRITES the first - invisible until two nodes both write.
    records: Annotated[list[dict], operator.add]
    # outcomes
    verdict: str | None
    violations: list[str]
    cautions: list[str]
    escalations: list[dict]
    approved: bool | None
    decision: dict | None
    halted: bool


def discover(state: OmicstraState) -> dict:
    """does this cohort have an evidence pack? the one branch at level 0.

    absent evidence is not an error and not a fallback - it decides which arm
    runs. a cohort that has never been evaluated cannot be routed from another
    cohort's table, so it goes to COMPUTE rather than borrowing an answer.
    """
    ev = load_routing_evidence(state.get("project_id"))
    has = bool(ev.get("tasks"))
    return {"has_evidence": has, "arm": "ask" if has else "compute"}


def _arm(state: OmicstraState) -> str:
    return state.get("arm", "compute")


def build_omicstra_graph(checkpointer: Any = None,
                         eda: Any = None,
                         encode: Any = None) -> Any:
    """the only graph a caller compiles.

    `eda` and `encode` are injected compiled subgraphs so this file does not
    import them at module scope - level 0 must stay importable in an
    environment that has no encoder dependencies installed, which is the same
    reason `pyproject` keeps them in an extra.

    pass `checkpointer=InMemorySaver()` for a single session, or a durable saver
    when a review must survive a restart. passing None means no interrupt in any
    subgraph can fire - that is a valid read-only configuration, not a default
    to rely on.
    """
    g = StateGraph(OmicstraState)
    g.add_node("discover", discover)

    if eda is not None:
        g.add_node("eda", eda)            # a compiled subgraph, added as ONE node
    if encode is not None:
        g.add_node("encode", encode)

    g.add_edge(START, "discover")

    targets: dict[str, str] = {}
    if eda is not None:
        targets["compute"] = "eda"
    if encode is not None and eda is not None:
        g.add_edge("eda", "encode")
        g.add_edge("encode", END)
    elif eda is not None:
        g.add_edge("eda", END)

    # the ASK arm is rule-resolved and holds no interrupt of its own, so it is
    # not a subgraph - it terminates here and the caller reads the record.
    targets.setdefault("ask", END)
    targets.setdefault("compute", END)
    g.add_conditional_edges("discover", _arm, targets)

    return g.compile(checkpointer=checkpointer)