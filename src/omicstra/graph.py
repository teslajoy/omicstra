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
import os
from pathlib import Path
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
    # --- the route subgraph's half of the boundary --------------------------
    # plain types, no reducer. the three modality agents ARE concurrent, but they
    # fan in inside the subgraph, which carries the reducer on its own
    # `modality`; the parent sees one completed list written once, by one node.
    # a reducer here would be cargo - the rule that bites at this boundary is
    # declaration, not merging: drop a name and the value silently vanishes.
    proposed_method: str | None
    modality: list[dict]
    resolution: str | None
    override: bool
    awaiting_human: bool
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


def make_checkpointer(spec: str | None = None):
    """`memory` or `sqlite:<path>`. the http path needs the second.

    InMemorySaver dies with the process, so `resume(run_id)` only works inside
    one invocation - which quietly makes "any instance handles any request"
    false: a second instance, or the same one after a restart, has never heard
    of the thread. a durable saver is what turns that sentence from a claim
    into a fact, and it is the difference between a gate a person can come back
    to tomorrow and one they must answer before their shell exits.

    gscratch cannot host the sqlite file - it has no POSIX locking and names
    sqlite as unsupported. use node-local disk, RDS, or postgres there.
    """
    spec = spec or os.environ.get("OMICSTRA_CHECKPOINT", "memory")
    if spec == "memory":
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver(), "memory"
    if spec.startswith("sqlite:"):
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        path = spec.split(":", 1)[1] or "omicstra_checkpoints.sqlite"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver, f"sqlite:{path}"
    raise ValueError(f"OMICSTRA_CHECKPOINT must be memory or sqlite:<path>, not {spec!r}")


def build_omicstra_graph(checkpointer: Any = None,
                         eda: Any = None,
                         encode: Any = None,
                         route: Any = None) -> Any:
    """the only graph a caller compiles.

    `eda`, `encode` and `route` are injected compiled subgraphs so this file does
    not import them at module scope - level 0 must stay importable in an
    environment that has no encoder dependencies installed, which is the same
    reason `pyproject` keeps them in an extra.

    the two arms take different subgraphs, and they are not interchangeable:
    `eda` (and later `encode`) run the COMPUTE arm, for a cohort with no
    evidence pack; `route` runs the ASK arm, for one that has it.

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
    if route is not None:
        g.add_node("route", route)

    g.add_edge(START, "discover")

    targets: dict[str, str] = {}
    if eda is not None:
        targets["compute"] = "eda"
    if encode is not None and eda is not None:
        g.add_edge("eda", "encode")
        g.add_edge("encode", END)
    elif eda is not None:
        g.add_edge("eda", END)

    # the ASK arm resolves from a lookup table, but it is still a subgraph,
    # because resolving is not the whole job: a contraindicated method stops the
    # graph and asks a person, and that interrupt needs a node to live in. an
    # arm that terminates here can report a contraindication; only one that can
    # pause enforces it.
    if route is not None:
        targets["ask"] = "route"
        g.add_edge("route", END)

    targets.setdefault("ask", END)
    targets.setdefault("compute", END)
    g.add_conditional_edges("discover", _arm, targets)

    return g.compile(checkpointer=checkpointer)