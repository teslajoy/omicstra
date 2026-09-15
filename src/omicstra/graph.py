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

import logging
import operator
import os
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from omicstra.contracts.project import ProjectConfig
from omicstra.routing import load_routing_evidence
from omicstra.settings import settings

_log = logging.getLogger(__name__)

# "refuse" is a third arm and not an error state. a question the contract does
# not declare has no method behind it, so neither resolving nor computing is
# available - and saying so is the answer rather than a failure to answer.
Arm = Literal["ask", "compute", "refuse"]


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
    has_evidence: bool            # does the cohort hold ANY evidence
    has_evidence_for_task: bool   # does it hold evidence for the task asked
    arm: Arm
    arm_reason: str
    # --- shared with subgraphs BY NAME ------------------------------------
    # a key the parent does not declare is DROPPED at the boundary, in both
    # directions. so every key a subgraph reads from or writes to the parent
    # must appear here, or it silently vanishes and the subgraph raises a
    # KeyError that looks like a bug in the subgraph.
    adata_path: str          # produced by inventory A7, consumed by eda.profile
    params: dict             # per-step params, cohort-supplied
    # --- the encode subgraph's half of the boundary ------------------------
    # the gate decides from DECLARATIONS plus counts the inventory already
    # measured. it re-measures nothing, which is why `unit_counts` crosses the
    # boundary rather than being recomputed one level down.
    cohort: dict             # the cohort declaration, read once by discover
    encoder: str             # which encoder the gates are being asked about
    unit_counts: dict        # sample -> n_obs, lifted from the inventory record
    min_scope: int | None    # declared minimum for an authoritative run
    gates: list[dict]
    answers: dict
    report: dict
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
    """which arm answers THIS question - resolved per task, not per cohort.

    absent evidence is not an error and not a fallback: a cohort that has never
    been evaluated cannot be routed from another cohort's table, so it goes to
    COMPUTE rather than borrowing an answer.

    the arm is a property of the QUESTION crossed with the cohort, not of the
    cohort alone. this read `bool(ev.get("tasks"))`, so a cohort holding evidence
    for one task of seven sent all seven to ask - and the six without evidence
    came back "this cohort records no evidence for X", which is a refusal
    standing where a routing decision belongs. "that needs compute, and here are
    the gates" is the true answer, and the compute arm exists to give it.

    three outcomes, and the third is the only real refusal:

        ask       this task has recorded evidence -> resolve from it
        compute   the task is in the contract, this cohort has not run it
        refuse    the task is not in the contract at all - nothing to compute
    """
    ev = load_routing_evidence(state.get("project_id"))
    tasks = ev.get("tasks") or {}
    task_id = state.get("task_id")
    has = bool(tasks)

    # the cohort declaration is read HERE, once, and travels. the encode gate
    # resolves five of its questions from it, and a subgraph that loaded the
    # file itself would read it again per invocation and could disagree with
    # what the parent recorded.
    cohort = _load_cohort(state.get("project_id"))

    arm, why = _arm_for(task_id, tasks, state.get("project_id"))
    out = {"has_evidence": has, "has_evidence_for_task": bool(task_id and task_id in tasks),
           "arm": arm, "arm_reason": why, "cohort": cohort}
    if not state.get("encoder"):
        out["encoder"] = _primary_encoder(state.get("project_id"))
    return out


def _known_task(task_id: str, project_id: str | None) -> bool:
    """is this a task the CONTRACT declares, whatever this cohort has run?

    the distinction carries the refusal. a declared task with no evidence here is
    work not yet done; an undeclared one is a question the system has no method
    for, and no amount of compute produces one.
    """
    from omicstra.routing import list_task_families

    try:
        fams = list_task_families(project_id=project_id)
    except Exception:                       # noqa: BLE001 - unreadable contract refuses
        return False
    return any(f.get("task_id") == task_id for f in fams.get("families", []))


def _arm_for(task_id: str | None, tasks: dict,
             project_id: str | None) -> tuple[Arm, str]:
    """the arm for one task, and the reason, which travels into the record."""
    if not task_id:
        # no task named: the caller is asking about the cohort rather than a
        # question. evidence decides, as it did before.
        return ("ask" if tasks else "compute",
                "no task named - the arm follows whether the cohort has any evidence")
    if task_id in tasks:
        return "ask", f"{task_id} has recorded evidence on this cohort"
    if _known_task(task_id, project_id):
        return "compute", (f"{task_id} is a declared task family and this cohort has "
                           "not run it - that is work outstanding, not a refusal")
    return "refuse", (f"{task_id} is not a task family the contract declares, so there "
                      "is no method to run and nothing to compute")


def _load_cohort(project_id: str | None) -> dict:
    """the cohort's declarations, or an empty dict.

    absent is not an error: a scaffolded cohort has declared nothing yet, and
    every gate then stays open, which is the correct fail-closed result rather
    than a crash.
    """
    import json

    p = settings.project_root(project_id) / "cohort.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _primary_encoder(project_id: str | None) -> str:
    """the encoder the cohort declares as primary, by name.

    the gates are asked about ONE encoder at a time because their answers differ
    per encoder - a gated model raises the terms question and an ungated one does
    not, and only some declare a unit floor.
    """
    try:
        cfg = ProjectConfig.load(project_id)
    except Exception as e:     # noqa: BLE001 - absent or unreadable is not an error here
        # a scaffolded cohort has no usable project.json yet, and the gates
        # staying open on an unnamed encoder is the correct fail-closed answer.
        _log.debug("no project config for %r: %s", project_id, e)
        return ""
    for enc in cfg.encoders:
        if getattr(enc, "role", "") == "primary" and getattr(enc, "name", ""):
            return enc.name
    return ""


def unit_counts_from_records(records: list[dict]) -> dict[str, int]:
    """sample -> n_obs, taken from the inventory's shape record.

    the compute contract says the platform_floor gate "re-asks nothing" and
    reads counts already measured. this is that reading: the inventory recorded
    n_obs per sample when it described the data, so the gate consumes it rather
    than opening the objects again.
    """
    for r in records:
        if r.get("step_id") == "shape":
            per = (r.get("observed") or {}).get("per_sample") or []
            return {s["sample"]: int(s["n_obs"]) for s in per
                    if s.get("sample") and s.get("n_obs") is not None}
    return {}


def _arm(state: OmicstraState) -> str:
    return state.get("arm", "compute")


def carry_counts(state: OmicstraState) -> dict:
    """lift per-sample unit counts out of the inventory records, for the gate.

    it exists because the encode gate must not re-measure what the inventory
    described - the compute contract says the platform_floor gate "re-asks
    nothing" - and a subgraph cannot reach the parent's records except through a
    declared key.
    """
    return {"unit_counts": unit_counts_from_records(state.get("records", []))}


def _halted(state: OmicstraState) -> str:
    """a stopped gate stops the run. encode does not start after a halt.

    without this the eda subgraph could refuse a cohort and the parent would
    walk straight into spending accelerated compute on it, which is the one
    thing the gate exists to prevent.
    """
    return "halt" if state.get("halted") else "encode"


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
        # a node between them, not a bare edge. the encode gate reads counts the
        # inventory already measured, and something has to lift them out of the
        # records the eda subgraph appended. putting that in the subgraph would
        # make it re-read the inventory; putting it in the edge is not possible.
        g.add_node("carry_counts", carry_counts)
        g.add_edge("eda", "carry_counts")
        g.add_conditional_edges("carry_counts", _halted,
                                {"halt": END, "encode": "encode"})
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

    # refuse terminates at the router when there is one - it can report the
    # reason as a decision - and at END otherwise.
    targets.setdefault("refuse", "route" if route is not None else END)
    targets.setdefault("ask", END)
    targets.setdefault("compute", END)
    g.add_conditional_edges("discover", _arm, targets)

    return g.compile(checkpointer=checkpointer)