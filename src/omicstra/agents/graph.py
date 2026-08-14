"""the agent coordination pattern - orchestrator, three modality agents, judge.

    task_id ──► orchestrator ──┬──► he_agent ──────┐
                               ├──► st_agent ──────┤──► judge ──► END
                               └──► pathway_agent ─┘      │
                                                          └── refuse ──► interrupt()
                                                                            │
                                                                     override_node

the three modality agents run concurrently and each answers only for its own
representation. the judge is the only node that routes, and it is the only node
that may emit a SelectionRecord - the modality agents cannot align or evaluate
(CLAUDE.md agent constraints).

the refusal edge is a real `interrupt()`, not a parameter: the graph stops,
checkpoints, and waits for a person. that is the difference between a system
that reports a contraindication and one that enforces it.

read path throughout - every embedding is cached, so no node touches a GPU.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from omicstra.agents.modality import MODALITY_AGENTS
from omicstra.routing import ledger, resolve


# MUST be module level. under `from __future__ import annotations` a
# function-local TypedDict raises NameError when langgraph resolves node
# annotations against module globals, and only when a conditional edge is
# present - so it surfaces late. same constraint as eda_graph.EDAState.
class RouteState(TypedDict, total=False):
    task_id: str
    question: str
    proposed_method: str | None
    project_id: str | None
    modality: Annotated[list[dict], operator.add]   # concurrent appends
    decision: dict
    resolution: str
    override: bool
    awaiting_human: bool


def _modality_node(name: str):
    """one node per representation. closes over which agent it is."""
    fn = MODALITY_AGENTS[name]

    def node(state: RouteState) -> dict:
        rec = fn(state["task_id"], project_id=state.get("project_id"))
        return {"modality": [rec.model_dump()]}

    return node


def orchestrator(state: RouteState) -> dict:
    """routes between agents only - it does not execute modality steps.

    with the task_id already chosen by the caller, its whole job is to fan out.
    keeping it as a node rather than an implicit edge is what makes the
    dispatch appear in a trace.
    """
    return {"modality": []}


def judge(state: RouteState) -> dict:
    """the only node that routes. consumes the modality reports, resolves, records."""
    rec = resolve(state["task_id"], question=state.get("question", ""),
                  proposed_method=state.get("proposed_method"),
                  override=bool(state.get("override")),
                  project_id=state.get("project_id"), log=False)

    # what each representation said, carried into the record so the decision and
    # its per-modality justification travel together
    for m in state.get("modality", []):
        rec.caveats.append(f"{m['step_id'].split('::')[0]}: {m['decision']}")

    resolution = ("override_ack" if (rec.contraindicated and rec.actor == "human")
                  else "refuse" if rec.contraindicated
                  else "escalate" if rec.escalated
                  else "tie" if rec.tie else "recommend")
    ledger(state.get("project_id")).append(rec)
    return {"decision": rec.model_dump(), "resolution": resolution}


def _after_judge(state: RouteState) -> str:
    """a refusal stops the graph and asks. everything else is terminal."""
    return "ask_human" if state.get("resolution") == "refuse" else "done"


def ask_human(state: RouteState) -> dict:
    """HITL. pauses hard, checkpoints, resumes here.

    the system does not proceed on its own behalf against its own evidence -
    that is what makes the override a decision someone took, rather than a
    default the system chose.
    """
    d = state.get("decision", {})
    answer = interrupt({
        "question": "The recorded evidence contraindicates this method for this "
                    "task. Proceed anyway?",
        "why": d.get("why", ""),
        "consequence": d.get("consequence", ""),
        "if_you_proceed": "the route is taken, the dissent is recorded, and every "
                          "downstream result carries the contraindication",
        "no_default": "the system does not pick",
    })
    proceed = (answer if isinstance(answer, bool)
               else str(answer).strip().lower() in ("yes", "y", "proceed", "true", "override"))
    return {"override": proceed, "awaiting_human": False}


def _after_human(state: RouteState) -> str:
    return "override" if state.get("override") else "done"


def override_node(state: RouteState) -> dict:
    """re-resolve with the human's decision on the record, as dissent."""
    rec = resolve(state["task_id"], question=state.get("question", ""),
                  proposed_method=state.get("proposed_method"), override=True,
                  project_id=state.get("project_id"), log=False)
    for m in state.get("modality", []):
        rec.caveats.append(f"{m['step_id'].split('::')[0]}: {m['decision']}")
    ledger(state.get("project_id")).append(rec)
    return {"decision": rec.model_dump(), "resolution": "override_ack"}


def build_route_graph(checkpointer=None):
    """cohort-free: everything specific arrives in state."""
    g = StateGraph(RouteState)
    g.add_node("orchestrator", orchestrator)
    for name in MODALITY_AGENTS:
        g.add_node(name, _modality_node(name))
    g.add_node("judge", judge)
    g.add_node("ask_human", ask_human)
    g.add_node("override", override_node)

    g.add_edge(START, "orchestrator")
    for name in MODALITY_AGENTS:            # fan out - these run concurrently
        g.add_edge("orchestrator", name)
        g.add_edge(name, "judge")           # fan in - judge waits for all three
    g.add_conditional_edges("judge", _after_judge,
                            {"ask_human": "ask_human", "done": END})
    g.add_conditional_edges("ask_human", _after_human,
                            {"override": "override", "done": END})
    g.add_edge("override", END)
    return g.compile(checkpointer=checkpointer)
