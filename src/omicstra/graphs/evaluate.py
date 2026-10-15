"""LEVEL 1: the evaluate gate. what a number is allowed to claim, asked before it is.

the other three gate sets protect compute or admissibility. this one protects a
CLAIM, which is why it is the last and why its gates look different: nothing here
spends GPU time and nothing here recomputes a published number. what it decides
is whether a verdict may be stated, over which bytes, and with what caveat.

    hypothesis_scope    which hypotheses this run answers, and which of them
                        have no fillable family on this cohort at all
    artifact_pinning    the sha256 of every artifact the verdict reads. pins
                        that DIFFER are a refusal rather than a question
    guard_failure       a guard the contract requires ran and failed. demote,
                        do not discard - the number becomes diagnostic

the verdict is computed BEFORE the gates, which is the opposite of the encode
arm and deliberate. `guard_failure` can only fire on a guard that actually ran,
so the guards have to have run; and computing them costs a few file reads rather
than a training run, so there is nothing to protect by deferring it. the gate
then asks the one thing the measurement cannot settle: what a failed guard does
to the claim.

this is judge #2's gate. the maths is `protocols/evaluate.py` - six guards and
the per-hypothesis gates over the artifacts the evidence pack cites - and no
model is called anywhere on the path. the judgement a person makes here is
authority over a caveat, not a score.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from omicstra.graphs.encode import _accept, _as_dicts, halt
from omicstra.protocols.encode import gate_record
from omicstra.protocols.evaluate import (
    ArtifactDrift,
    EvidencePackMissing,
    claims,
    preflight_evaluate,
    verdicts,
)
from omicstra.records import timed_node


class EvaluateState(TypedDict, total=False):
    project_id: str
    arm: str                   # "evaluate". a key the state does not DECLARE is
                               # dropped at the boundary, and a handle that lost
                               # its arm resumes into the wrong graph
    cohort: dict[str, Any]
    hypotheses: list[str]
    verdict: dict              # computed in `gates`, reported in `evaluate`
    gates: list[dict]
    answers: dict[str, str]
    records: list[dict]
    approved: bool | None
    halted: bool
    report: dict


def gates(state: EvaluateState) -> dict:
    """compute the verdict, then the gates it leaves open.

    two refusals and no question. a cohort with no curated pack has nothing to
    return a verdict OVER, and an artifact whose bytes differ from the pin the
    cohort declared cannot be made evaluable by any answer - so neither is
    offered as a gate.
    """
    hs = list(state.get("hypotheses") or []) or None
    pid = state.get("project_id")
    try:
        v = verdicts(pid, hypotheses=hs)
        reqs = preflight_evaluate(state.get("cohort"), pid, hypotheses=hs, verdict=v)
    except (EvidencePackMissing, ArtifactDrift) as e:
        kind = ("evidence_pack_missing" if isinstance(e, EvidencePackMissing)
                else "artifact_drift")
        return {"halted": True,
                "report": {"status": "refused", "why": str(e), "refusal": kind},
                "records": [*state.get("records", []),
                            {"step_id": "preflight_evaluate", "kind": "gate",
                             "status": "not_run", "actor": "deterministic",
                             "reason": str(e)}]}

    if not any(r["evaluable"] for r in v.values()):
        # not a gate either: every family of every hypothesis asked for is
        # undeclared or declared unavailable, so there is no option set.
        return {"halted": True, "verdict": v,
                "report": {"status": "not_run",
                           "why": ("no hypothesis asked for has a fillable task "
                                   "family on this cohort"),
                           "per_hypothesis": {h: r["families"] for h, r in v.items()}},
                "records": [*state.get("records", []),
                            {"step_id": "preflight_evaluate", "kind": "gate",
                             "status": "not_run", "actor": "deterministic",
                             "reason": "no fillable task family in scope"}]}

    rec = gate_record(reqs, "n/a - the evaluate stage selects no encoder",
                      step_id="preflight_evaluate")
    return {"gates": _as_dicts(reqs), "verdict": v,
            "records": [*state.get("records", []), rec.model_dump(mode="json")]}


def ask(state: EvaluateState) -> dict:
    """HITL. the open gates, with the guard that failed and the bytes in hand."""
    from langgraph.types import interrupt

    open_gates = [g for g in state.get("gates", []) if g["open"]]
    answer = interrupt({
        "question": (f"{len(open_gates)} gate(s) stand between these artifacts and a "
                     "stated verdict."),
        "gates": open_gates,
        "how_to_answer": ("reply with a mapping of gate id -> one of that gate's "
                          'options, e.g. {"guard_failure": "report_as_diagnostic"}'),
        "consequence": ("a verdict inherits what is answered here. a number whose "
                        "guard failed is diagnostic wherever it is cited, and a "
                        "verdict evaluated unpinned names no bytes."),
        "no_default": "the system does not pick",
    })
    return _accept(state, answer, step_id="preflight_evaluate")


def evaluate(state: EvaluateState) -> dict:
    """state the verdict, carrying every caveat that was answered for.

    computes nothing: `gates` already did, because the guard gate needed it. this
    node's whole job is to attach the authority answers to the measurement and
    refuse if a gate was left open.
    """
    still_open = [g for g in state.get("gates", []) if g["open"]]
    if still_open:
        return {"report": {"status": "refused",
                           "why": f"reached the verdict with {[g['id'] for g in still_open]} open"}}

    answered = state.get("answers", {})
    v = state.get("verdict") or {}
    pins = next((g.get("observed") or {} for g in state.get("gates", [])
                 if g["id"] == "artifact_pinning"), {})
    pinning = answered.get("artifact_pinning") or next(
        (g.get("answer") for g in state.get("gates", []) if g["id"] == "artifact_pinning"),
        None)

    hyp = {}
    for h, r in sorted(v.items()):
        failed = r["guards_failed"]
        hyp[h] = {
            "evaluable": r["evaluable"],
            # the whole point of the guard gate. `diagnostic` is a verdict and
            # not an error - the number stands and may not be cited as biology -
            # and `unguarded` is the state where nothing checked it at all.
            "claims": claims(r),
            "guards_ran": r["guards_ran"],
            "guards_failed": failed,
            "guards_not_run": r["guards_not_run"],
            "families": r["families"],
            "rollup": r["rollup"],
            "gates": r["gates"],
        }

    records = list(state.get("records", []))
    if pinning == "pin_these_bytes":
        # the PROPOSAL, in the ledger. the package does not write a cohort's
        # declaration - a pin is a committed human act, and a server that wrote
        # one would be attesting bytes on its own account.
        records.append(
            {"step_id": "evaluate_pins", "kind": "selection", "status": "pass",
             "actor": "client", "confirmed_by": None,
             "chosen": {"artifact_pinning": "pin_these_bytes"},
             "pins": pins.get("sha256") or {},
             "note": ("the sha256 of every artifact this verdict read. commit it "
                      "to cohort.json#evidence_pins to make the next run refuse "
                      "on drift - this record is the proposal, not the pin")})

    records += _verdict_records(hyp, pins)
    return {"records": records,
            "report": {"status": "evaluated",
                       "hypotheses": hyp,
                       "pinning": pinning,
                       "artifacts": {k: pins.get(k) for k in
                                     ("runs_root", "n_artifacts", "absent", "matched",
                                      "unpinned", "named_without_location")},
                       "answers": answered,
                       "note": ("every number here is read from an artifact the "
                                "cohort's evidence_sources declaration names. "
                                "nothing is recomputed, and a gate whose input is "
                                "missing says which input.")}}


def _verdict_records(hyp: dict, pins: dict) -> list[dict]:
    """one DiagnosticRecord per hypothesis, so the verdict outlives the run.

    it used to live only in the run's `report`, which is state and not a record -
    so it reached the checkpoint and stopped there, visible to a caller holding
    the handle and to nobody else. the report takes a pack and a LEDGER, so a
    verdict that never became a record could never reach it.

    the shape already existed: method / scope / observed / criterion / result /
    decision is the scientific unit every EDA step emits, and a hypothesis
    verdict is the same unit one layer up.
    """
    from omicstra.records import DiagnosticRecord, GuardVerdict, stable_id

    out = []
    for h, d in sorted(hyp.items()):
        guards = [GuardVerdict(id=g, passed=g not in d["guards_failed"],
                               note="required by the contract for this hypothesis")
                  for g in d["guards_ran"]]
        rec = DiagnosticRecord(
            step_id=f"verdict::{h}", status="pass" if d["evaluable"] else "not_run",
            actor="deterministic",
            record_id=stable_id("verdict", h, d["claims"], sorted(d["guards_failed"]),
                                sorted(pins.get("sha256", {}).items())),
            method="the guards the routing contract requires for this hypothesis, "
                   "over the artifacts the cohort's evidence declaration names",
            scope=f"{len(d['families']['fillable'])} fillable task family(s); "
                  f"{len(pins.get('sha256') or {})} artifact(s) read",
            observed={"families": d["families"], "rollup": d["rollup"],
                      "guards_not_run": d["guards_not_run"],
                      "artifacts": pins.get("sha256") or {}},
            criterion="every required guard runs and passes, over pinned bytes",
            result=d["claims"],
            decision=_DECISION[d["claims"]],
            guards=guards)
        out.append(rec.model_dump(mode="json"))
    return out


# what each claim licenses. stated once, here, so a reader of the ledger does not
# have to remember which of the four is a result and which is an absence.
_DECISION = {
    "biological": "may be cited as a biological result",
    "diagnostic": "stands, and may NOT be cited as biology alone - a required "
                  "guard ran and failed",
    "unguarded": "computed, but no required guard ran. not a clean result and "
                 "not a failure - nothing checked it",
    "not_run": "no fillable task family on this cohort for this hypothesis",
}


def _route_gates(state: EvaluateState) -> str:
    if state.get("halted"):
        return "halt"
    return "ask" if any(g["open"] for g in state.get("gates", [])) else "evaluate"


def _route_answer(state: EvaluateState) -> str:
    return "evaluate" if state.get("approved") else "halt"


def build_evaluate_graph(checkpointer=None):
    """gates -> (ask) -> evaluate, with halt on refusal or an unfillable scope."""
    g = StateGraph(EvaluateState)
    g.add_node("gates", timed_node(gates))
    g.add_node("ask", timed_node(ask))
    g.add_node("evaluate", timed_node(evaluate))
    g.add_node("halt", timed_node(halt))

    g.add_edge(START, "gates")
    g.add_conditional_edges("gates", _route_gates,
                            {"ask": "ask", "evaluate": "evaluate", "halt": "halt"})
    g.add_conditional_edges("ask", _route_answer, {"evaluate": "evaluate", "halt": "halt"})
    g.add_edge("evaluate", END)
    g.add_edge("halt", END)
    return g.compile(checkpointer=checkpointer)


__all__ = ["build_evaluate_graph", "evaluate"]
