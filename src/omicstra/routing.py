"""task-conditional routing - which method can answer which biological question.

the companion to the EDA gate, and the same split. `configs/routing_contract.json`
is generalizable: the task taxonomy, the outcome vocabulary, and the rules that
turn evidence into an outcome. `{project_root}/routing_evidence.json` is this
cohort's measured evaluation. a cohort with no evidence file is not routable, and
the system says so rather than reusing another cohort's winner.

this module does not read natural language. the caller - a model - reads the task
families, decides which one a question belongs to, and asks the human when the
level of abstraction is ambiguous. everything after the task_id is deterministic:

    model   question -> task_id  (+ proposed_method, when the asker names one)
    rule    task_id  -> route, ties, refusal, guards

that boundary is the tool signature itself rather than a claim in a field, which
is why a model swap changes how a question is understood without changing where
it is sent.

the emission is a `SelectionRecord` - the same shape the encoder-choice node uses,
because picking an encoder and picking a method are the same operation. five
outcomes, in the record's own vocabulary:

    chosen set                       recommend    one candidate leads on every axis
    tie True, escalated False        tie          the measurement does not order them
    tie True, escalated True         escalate     the rules cannot resolve it at all
    contraindicated, actor rule      refuse       the evidence says do not, and why
    contraindicated, actor human     override_ack the caller proceeded anyway

tie and escalate both decline to pick and are not the same thing: a tie is a
statement about the measurement, an escalation is a statement about the rules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omicstra.records import Candidate, RecordStore, SelectionRecord, stable_id
from omicstra.settings import settings


# --- loading ---------------------------------------------------------------
def load_routing_contract(configs_dir: str | Path | None = None) -> dict:
    """ships with the PACKAGE - cohort-free by construction."""
    base = Path(configs_dir) if configs_dir else settings.configs_dir
    return json.loads((settings.resolve(base) / "routing_contract.json").read_text())


def load_routing_evidence(project_id: str | None = None) -> dict:
    """belongs to the PROJECT. absent = this cohort has not been evaluated.

    an absent file must not fall back to a default routing table. inheriting
    another cohort's winner is the exact failure this split exists to stop.
    """
    try:
        p = settings.project_root(project_id) / "routing_evidence.json"
    except ValueError:
        return {}
    return json.loads(p.read_text()) if p.exists() else {}


def ledger(project_id: str | None = None) -> RecordStore:
    """the file-backed decision ledger for a cohort."""
    try:
        pid = project_id or settings.project_root(project_id).name
    except ValueError:
        pid = "unrouted"
    return RecordStore(settings.resolve(settings.runs_dir) / pid / "records" / "decisions.jsonl")


def list_task_families(contract: dict | None = None,
                       project_id: str | None = None) -> dict:
    """the taxonomy the caller classifies against, with per-cohort routability.

    a family the contract defines but this cohort has no evidence for is listed
    as not routable rather than hidden - the caller should be able to say "that
    is a real question, this cohort just cannot answer it".
    """
    contract = contract or load_routing_contract()
    ev = load_routing_evidence(project_id)
    tasks = ev.get("tasks") or {}
    return {
        "scope": ev.get("scope_statement", ""),
        "families": [{"task_id": f["id"], "level": f["level"], "asks": f["asks"],
                      "hypothesis": f["hypothesis"],
                      "routable_on_this_cohort": f["id"] in tasks,
                      "guards": f.get("required_guards", [])}
                     for f in contract["task_families"]],
    }


def _family(contract: dict, task_id: str) -> dict | None:
    return next((f for f in contract["task_families"] if f["id"] == task_id), None)


# --- resolving one metric axis ---------------------------------------------
def _blocks(task: dict) -> list[dict]:
    """a single-axis task is read as one block. keeps both shapes on one path."""
    if task.get("metrics"):
        return task["metrics"]
    return [{k: task[k] for k in
             ("metric", "candidates", "higher_is_better", "floor", "reference", "unit")
             if k in task}]


def _clears(value: float, bar: dict | None, hib: bool) -> bool:
    if not bar or bar.get("value") is None:
        return True
    return value > bar["value"] if hib else value < bar["value"]


class _Axis:
    """one metric resolved: who leads, who is tied with them, what fell short."""

    def __init__(self, blk: dict, names: dict, banned: set[str]):
        self.metric = blk.get("metric", "")
        self.hib = blk.get("higher_is_better", True)
        self.floor, self.reference = blk.get("floor") or {}, blk.get("reference") or {}
        cands = [Candidate(name=names.get(c["id"], c["id"]), metric=self.metric,
                           value=c["value"], ci=c.get("ci"), evidence=c.get("note", ""))
                 for c in blk["candidates"]]
        self.by_name = {c["id"]: names.get(c["id"], c["id"]) for c in blk["candidates"]}
        self.ids = {names.get(c["id"], c["id"]): c["id"] for c in blk["candidates"]}
        cands.sort(key=lambda c: -(c.value or 0) if self.hib else (c.value or 0))
        self.all = cands
        ranked = [c for c in cands if self.ids[c.name] not in banned]
        self.above = [c for c in ranked if _clears(c.value, self.floor, self.hib)]
        self.below = [c for c in ranked if c not in self.above]
        self.leaders: list[Candidate] = []
        if self.above:
            lead = self.above[0]
            self.leaders = [lead] + [c for c in self.above[1:] if self._ties(lead, c)]
        self.beats_reference = (_clears(self.leaders[0].value, self.reference, self.hib)
                                if (self.leaders and self.reference) else None)

    def _ties(self, a: Candidate, b: Candidate) -> bool:
        """overlapping intervals, or equal values with no interval to separate."""
        if a.ci and b.ci:
            return a.ci[0] <= b.ci[1] and b.ci[0] <= a.ci[1]
        return a.value == b.value

    @property
    def leader_ids(self) -> set[str]:
        return {self.ids[c.name] for c in self.leaders}

    def phrase(self) -> str:
        return "; ".join(f"{c.name} {c.value}" + (f" {c.ci}" if c.ci else "")
                         for c in self.leaders)


# --- the decision ----------------------------------------------------------
def resolve(task_id: str, question: str = "", proposed_method: str | None = None,
            override: bool = False, project_id: str | None = None,
            contract: dict | None = None, evidence: dict | None = None,
            log: bool = True) -> SelectionRecord:
    contract = contract or load_routing_contract()
    evidence = evidence if evidence is not None else load_routing_evidence(project_id)
    pid = evidence.get("project_id") or project_id or ""
    scope = evidence.get("scope_statement", "")

    params: dict[str, Any] = {"task_id": task_id, "question": question,
                              "proposed_method": proposed_method, "override": override,
                              "contract_version": contract.get("contract_version"),
                              "evidence_version": evidence.get("evidence_version"),
                              "scope": scope}
    rid = stable_id("route", task_id, contract.get("contract_version"),
                    evidence.get("evidence_version"), proposed_method, override)

    def emit(**kw: Any) -> SelectionRecord:
        r = SelectionRecord(step_id=f"route::{task_id or '?'}", record_id=rid,
                            params=params, **kw)
        if log:
            ledger(project_id).append(r)
        return r

    if not evidence:
        return emit(status="not_run", tie=True, escalated=True,
                    why="this cohort has no routing_evidence.json - it has not been "
                        "evaluated, and another cohort's routing table is not inherited",
                    consequence="derive the evidence for this cohort, or route by hand")

    fam = _family(contract, task_id)
    if fam is None:
        known = ", ".join(f["id"] for f in contract["task_families"])
        return emit(status="not_run", tie=True, escalated=True,
                    why=f"{task_id!r} is not a task family in this contract",
                    consequence=f"known families: {known}")

    task = (evidence.get("tasks") or {}).get(task_id)
    if not task:
        return emit(status="not_run", tie=True, escalated=True,
                    why=f"this cohort records no evidence for {task_id}",
                    consequence="a real question this cohort cannot answer - not "
                                "routable until the evidence is derived")

    names = {k: v for k, v in (evidence.get("method_names") or {}).items() if k != "note"}
    contras = [c for c in evidence.get("contraindications", [])
               if task_id in c.get("applies_to", [])]
    banned = {c["method"] for c in contras}
    guards = [contract["guards"][g] for g in fam.get("required_guards", [])
              if g in contract["guards"]]
    caveats = ([scope] if scope else []) + guards + \
              ([task["caveat"]] if task.get("caveat") else [])

    axes = [_Axis(b, names, banned) for b in _blocks(task)]
    all_cands = [c for a in axes for c in a.all]

    # 1. the asker named a method this cohort contraindicates for this task
    for c in contras:
        if proposed_method in (c["method"], c.get("display"), names.get(c["method"])):
            if not override:
                return emit(candidates=all_cands, contraindicated=True, caveats=caveats,
                            why=f"{c['display']} is contraindicated for this task. "
                                f"{c['statement']}. evidence: {c['evidence']}",
                            consequence="proceeding anyway is available, and is recorded "
                                        "as a decision taken against the evidence")
            return emit(candidates=all_cands, contraindicated=True, tie=False,
                        actor="human", status="pass",
                        chosen=names.get(c["method"], c["method"]),
                        caveats=caveats + [f"OVERRIDE: routed to {c['display']} against "
                                           f"the recorded evidence - {c['evidence']}"],
                        why=f"the system advised against {c['display']} and the caller "
                            f"proceeded. recorded as dissent, not as a recommendation",
                        consequence="every result derived from this route carries the "
                                    "contraindication forward")

    # 2. nothing clears the floor on some axis - the task is refused, not a method
    for a in axes:
        if not a.leaders:
            return emit(candidates=all_cands, contraindicated=True, caveats=caveats,
                        why=f"no method clears this cohort's floor on {a.metric} "
                            f"({a.floor.get('id')} = {a.floor.get('value')})",
                        consequence=contract["decision_rules"]["refusal_rule"]["basis"])

    # 3. every axis must name the same leader. disagreement escalates - it is not
    #    averaged, ranked or resolved by picking an axis.
    agreed = set.intersection(*(a.leader_ids for a in axes))
    detail = " | ".join(f"{a.metric}: {a.phrase()}" for a in axes)

    if not agreed:
        return emit(candidates=all_cands, tie=True, escalated=True, caveats=caveats,
                    why=f"the declared axes name different leaders - {detail}",
                    consequence=contract["decision_rules"]["multi_metric_rule"]["basis"])

    lead_axis = axes[0]
    shortfall = ""
    if lead_axis.beats_reference is False:
        ref = lead_axis.reference
        shortfall = (f"{ref['id']} scores {ref['value']}; the best method here scores "
                     f"{lead_axis.leaders[0].value}. ")

    below = ", ".join(c.name for a in axes for c in a.below)
    tail = f". below the floor: {below}" if below else ""
    banned_note = ("; excluded as contraindicated: "
                   + ", ".join(c["display"] for c in contras)) if contras else ""

    if len(agreed) > 1:
        # display names contain commas, so a comma join reads as more methods
        # than there are. this is a legibility bug that becomes a miscount when
        # someone reads the tie aloud.
        tied = sorted(names.get(i, i) for i in agreed)
        picks = " | ".join(tied)
        return emit(candidates=all_cands, tied=tied, tie=True, caveats=caveats,
                    why=shortfall + detail + ". the intervals overlap - the measurement "
                        "does not order them" + tail + banned_note,
                    consequence=f"{len(agreed)} methods are defensible and the evidence "
                                f"does not choose between them: {picks}")

    chosen_id = next(iter(agreed))
    return emit(candidates=all_cands, chosen=names.get(chosen_id, chosen_id),
                caveats=caveats,
                why=shortfall + detail + tail + banned_note,
                consequence=f"holds for {fam['asks']} on this cohort only; a new cohort "
                            "derives its own evidence or is not routable")


def decision_record(project_id: str | None = None) -> dict:
    """the reproducibility claim as a number, from the ledger."""
    store = ledger(project_id)
    recs = store.all()
    sel = [r for r in recs if r.kind == "selection"]

    def outcome(r: Any) -> str:
        if getattr(r, "contraindicated", False):
            return "override_ack" if r.actor == "human" else "refuse"
        if getattr(r, "escalated", False):
            return "escalate"
        return "tie" if getattr(r, "tie", False) else "recommend"

    return {
        "project_id": project_id or "",
        "total": len(sel),
        "by_outcome": {o: sum(1 for r in sel if outcome(r) == o)
                       for o in ("recommend", "tie", "escalate", "refuse",
                                 "override_ack")},
        "by_actor": {a: sum(1 for r in sel if r.actor == a)
                     for a in ("deterministic", "model", "human")},
        "interpretation_boundary": (
            "question -> task_id happened in the calling model and is not recorded "
            "here. task_id -> route is rule-resolved from the contract plus this "
            "cohort's recorded evidence."),
        "decisions": [{"record_id": r.record_id, "step_id": r.step_id,
                       "outcome": outcome(r), "chosen": getattr(r, "chosen", None),
                       "actor": r.actor} for r in sel],
        "ledger": str(store.path),
    }