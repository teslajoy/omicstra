"""the record contract - what every node emits.

one Record per node, four kinds discriminated on `kind`. the base carries
provenance and the actor; the payload carries what that kind of node actually
produces.

three properties fall out of putting `actor`, `inputs` and `outputs` on the base:

  - the decision ledger is `[r.actor for r in records]`, so the reproducibility
    claim is a number rather than a claim
  - hashed inputs/outputs extend the sha-lock pattern to every node for free
  - kind-specific payloads keep nodes honest: a transform cannot report a
    criterion it never evaluated
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

Status = Literal["pass", "fail", "not_run", "not_applicable", "error"]
# the CLOSED set, and it is the one compute_contract.json#gate_schema declares.
# this read `deterministic | model | human` while the gate nodes were writing
# `client` and `system`, so the records the compute arms emit could not be
# validated as Records at all - which is why they lived in the checkpoint and
# the routing ledger lived in a file, two stores for one claim.
#
# `client` is the distinction the actor ledger exists for: an answer arrived
# through the surface, and the server observed a tool call rather than a person.
# `model` is kept because a routing decision can be one a model mapped onto a
# task_id; it is not a synonym for client.
Actor = Literal["deterministic", "model", "human", "client", "system"]
# `dispatch` joined the four when the compute arms started handing shards to an
# executor: it produces no artifact yet and judges nothing, so it is neither a
# transform nor a diagnostic, and calling it one of those would make the ledger
# say something false about what happened.
Kind = Literal["transform", "diagnostic", "gate", "selection", "dispatch"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_id(*parts: Any, n: int = 12) -> str:
    """content id over the INPUTS to a decision. deterministic everywhere.

    the id exists to answer one question - did swapping the model change any
    decision - and that only works if it is stable across machines, models and
    commits. so `started_at`, `duration_s` and `code_version` are deliberately
    not hashed: an id that moved on every commit could not support the
    comparison it exists for.

    two ledgers join on this id and either differ on outcome, or do not differ.
    """
    canon = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:n]


class ArtifactRef(BaseModel):
    """a file this node read or wrote. sha256 is what makes a re-run checkable."""
    path: str
    sha256: str | None = None
    tracked: bool | None = None  # is it in version control? decides if code_version means anything here
    note: str = ""


class Record(BaseModel):
    """base for every node's emission. never instantiated directly."""
    step_id: str
    kind: Kind
    status: Status = "pass"
    actor: Actor = "deterministic"

    # content id over this decision's inputs - see stable_id(). set by the
    # producer, because each kind hashes a different tuple.
    record_id: str = ""

    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[ArtifactRef] = Field(default_factory=list)
    outputs: list[ArtifactRef] = Field(default_factory=list)

    caveats: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=_now)
    duration_s: float | None = None
    code_version: str | None = None  # git sha of the package at run time

    # the mcp protocol revision this run was produced under, read from `_meta`
    # per request. the 2026-07-28 revision removed the initialize handshake, so
    # there is no session to ask - it travels with every request or not at all.
    # a run is only reproducible if you know which protocol produced it, and
    # this is also what makes REQUIRE_PROTOCOL_2026 mean something after the
    # fact rather than only at the door.
    protocol_version: str | None = None

    def headline(self) -> str:  # overridden per kind
        return f"{self.step_id}: {self.status}"


# --- transform -------------------------------------------------------------
class Produced(BaseModel):
    path: str
    shape: list[int] | None = None
    dtype: str | None = None
    sha256: str | None = None


class TransformRecord(Record):
    """produced an artifact. no verdict, no criterion - it did not judge anything."""
    kind: Literal["transform"] = "transform"
    produced: list[Produced] = Field(default_factory=list)

    def headline(self) -> str:
        n = len(self.produced)
        return f"{self.step_id}: produced {n} artifact{'s' if n != 1 else ''}"


# --- diagnostic ------------------------------------------------------------
class GuardVerdict(BaseModel):
    id: str                 # G1..G6
    passed: bool
    value: float | None = None
    note: str = ""


class DiagnosticRecord(Record):
    """observed something and compared it against a declared criterion.

    this is the scientific unit: method and params say what was done, scope says
    on what, observed says what was seen, criterion says what would count, result
    states the comparison, decision states what follows. a status alone is not a
    finding.
    """
    kind: Literal["diagnostic"] = "diagnostic"
    method: str = ""                    # "Moran's I, k-NN spatial weights, permutation null"
    scope: str = ""                     # "1,075 tissue-selected spots, CN1/C1 (1 of 281)"
    observed: dict[str, Any] = Field(default_factory=dict)
    criterion: str = ""                 # "I > 0.3 for >= 3 of top 5 positive markers"
    result: str = ""                    # "5 of 5 above threshold, all p < 0.001"
    decision: str = ""                  # "spatial structure present -> spatial GNN branch"
    guards: list[GuardVerdict] = Field(default_factory=list)

    def headline(self) -> str:
        bits = [b for b in (self.result, self.decision) if b]
        return f"{self.step_id}: " + " — ".join(bits) if bits else f"{self.step_id}: {self.status}"


# --- gate ------------------------------------------------------------------
class GateRecord(Record):
    """resolved to an edge. the verdict IS the output."""
    kind: Literal["gate"] = "gate"
    verdict: str = ""                   # proceed | proceed_with_caution | stop | exists | new | winner
    edge_taken: str = ""
    escalated: bool = False
    reason: str = ""
    violations: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)

    def headline(self) -> str:
        s = f"{self.step_id}: {self.verdict}"
        return s + f" — {self.reason}" if self.reason else s


# --- selection -------------------------------------------------------------
class Candidate(BaseModel):
    name: str
    metric: str = ""
    value: float | None = None
    ci: list[float] | None = None
    evidence: str = ""


class SelectionRecord(Record):
    """chose among candidates, or declined to.

    covers two nodes that look nothing alike operationally - picking an encoder
    and picking a method - and are identical in shape. `chosen is None` with
    `tie=True` is an escalation: the system does not pick.

    `consequence` is load-bearing. numbers alone do not tell a human what the
    choice forecloses.
    """
    kind: Literal["selection"] = "selection"
    candidates: list[Candidate] = Field(default_factory=list)
    # ONE choice, or a mapping of gate id -> option. the router picks a method,
    # which is one string; a gate arm answers every open gate in a single call,
    # which is a mapping, and squeezing that into a string lost which answer
    # belonged to which gate. `headline()` renders both.
    chosen: str | dict[str, str] | None = None
    tie: bool = False
    # who is actually tied. `candidates` carries every method that was scored,
    # including ones that fell below the floor or were contraindicated, so it
    # must not be used to name a tie - that reports a method the evidence
    # separated, or excluded, as though it were defensible.
    tied: list[str] = Field(default_factory=list)
    # a tie and an escalation both decline to pick, and they are not the same
    # thing. a tie means the measurement does not order the candidates; an
    # escalation means the contract's rules cannot resolve the task at all -
    # the axes disagree, or this cohort has no evidence. same field name as
    # GateRecord, because it is the same idea one layer down.
    escalated: bool = False
    contraindicated: bool = False
    why: str = ""
    consequence: str = ""               # what choosing this forecloses downstream

    def headline(self) -> str:
        if self.contraindicated:
            verb = "overridden" if self.actor == "human" else "contraindicated"
            return f"{self.step_id}: {verb} — {self.why}"
        if self.escalated:
            return f"{self.step_id}: escalated — {self.why}"
        if self.tie:
            src = self.tied or [c.name for c in self.candidates]
            names = " | ".join(dict.fromkeys(src))
            return f"{self.step_id}: tie between {names} — the system does not pick"
        if isinstance(self.chosen, dict):
            return (f"{self.step_id}: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(self.chosen.items())))
        return f"{self.step_id}: {self.chosen}"


class DispatchRecord(Record):
    """handed N shards to an executor. no verdict, and no artifact yet.

    `executor` is a FIELD and not a param because a resume reads it first:
    re-dispatching a thread and re-attaching to a workflow are different
    recoveries, and a ledger that buried the distinction in a params blob would
    make the resume path guess.
    """
    kind: Literal["dispatch"] = "dispatch"
    executor: str = ""
    n_shards: int | None = None
    n_pending: int | None = None

    def headline(self) -> str:
        n = "?" if self.n_shards is None else self.n_shards
        return f"{self.step_id}: {n} shard(s) to {self.executor or 'an executor'}"


AnyRecord = (TransformRecord | DiagnosticRecord | GateRecord | SelectionRecord
             | DispatchRecord)


_KINDS: dict[str, type[Record]] = {
    "transform": TransformRecord,
    "diagnostic": DiagnosticRecord,
    "gate": GateRecord,
    "selection": SelectionRecord,
    "dispatch": DispatchRecord,
}

# DERIVED from `Actor`, not hand-copied beside it. this was a second literal
# listing the same five strings, which existed only because a `Literal` cannot
# be iterated at runtime - the same cause as the routing arm being an
# unenumerable Literal, and `test_one_ledger.py` names the risk in its own
# words: "a closed set declared in one file and enforced in another is two
# sets". `actor_ratio` counts from this, so a drift between the two would have
# been a KeyError on the one actor the ledger was extended to carry.
#
# the field TYPE stays a Literal rather than becoming a StrEnum: these are
# pydantic fields on every record in the ledger, and changing their type is a
# serialisation question for the whole committed store, not a cleanup.
_ACTORS: tuple[str, ...] = get_args(Actor)
_VOLATILE = ("started_at", "duration_s", "code_version", "record_id")


def coerce(d: dict) -> Record:
    """a graph node's record dict as a Record. the one store's door.

    the gate nodes build plain dicts because a langgraph state value has to be
    JSON, and they carry keys no Record declares - `executor`, `n_shards`,
    `chosen`, `pins`. those are not noise and are not droppable: which executor
    ran a shard is the first thing a resume needs. so they go to `params`, which
    is the field that exists for exactly this, and the record validates.

    `record_id` is content-addressed over the dict when absent, which is what
    makes appending the same run's records twice - a resume re-entering a graph
    carries the whole list forward - idempotent rather than duplicative.
    """
    cls = _KINDS.get(str(d.get("kind", "")), Record)
    known = set(cls.model_fields)
    core = {k: v for k, v in d.items() if k in known}
    extra = {k: v for k, v in d.items() if k not in known}
    if extra:
        core["params"] = {**(core.get("params") or {}), **extra}
    if not core.get("record_id"):
        # the DECISION, not the clock - the same exclusions stable_id() states.
        # an id over the whole dict would move on every call, and the id exists
        # to answer whether two ledgers differ on an outcome.
        core["record_id"] = stable_id({k: v for k, v in d.items()
                                       if k not in _VOLATILE})
    return cls.model_validate(core)


def timed_node(fn):
    """stamp the records a graph node ADDS with how long that node took.

    every record already carries `started_at`, so a ledger has always given a
    reader an order. none carried `duration_s`, so it gave an order and no cost -
    and "where did this run spend its time" is the question a trace exists for.

    the node returns the whole accumulated list, so only the tail is stamped:
    the ones already in `state` were timed by the node that made them, and
    re-stamping them would attribute every earlier step's cost to this one.

    a wrapper rather than a timer at each call site, because a node that forgets
    to stop its own stopwatch is a node that reports a plausible wrong number.
    """
    import functools
    import time

    @functools.wraps(fn)
    def wrapper(state, *a, **kw):
        t0 = time.perf_counter()
        out = fn(state, *a, **kw)
        if not isinstance(out, dict) or not out.get("records"):
            return out
        before = len(state.get("records") or []) if isinstance(state, dict) else 0
        d, now = round(time.perf_counter() - t0, 3), _now()
        # `started_at` too, because a node's raw dict is not a Record yet - the
        # default only fires in `coerce`, on the way into the ledger. so the
        # checkpoint and the ledger disagreed about whether a step had a time,
        # and the two are meant to be two VIEWS of one store.
        # `is None`, not `or`. a step that did not run takes ~0 s, which rounds
        # to 0.0 and is FALSY - so `or` replaced three known zeros with the
        # node's 148.1 s and the trace summed 588.6 s for a 150 s run. a zero is
        # a measurement; only None means nobody measured.
        out["records"] = [*out["records"][:before],
                          *({**r, "started_at": r.get("started_at") or now,
                             "duration_s": (d if r.get("duration_s") is None
                                            else r["duration_s"])}
                            for r in out["records"][before:])]
        return out

    return wrapper


class RecordStore:
    """the ledger. append-only within a run.

    optionally file-backed: a store bound to a path appends each record as it
    arrives and reloads what is already there on open. that is what makes the
    ledger survive a process boundary - the MCP server answers each tool call in
    its own call frame, and a purely in-memory ledger would report one decision
    however many had actually been taken.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._records: list[Record] = []
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self._reload()

    def _reload(self) -> None:
        assert self.path is not None
        for line in self.path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                self._records.append(_KINDS.get(d.get("kind", ""), Record).model_validate(d))

    def append(self, r: Record) -> Record:
        self._records.append(r)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(r.model_dump_json() + "\n")
        return r

    def append_new(self, records) -> list[Record]:
        """append the ones not already here, by record_id. returns what landed.

        the arms carry their whole record list forward in the graph state, so a
        resume hands the store everything the run has ever decided. without the
        id check a three-gate run that was resumed twice would read as nine
        decisions, and the actor ratio is a published number.
        """
        have = {r.record_id for r in self._records if r.record_id}
        out = []
        for d in records:
            r = d if isinstance(d, Record) else coerce(d)
            if r.record_id and r.record_id in have:
                continue
            have.add(r.record_id)
            out.append(self.append(r))
        return out

    def all(self) -> list[Record]:
        return list(self._records)

    def by_kind(self, kind: Kind) -> list[Record]:
        return [r for r in self._records if r.kind == kind]

    def actor_ratio(self) -> dict[str, int]:
        """the reproducibility claim as a number: N decisions, X deterministic, ...

        every actor in the vocabulary, counted from the vocabulary rather than
        from a literal: the three hardcoded keys raised KeyError the first time a
        `client` record reached the store, which is the one actor the ledger was
        extended to be able to say.
        """
        out = dict.fromkeys(_ACTORS, 0)
        for r in self._records:
            out[r.actor] += 1
        return out

    def to_jsonl(self) -> str:
        return "\n".join(r.model_dump_json() for r in self._records)