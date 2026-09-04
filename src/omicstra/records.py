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
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["pass", "fail", "not_run", "not_applicable", "error"]
Actor = Literal["deterministic", "model", "human"]
Kind = Literal["transform", "diagnostic", "gate", "selection"]


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
    chosen: str | None = None
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
        return f"{self.step_id}: {self.chosen}"


AnyRecord = TransformRecord | DiagnosticRecord | GateRecord | SelectionRecord


_KINDS: dict[str, type[Record]] = {
    "transform": TransformRecord,
    "diagnostic": DiagnosticRecord,
    "gate": GateRecord,
    "selection": SelectionRecord,
}


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

    def all(self) -> list[Record]:
        return list(self._records)

    def by_kind(self, kind: Kind) -> list[Record]:
        return [r for r in self._records if r.kind == kind]

    def actor_ratio(self) -> dict[str, int]:
        """the reproducibility claim as a number: N decisions, X deterministic, ..."""
        out = {"deterministic": 0, "model": 0, "human": 0}
        for r in self._records:
            out[r.actor] += 1
        return out

    def to_jsonl(self) -> str:
        return "\n".join(r.model_dump_json() for r in self._records)