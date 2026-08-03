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

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["pass", "fail", "not_run", "not_applicable", "error"]
Actor = Literal["deterministic", "model", "human"]
Kind = Literal["transform", "diagnostic", "gate", "selection"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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

    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[ArtifactRef] = Field(default_factory=list)
    outputs: list[ArtifactRef] = Field(default_factory=list)

    caveats: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=_now)
    duration_s: float | None = None
    code_version: str | None = None  # git sha of the package at run time

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
    contraindicated: bool = False
    why: str = ""
    consequence: str = ""               # what choosing this forecloses downstream

    def headline(self) -> str:
        if self.contraindicated:
            return f"{self.step_id}: contraindicated — {self.why}"
        if self.tie:
            names = ", ".join(c.name for c in self.candidates)
            return f"{self.step_id}: tie between {names} — escalated"
        return f"{self.step_id}: {self.chosen}"


AnyRecord = TransformRecord | DiagnosticRecord | GateRecord | SelectionRecord


class RecordStore:
    """the ledger. append-only within a run."""

    def __init__(self) -> None:
        self._records: list[Record] = []

    def append(self, r: Record) -> Record:
        self._records.append(r)
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