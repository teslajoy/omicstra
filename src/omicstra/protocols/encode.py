"""the encode chain - "produce the vectors", and what must be settled first.

this is LEVEL 2, so it decides nothing a person should decide. it computes what
each preflight gate would be asked, resolves the ones a cohort has declared an
answer to, and returns the rest as open requests. `graphs/encode.py` is what
turns an open request into an `interrupt()`; nothing here pauses, because a
protocol that can block is a graph wearing the wrong name.

the split is the point
----------------------
    protocols/encode.py   what the gates ARE on this cohort, measured
    graphs/encode.py      asks the person, records the answer, resumes

so the gate logic is testable without a checkpointer, a thread_id or a human,
and the same evidence reaches a CLI, an MCP client and a scheduler unchanged.

two modes, same as align
------------------------
    compute=False   DEFAULT. resolve what a previous extraction produced and
                    refuse when absent. a host serving a published pack never
                    loads a tensor library.
    compute=True    run the encoder. gated behind `[encode]`, and every preflight
                    gate must be resolved before this is allowed to start.

the acceptance test is NOT a re-extraction. 75 GB of embeddings are already
cached and they are the oracle: extract a slice, diff against the cache. that
tests our port rather than testing Virchow2, and it runs in minutes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omicstra.models.encoders import EncoderUnavailable, spec
from omicstra.records import DiagnosticRecord

Ctx = dict[str, Any]

_CONTRACT = Path(__file__).resolve().parents[1] / "configs" / "compute_contract.json"


class ComputeRefused(RuntimeError):
    """compute was asked for while a preflight gate is still open.

    deliberately not the same exception as `ComputeUnavailable`, which means the
    artifact is missing. "nobody has answered yet" and "it is not there" are
    different states and the ledger has to be able to tell them apart.
    """


@dataclass(frozen=True)
class GateRequest:
    """one preflight gate, as it stands on THIS cohort.

    `answer` is None when a person still has to decide. `source` says where a
    resolved answer came from, because a declared answer is still a human answer -
    it is recorded with actor=human and the declaration as its provenance, which
    is what stops a fully-declared cohort's unattended run from looking like the
    system decided anything.
    """
    id: str
    when: str
    question: str
    options: tuple[str, ...]
    forecloses: str
    pre_answerable_by: str | None
    answer: str | None = None
    source: str | None = None
    observed: dict[str, Any] | None = None

    @property
    def open(self) -> bool:
        return self.answer is None


def contract() -> dict:
    return json.loads(_CONTRACT.read_text())


def _declared(cohort: dict, pointer: str | None):
    """resolve `cohort.json#field` against the cohort declaration.

    the pointer names its own file, so a gate answerable from somewhere else
    cannot be silently answered from here. anything that is not a cohort.json
    pointer returns nothing rather than guessing where to look.
    """
    if not pointer or not pointer.startswith("cohort.json#"):
        return None
    return cohort.get(pointer.split("#", 1)[1])


def _resolve_option(value, options: tuple[str, ...], gate_id: str) -> str:
    """a declared answer must land on ONE of the gate's closed options.

    the contract says the option set is closed, and until this existed that was
    true of the interrupt path only - a declaration could put any string in the
    `answer` field and it would travel into the ledger as though a person had
    picked it. the closed set has to be closed on both paths or it is not closed.

    a readable shorthand is allowed (`drop` for `drop_below_floor`) because the
    declaration is written by a person, but only when it identifies exactly one
    option. ambiguous or unknown raises, naming what was offered - a near-miss
    silently ignored is how a cohort ends up running a policy nobody chose.

    free-text gates - anything whose options are not a closed vocabulary, like a
    path or a token source - pass through unchanged.
    """
    if value is None:
        return None
    v = str(value)
    if v in options:
        return v
    if gate_id in _FREE_TEXT:
        return v
    hits = [o for o in options if v == o or v in o.split("_")]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(
        f"{gate_id}: declared answer {v!r} is not one of the gate's options "
        f"{list(options)}"
        + (f" - it matches {hits}, which is ambiguous" if hits else "")
        + ". the option set is closed; a declaration cannot widen it.")


# gates whose answer is a value, not a choice: a token source and an output path
# are supplied, not selected, so their `options` describe the shape of the
# decision rather than enumerating the legal answers.
_FREE_TEXT = frozenset({"gated_weights", "capacity", "encoder_compatibility"})


def preflight(cohort: dict, encoder: str, *, unit_counts: dict[str, int] | None = None,
              min_scope: int | None = None) -> list[GateRequest]:
    """every preflight gate, resolved as far as the declarations allow.

    deterministic and side-effect free: same cohort and same counts give the same
    list. it reads the inventory's counts rather than re-measuring them - the
    platform_floor gate's own contract entry says it re-asks nothing.
    """
    try:
        es = spec(encoder)
    except EncoderUnavailable:
        es = None

    out: list[GateRequest] = []
    for g in contract()["gates"]:
        if g["when"] != "preflight":
            continue
        answer = _resolve_option(_declared(cohort, g.get("pre_answerable_by")),
                                 tuple(g["options"]), g["id"])
        observed: dict[str, Any] = {}

        if g["id"] == "gated_weights":
            if es is not None and not es.gated:
                # not applicable rather than answered. an ungated encoder was
                # never asked this, and recording a 'yes' would imply someone was.
                continue
            observed = {"encoder": encoder, "gated": bool(es and es.gated)}

        elif g["id"] == "platform_floor":
            floor = es.min_units if es else None
            if floor is None:
                continue                     # this encoder declares no floor
            counts = unit_counts or {}
            below = sorted(s for s, n in counts.items() if n < floor)
            observed = {"floor": floor, "n_below": len(below),
                        "n_samples": len(counts), "below": below[:10]}
            if counts and not below:
                continue                     # nothing to decide

        elif g["id"] == "scope_below_minimum":
            n = sum((unit_counts or {}).values())
            observed = {"units": n, "declared_minimum": min_scope}
            if min_scope is None or n >= min_scope:
                continue

        elif g["id"] == "encoder_compatibility":
            observed = {"encoder": encoder,
                        "trained_on": es.trained_on if es else None,
                        "probe": "not_run"}

        out.append(GateRequest(
            id=g["id"], when=g["when"], question=g["question"],
            options=tuple(g["options"]), forecloses=g["forecloses"],
            pre_answerable_by=g.get("pre_answerable_by"),
            answer=answer,
            source=g.get("pre_answerable_by") if answer is not None else None,
            observed=observed or None))
    return out


def gate_record(reqs: list[GateRequest], encoder: str) -> DiagnosticRecord:
    """the preflight state as one record. describes; it does not decide."""
    open_ = [r.id for r in reqs if r.open]
    return DiagnosticRecord(
        step_id="preflight_encode",
        # `not_run`, not `fail`. an unanswered gate is not a failed check - the
        # question has simply not been put to anyone yet, and calling that a
        # failure would put a violation in the ledger where a pending decision
        # belongs. escalating is the graph's job, one level up.
        status="pass" if not open_ else "not_run",
        method="compute_contract preflight gates, resolved against cohort.json",
        scope=f"encoder={encoder}, {len(reqs)} applicable gate(s)",
        observed={r.id: {"answer": r.answer, "source": r.source,
                         **(r.observed or {})} for r in reqs},
        criterion="every preflight gate resolved before compute starts",
        result=(f"{len(reqs) - len(open_)} of {len(reqs)} resolved from declarations"
                if reqs else "no preflight gate applies to this encoder"),
        decision=("proceed to compute" if not open_
                  else f"escalate: {', '.join(open_)} - no default, the system does not pick"))


def assert_clear(reqs: list[GateRequest]) -> None:
    """refuse to compute while any preflight gate is open.

    the contract's own words: an unattended run cannot accept terms on your
    behalf, and a gate answered by the system is not a gate. so this raises
    rather than defaulting, and names what is unanswered and what would answer it.
    """
    if open_ := [r for r in reqs if r.open]:
        lines = []
        for r in open_:
            declared = (f"declare {r.pre_answerable_by} to pre-answer it"
                        if r.pre_answerable_by else "no declaration can pre-answer it")
            lines.append(f"  {r.id} - {r.question}\n"
                         f"    options: {', '.join(r.options)}\n"
                         f"    {declared}\n"
                         f"    forecloses: {r.forecloses}")
        raise ComputeRefused(
            f"{len(open_)} preflight gate(s) unanswered; compute does not start:\n"
            + "\n".join(lines))
