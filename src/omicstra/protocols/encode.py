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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omicstra.models.encoders import EncoderUnavailable, spec
from omicstra.records import DiagnosticRecord, Produced, TransformRecord

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


def _resolve_option(value, options: tuple[str, ...], gate_id: str,
                    answer_kind: str = "choice") -> str:
    """a declared answer must land on ONE of the gate's closed options.

    the contract says the option set is closed, and until this existed that was
    true of the interrupt path only - a declaration could put any string in the
    `answer` field and it would travel into the ledger as though a person had
    picked it. the closed set has to be closed on both paths or it is not closed.

    a readable shorthand is allowed (`drop` for `drop_below_floor`) because the
    declaration is written by a person, but only when it identifies exactly one
    option. ambiguous or unknown raises, naming what was offered - a near-miss
    silently ignored is how a cohort ends up running a policy nobody chose.

    a `value` gate - one whose answer is supplied rather than selected, like a
    token source or an output path - passes through unchanged. which gates those
    are is DECLARED in the contract's `answer_kind`, never decided here: an
    exemption the resolver grants itself when a match fails is indistinguishable
    from broken validation.
    """
    if value is None:
        return None
    v = str(value)
    if v in options:
        return v
    if answer_kind == "value":
        return v
    hits = [o for o in options if v == o or v in o.split("_")]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(
        f"{gate_id}: declared answer {v!r} is not one of the gate's options "
        f"{list(options)}"
        + (f" - it matches {hits}, which is ambiguous" if hits else "")
        + ". the option set is closed; a declaration cannot widen it.")


def preflight(cohort: dict, encoder: str, *, unit_counts: dict[str, int] | None = None,
              min_scope: int | None = None, device: str | None = None,
              answers: dict[str, str] | None = None) -> list[GateRequest]:
    """every preflight gate, resolved as far as the declarations allow.

    deterministic and side-effect free: same cohort and same counts give the same
    list. it reads the inventory's counts rather than re-measuring them - the
    platform_floor gate's own contract entry says it re-asks nothing.

    `answers` are decisions already taken ON THIS RUN - a person answered the
    interrupt, and the answer is in the record. they resolve a gate the same way
    a declaration does, and for the same reason: the gate asks whether a decision
    exists, not whether it was made recently. without this, re-entering the graph
    to finish a run whose worker died reopens every gate and stops at the
    interrupt again, so a run could be answered and still never complete.

    a declaration wins where both exist. `source` says which resolved it, so
    "declared for every run" and "answered on this one" never read alike.

    `device` is what turns capacity from a declared value into a checked one. it
    is a parameter rather than a probe because probing it needs a tensor library,
    and this function is the one thing that must stay callable on a machine that
    could not run the encoder it is planning for. a cohort whose pool declares a
    device supplies it without being asked.
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
                                 tuple(g["options"]), g["id"],
                                 g.get("answer_kind", "choice"))
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
            # "not_run" and "ran and failed" are different states and the person
            # answering needs to know which. nothing runs the probe yet, so it is
            # always the former here - said plainly rather than implied by a
            # question that asserts a result.
            observed = {"encoder": encoder,
                        "trained_on": es.trained_on if es else None,
                        "probe": "not_run",
                        "probe_state": "no probe has been run on this cohort - this gate "
                                       "is open because fit is UNESTABLISHED, not because "
                                       "a measurement came back below chance"}

        elif g["id"] == "capacity":
            # the one gate whose declared answer can be CONTRADICTED by a
            # measurement. a declared output_dir says where to write; it does not
            # say the volume can hold what is about to be written. when the plan
            # says it cannot, the declaration stops being sufficient and the gate
            # re-opens - the same rule the inventory's geometry step follows when
            # a measured lattice disagrees with a declared pitch.
            # ONLY on does_not_fit. an absent measurement is not a contradicted
            # declaration: "we never checked" and "we checked and it will not
            # fit" are different states, and pausing a cohort for the first would
            # stop every unattended run that declares no device. the record still
            # carries the why_not, so unchecked never reads as checked.
            plan = capacity_plan(cohort, encoder, unit_counts=unit_counts, device=device)
            observed = plan
            if plan.get("verdict") == "does_not_fit" and answer is not None:
                observed = plan | {
                    "declaration_overridden": answer,
                    "why": (f"{g.get('pre_answerable_by')} answers where to write, not "
                            f"whether it fits. "
                            f"the plan says {plan['verdict']}, so this is re-asked")}
                answer = None

        source = g.get("pre_answerable_by") if answer is not None else None
        if answer is None and g["id"] in (answers or {}):
            # the run's own answer faces the SAME resolver a declaration faces.
            # membership in `options` is the wrong test on its own: a `value`
            # gate's answer is supplied rather than selected, so a token source
            # or an output path is never in the list.
            try:
                answer = _resolve_option(answers[g["id"]], tuple(g["options"]),
                                         g["id"], g.get("answer_kind", "choice"))
            except ValueError:
                answer = None  # not one of the options, so not an answer
            else:
                source = "answered on this run" if answer is not None else None

        out.append(GateRequest(
            id=g["id"], when=g["when"], question=g["question"],
            options=tuple(g["options"]), forecloses=g["forecloses"],
            pre_answerable_by=g.get("pre_answerable_by"),
            answer=answer, source=source,
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


# --- the compute path -------------------------------------------------------
@dataclass(frozen=True)
class HeGeometry:
    """the declared tile geometry for one platform, lifted out of platform.json.

    every field is read from a declaration rather than computed here. `tile_px`
    is the integer that sized the published crops; the micron figure it works out
    to is derived and deliberately absent, because hashing a derived value would
    move the run identity whenever someone re-measures the lattice.
    """
    scale: float
    tile_px: int
    out_px: int
    k: int = 6
    batch_size: int = 32

    @classmethod
    def from_platform(cls, platform: dict, name: str, k: int = 6) -> HeGeometry:
        p = platform["platforms"][name]
        return cls(scale=float(p["image_pyramid"]["scale_to_hd"]),
                   tile_px=int(p["he_tile"]["tile_px_hd"]),
                   out_px=int(p["he_tile"]["resize_to"]), k=k)


@dataclass(frozen=True)
class StGraph:
    """the declared spatial graph for one platform, lifted out of platform.json.

    the ST counterpart of `HeGeometry`, and read the same way: from a
    declaration, never computed here. `scale_to_microns` is the field that makes
    this more than bookkeeping - Novae feeds every edge length to the GAT scaled
    by it, so it moves the embedding and belongs in the record next to
    `tile_px`. there is no default. a platform that does not declare it is
    refused, because the only defaults on offer are Novae's 1.0 (pixels read as
    microns) or a value guessed from a pitch, and neither is what any cache was
    built with.
    """
    method: str
    radius_cap_px: float | None
    scale_to_microns: float

    @classmethod
    def from_platform(cls, platform: dict, name: str) -> StGraph:
        g = platform["platforms"][name].get("st_graph") or {}
        missing = [f for f in ("method", "scale_to_microns") if g.get(f) is None]
        if missing:
            raise ValueError(
                f"platform {name!r} st_graph declares no {', '.join(missing)}. "
                "scale_to_microns is an input to the ST encoder's edge features, so it "
                "cannot default: declare the value the cache was built with, or the "
                "one this run intends as its own arm.")
        cap = g.get("radius_cap_px")
        return cls(method=str(g["method"]),
                   radius_cap_px=None if cap is None else float(cap),
                   scale_to_microns=float(g["scale_to_microns"]))

    def params(self) -> dict:
        """the fields a record carries, so two runs at different scales differ."""
        return {"st_graph": self.method, "radius_cap_px": self.radius_cap_px,
                "scale_to_microns": self.scale_to_microns}


# --- what a run will cost ---------------------------------------------------
def machine(at: str | Path | None = None) -> dict:
    """what THIS machine has, in the POOL's vocabulary. measured, and the parts
    that cannot be are None.

    the field names are `data_contract.json#pool.fields_in_1_2`, not names of
    this function's choosing. a declared pool and a measured one have to be
    comparable without a translation step, because a translation step is where a
    declared `memory_gb_per_task` quietly stops being checked.

    a measured machine is a pool of one: nothing here launches two shards at
    once, so per-task is the whole machine and `max_concurrent_tasks` says so.
    a walltime limit is a property of a scheduler and cannot be measured, so it
    is None rather than absent - the check is skipped, not silently passed.
    """
    import shutil

    out: dict = {"cpus_per_task": os.cpu_count(), "max_concurrent_tasks": 1,
                 "walltime_limit_s": None}
    try:
        page, pages = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_PHYS_PAGES")
        out["memory_gb_per_task"] = round(page * pages / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        out["memory_gb_per_task"] = None
    # the free space that matters is the free space on the volume being WRITTEN
    # to, which is the declared output location and not necessarily the one this
    # process happens to be running from.
    where = Path(at) if at else Path.cwd()
    while not where.exists() and where != where.parent:
        where = where.parent          # the output dir may not exist yet; its volume does
    try:
        out["free_disk_gb"] = round(shutil.disk_usage(where).free / 1e9, 1)
        out["measured_at"] = str(where)
    except OSError:
        out["free_disk_gb"] = None
    return out


# the pool fields this package knows how to check, read from the data contract
# rather than repeated here: a field added to the contract and not to a gate is
# a field that looks declared and is never compared against anything.
def _pool_fields() -> set[str]:
    from omicstra.settings import settings

    data = json.loads((settings.resolve(settings.configs_dir) / "data_contract.json").read_text())
    return set(data.get("pool", {}).get("fields_in_1_2") or {})


def pool_resources(cohort: dict) -> tuple[dict | None, str]:
    """the cohort's declared pool as resources, plus how it was obtained.

    1.1 let a cohort declare `pool` as a bare NAME - "mac" - which says where the
    work runs and nothing about what it has. that is still valid and still means
    the local machine must be measured; what it must not do is look like a
    declaration that was checked. hence the note, which travels into the plan.

    an unknown field is refused rather than ignored. ignoring it is the failure
    this is here to prevent: a pool declaring a resource the checker does not
    read is a pool whose limit is never enforced, and it fails at the far end of
    a six-hour run instead of before it starts.
    """
    pool = cohort.get("pool")
    if pool is None:
        return None, "no pool declared; this machine measured instead"
    if isinstance(pool, str):
        return None, f"pool {pool!r} is declared by name only; this machine measured instead"
    if not isinstance(pool, dict):
        raise ComputeRefused(
            f"pool must be a name or an object of declared resources, got {type(pool).__name__}")

    known = _pool_fields() | {"id"}
    unknown = sorted(set(pool) - known)
    if unknown:
        raise ComputeRefused(
            f"pool declares {unknown}, which no gate reads. the fields a pool may carry are "
            f"{sorted(known)} (data_contract.json#pool.fields_in_1_2). a resource nothing "
            "compares against is a limit that is not enforced, so it is refused here rather "
            "than at the end of the run it should have stopped")
    return dict(pool), f"pool {pool.get('id', '<unnamed>')!r} declared with resources"


def capacity_plan(cohort: dict, encoder: str, *, unit_counts: dict[str, int] | None = None,
                  device: str | None = None, at: str | Path | None = None) -> dict:
    """what this cohort's run will cost, or why that cannot be said yet.

    the bridge between the two halves that already existed and never met: the
    inventory counted the units, the encoder carries seconds and bytes per unit,
    and nothing multiplied them. every input here is something the cohort has
    already declared or already measured - this invents no number of its own.

    it never raises on a MISSING input. "cannot be said yet" is a legitimate
    state before inventory has run, and a planner that throws cannot be called
    from a gate that is trying to describe the situation. a malformed pool is a
    different thing - that is a wrong declaration rather than an absent one, and
    it is refused where it is read.
    """
    n_units = sum((unit_counts or {}).values())
    resources, pool_note = pool_resources(cohort)
    dev = device or (resources or {}).get("device")
    # `at` is where the shards actually land, which is not always what the
    # cohort declares: a run writing to runs/ must have runs/'s volume checked,
    # not the declared cache's.
    out_dir = str(at) if at else cohort.get("output_dir")
    base = {"n_units": n_units, "pool_note": pool_note, "output_dir": out_dir}

    if not n_units:
        return base | {"estimated": False, "verdict": "unknown",
                       "why_not": ("no unit counts. the inventory produces them and this "
                                   "reads them; it does not re-measure the cohort")}
    if not dev:
        return base | {"estimated": False, "verdict": "unknown",
                       "why_not": ("no device. declare one on the pool "
                                   "(data_contract.json#pool.fields_in_1_2.device) or pass "
                                   "it in - it is not probed here, because planning must "
                                   "work on a machine that cannot run the encoder")}
    have = resources or machine(at=out_dir)
    return base | estimate(encoder, n_units, device=dev, pool=have)


def estimate(encoder: str, n_units: int, device: str | None = None,
             pool: dict | None = None) -> dict:
    """how long, how much disk, how much memory - or why it cannot be said.

    an estimate refuses rather than guesses. the encoder carries measurements
    per DEVICE because the same model is a different job on a laptop and on a
    card, and a run planned against an invented number is worse off than one
    planned against none: peak memory is what has actually stopped a run here.

    `pool` is what the machine offers. when it is not supplied the local one is
    measured, which is right for a laptop and wrong for a scheduler - so a
    scheduler declares its own and the verdict says which was used.
    """
    from omicstra.models.encoders import spec

    dev = device or str(pick_device()).replace("device(type=", "").strip("')")
    es = spec(encoder)
    c = es.cost_on(dev)
    have = pool or machine()
    out = {"encoder": encoder, "unit": es.unit, "n_units": int(n_units),
           "device": dev, "pool": "declared" if pool else "measured on this machine",
           "have": have}
    if c is None:
        return out | {"estimated": False,
                      "why_not": (f"no measurement for {encoder} on {dev}. measured devices: "
                                  f"{[x.device for x in es.cost] or 'none'}. run a few units and "
                                  "record what they cost rather than estimating from another "
                                  "device"),
                      "verdict": "unknown"}

    seconds = n_units * c.seconds_per_unit
    hours = seconds / 3600
    out_gb = n_units * c.bytes_per_unit / 1e9
    checks = []
    if have.get("free_disk_gb") is not None:
        checks.append({"resource": "disk", "need_gb": round(out_gb, 2),
                       "have_gb": have["free_disk_gb"],
                       "fits": out_gb < have["free_disk_gb"] * 0.9})
    if have.get("memory_gb_per_task") is not None:
        checks.append({"resource": "memory", "need_gb": c.peak_memory_gb,
                       "have_gb": have["memory_gb_per_task"],
                       "fits": c.peak_memory_gb < have["memory_gb_per_task"] * 0.8})
    if have.get("walltime_limit_s"):
        checks.append({"resource": "walltime", "need_gb": None,
                       "need_hours": round(hours, 2),
                       "have_hours": round(have["walltime_limit_s"] / 3600, 2),
                       "fits": hours < have["walltime_limit_s"] / 3600})
    # a resource with no declared or measured value is NOT a resource that
    # passed. saying so is the difference between "checked and fine" and "never
    # looked at", which is the whole reason a declared pool is refused when it
    # carries a field nothing reads.
    #
    # walltime is deliberately not in this set. an absent disk figure means
    # nobody looked; an absent walltime means there is no limit, which is a pass
    # rather than a gap - a laptop has none and is not thereby under-checked.
    not_checked = sorted({"disk", "memory"} - {c_["resource"] for c_ in checks})
    blocked = [c_["resource"] for c_ in checks if not c_["fits"]]
    return out | {
        "estimated": True,
        # seconds is the exact figure; hours is the one a person reads. a short
        # run rounded to two decimals reads "0.0 hours", which looks like a bug
        # in the planner rather than a cheap job.
        "seconds": round(seconds, 1),
        "hours": round(hours, 2) if hours >= 0.1 else round(hours, 4),
        "output_gb": round(out_gb, 2),
        "peak_memory_gb": c.peak_memory_gb,
        "measured_on": c.measured_on,
        "checks": checks,
        "not_checked": not_checked,
        "verdict": "fits" if not blocked else "does_not_fit",
        "blocked_by": blocked,
        "advice": ((("run it here" if not not_checked else
                     f"run it here; {', '.join(not_checked)} was never checked because the "
                     "pool declares no value for it")) if not blocked else
                   f"{', '.join(blocked)} is short on this machine - run it somewhere with "
                   "more, or reduce the scope"),
    }


def pick_device(declared: str | None = None):
    """mps where available, as the published grid ran. cpu otherwise.

    NOT a silent choice: the device lands in the record, because float32
    accumulation on mps and on cpu are not bit-identical and a slice-diff that
    misses by 1e-6 should point at the device before it points at the port.
    """
    import torch

    if declared:
        return torch.device(declared)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_one_he(coords, image_path, geom: HeGeometry, encoder: str = "virchow2",
               device: str | None = None, model=None):
    """one subarray of coordinates -> (n, dim) niche vectors, plus a record.

    the whole H&E arm for one sample, and nothing cohort-specific in it: the
    caller supplies coordinates, an image and a declared geometry. the ORDER is
    the published one and is not an implementation detail -

        tiles -> encode every tile -> pool over the niche

    encoding first and pooling second is what makes the niche a mean of encoded
    tiles rather than an encoding of a mean tile. those are different vectors and
    only one of them is in the cache.
    """
    import time

    import numpy as np
    from PIL import Image

    from omicstra.measures.tiling import TileGeometry, cut_tiles, niche_neighbours
    from omicstra.models.encoders import load, spec

    t0 = time.time()
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    dev = pick_device(device)
    es = spec(encoder)

    if model is None:
        model = load(encoder)
        if hasattr(model, "to"):
            model = model.to(dev)

    Image.MAX_IMAGE_PIXELS = None          # the HD slides are far over PIL's bomb limit
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        tiles = cut_tiles(im, coords,
                          TileGeometry(scale=geom.scale, tile_px=geom.tile_px,
                                       out_px=geom.out_px))
        image_size = im.size

    per_tile = model.embed_images(tiles, batch_size=geom.batch_size)
    idx, counts = niche_neighbours(coords, geom.k)
    niche = pool_niche_f32(per_tile, idx)

    rec = TransformRecord(
        step_id="encode_he",
        params={"encoder": encoder, "dim": es.dim, "k": geom.k,
                "tile_px": geom.tile_px, "out_px": geom.out_px,
                "scale": geom.scale, "batch_size": geom.batch_size,
                "device": str(dev), "image_size": list(image_size)},
        produced=[Produced(path=str(image_path), shape=list(niche.shape),
                           dtype=str(niche.dtype))],
        duration_s=round(time.time() - t0, 2),
    )
    rec.params["neighbour_counts"] = {int(c): int(n) for c, n in
                                      zip(*np.unique(counts, return_counts=True))}
    return niche, rec


def pool_niche_f32(per_tile, idx):
    """the published pooling, re-exported so the compute path has one import.

    thin on purpose: the maths lives in measures/ and is diffed against the
    script. a second implementation here is exactly the duplication the tiling
    clamp already taught us about.
    """
    from omicstra.measures.tiling import pool_niche

    return pool_niche(per_tile, idx)


# measured on TNBC1_CN1_C1, 2026-09-11, and the reason it is not zero:
#
#   cpu vs mps        8.97e-05      two backends, same code, same machine
#   mps vs cache      1.06e-04
#   cpu vs cache      1.05e-04
#
# our port sits the SAME distance from the cache as the two backends sit from
# each other, so nothing here is attributable to the port. bit-identity is
# available for align - deterministic linear algebra, and it hit 0.00e+00 on
# 13/13 metrics - and is not available for a 631M-parameter float32 transformer
# across torch and backend versions. that is the EXACT_RUNS / TOLERANCE_RUNS
# split v1_1_scope.md already declared; encode is a tolerance run.
#
# the tolerance is expressed against the ROW NORM (~32.7 here) rather than as a
# bare absolute, so it means the same thing on an encoder with a different scale.
SLICE_ATOL_ROW = 1e-5          # measured max is 3.2e-06, so ~3x headroom


def slice_diff(computed, cached_path, atol: float = 0.0):
    """our port against the cache, on real vectors.

    the acceptance test for encode is NOT a re-extraction. the cache is the
    oracle: it is what the published grid consumed, so reproducing it proves the
    port. re-extracting and comparing to a fresh run would only prove Virchow2 is
    deterministic, which is not in question.

    `atol=0.0` demands bit-identity. that is the right default for a port that
    changed no arithmetic, and for this encoder it does NOT hold - see the note
    above. use `accepts()` for the acceptance decision; this function reports the
    numbers and does not decide.
    """
    import numpy as np

    cached = np.load(cached_path)
    if computed.shape != cached.shape:
        return {"match": False, "why": "shape",
                "computed": list(computed.shape), "cached": list(cached.shape)}

    a = computed.astype(np.float64)
    b = cached.astype(np.float64)
    d = np.abs(a - b)

    # relative error per ELEMENT is meaningless here: a 1280-d embedding has
    # elements near zero, and dividing by one turns a 1e-7 absolute difference
    # into a relative blow-up that says nothing about the vector. scale by the
    # row norm instead, which is the quantity every downstream metric uses.
    row = np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * row[:, 0], 1e-12)

    return {"match": bool(d.max() <= atol), "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "max_rel_to_row_norm": float((d / row).max()),
            "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
            "n": int(computed.shape[0]), "dim": int(computed.shape[1]), "atol": atol}


def retrieval_invariant(computed, cached, k: int = 6):
    """does the difference move anything a published number depends on?

    the vectors feed cosine retrieval and linear probes, so element-wise equality
    is not the question the grid asks. the load-bearing one is the THIRD measure
    below: H1 is a retrieval metric, so "same neighbours" is the property the
    grid actually rests on, and it can hold exactly while the vectors differ.

      min_cosine    per-row agreement between our vector and the cached one
      knn_set       the top-k neighbours in embedding space, as a SET, identical
                    for what fraction of rows. set rather than sequence: two
                    neighbours a float apart can swap order without any metric
                    noticing, so ordered equality would fail on a difference that
                    changes nothing
      knn_ordered   reported beside it, never the criterion - it is the stricter
                    thing, and saying which is which is the point

    `self_retrieval_at_1` is deliberately GONE. it queried our vectors against
    the cached bank, which is a cross-device comparison by construction, and on
    an 8-spot subarray it reports 0.25 for a port that is correct.
    """
    import numpy as np

    a = np.asarray(computed, dtype=np.float32)
    b = np.asarray(cached, dtype=np.float32)
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)

    cos = (an * bn).sum(1)
    n = len(a)
    kk = min(k, n - 1) if n > 1 else 0
    if kk <= 0:
        return {"min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
                "knn_set_agreement": 1.0, "knn_ordered_agreement": 1.0,
                "k": 0, "n": n, "note": "single row - no neighbourhood to preserve"}

    def ranked(x):
        sim = x @ x.T
        np.fill_diagonal(sim, -np.inf)
        return sim, np.argsort(-sim, axis=1)[:, :kk]

    _, ours = ranked(an)
    sb, theirs = ranked(bn)

    # a TIE is not a disagreement. where the k-th and (k+1)-th neighbours have
    # the same similarity, which one lands inside the top-k is decided by argsort,
    # not by the data - and two implementations may split it differently while
    # encoding identical structure. small subarrays make this the normal case
    # rather than the exception: at 8 spots every niche overlaps every other, so
    # the pooled vectors are duplicates and the whole neighbourhood is tied.
    # counting that as a failure measures the sort, not the port.
    tie_eps = 1e-6
    same, tied_only = np.zeros(n, bool), np.zeros(n, bool)
    for j in range(n):
        u, v = set(ours[j].tolist()), set(theirs[j].tolist())
        if u == v:
            same[j] = True
            continue
        diff = list(u ^ v)
        boundary = sb[j][theirs[j][-1]]
        tied_only[j] = all(abs(float(sb[j][m]) - float(boundary)) <= tie_eps for m in diff)

    return {"min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
            "knn_set_agreement": float((same | tied_only).mean()),
            "knn_set_exact": float(same.mean()),
            "knn_resolved_by_tie": int(tied_only.sum()),
            "knn_ordered_agreement": float((ours == theirs).all(1).mean()),
            "k": kk, "n": n}


# the acceptance criterion, DECLARED - written down with its reason rather than
# chosen after seeing a diff. observed values are in the note above slice_diff.
ACCEPTANCE = {
    "max_abs_over_row_norm": 1e-5,      # observed 3.2e-06, ~3x headroom
    "min_cosine": 1 - 1e-6,             # observed 1 - 2e-07
    "knn_set_agreement": 0.999,         # the one H1 actually depends on
    "reason": (
        "float32 transformer with backend-dependent kernels; cpu and mps on this "
        "machine differ by the same order as either differs from the cache, so "
        "bit-identity is not available and closeness alone is not the question. "
        "k-NN preservation is, because H1 is a retrieval metric and 'same "
        "neighbours' can hold exactly while vectors do not."),
    "cross_device_footnote": (
        "the cache does not record which device built it - the extraction script "
        "never wrote one. manifests from now on carry device, torch, timm and the "
        "model revision so this footnote stops being needed for anything built "
        "after 2026-09-11."),
    "cache_device_inferred": (
        "INFERRED, not declared. on TNBC1_CN1_C1 the top-6 neighbourhood agreement "
        "is cpu-vs-cache 0.9991, cpu-vs-mps 0.9991, mps-vs-cache 0.9981 - our CPU "
        "run sits as close to the cache as it sits to our own MPS run, and closer "
        "than MPS does. the cache therefore behaves like a CPU build. so the "
        "comparison is run on CPU: the criterion below was never too tight, the "
        "BACKEND was mismatched, and relaxing a threshold because a mismatched "
        "backend missed it would have buried that."),
}


def accepts(diff: dict, inv: dict) -> tuple[bool, str]:
    """the acceptance decision for an encode port, in one place.

    three conditions asking three different questions: is the drift bounded, do
    the vectors still agree, and did the drift move the neighbourhood. a port can
    pass the first two and fail the third, and only the third would have changed
    a published number.
    """
    checks = [
        ("drift", diff["max_rel_to_row_norm"] <= ACCEPTANCE["max_abs_over_row_norm"],
         f"{diff['max_rel_to_row_norm']:.2e} vs {ACCEPTANCE['max_abs_over_row_norm']:g}"),
        ("cosine", inv["min_cosine"] >= ACCEPTANCE["min_cosine"],
         f"{inv['min_cosine']:.7f} vs {ACCEPTANCE['min_cosine']:.7f}"),
        ("knn", inv["knn_set_agreement"] >= ACCEPTANCE["knn_set_agreement"],
         (f"top-{inv['k']} set {inv['knn_set_agreement']:.4f} vs "
          f"{ACCEPTANCE['knn_set_agreement']}")),
    ]
    failed = [f"{n} {d}" for n, ok, d in checks if not ok]
    if failed:
        return False, "; ".join(failed)
    return True, "; ".join(f"{n} {d}" for n, _, d in checks)


def encoder_provenance(device) -> dict:
    """what the cache should have recorded and did not.

    the whole reason the comparison above is labelled cross-device is that the
    extraction script wrote no device and no versions. this is the fix going
    forward, and it costs four lines.
    """
    import timm
    import torch

    return {"device": str(device), "torch": torch.__version__, "timm": timm.__version__,
            "model_revision": "hf-hub:paige-ai/Virchow2", "dtype": "float32"}


# --- the cohort run: N shards of run_one_he ---------------------------------
def he_shards(samples, out_dir):
    """one shard per subarray, named by the file that proves it finished.

    the output is the `.npy` the next stage reads, which is what makes resume
    free: `dispatch` asks the filesystem, not a ledger, so a run killed at
    sample 200 restarts at 200 whether it was killed by a scheduler, a laptop
    lid, or a SIGKILL.
    """
    from pathlib import Path

    from omicstra.dispatch import Shard

    out = Path(out_dir)
    return [Shard(id=sid, output=out / f"{sid}.npy",
                  params={"image": str(image), "coords": str(coords)})
            for sid, image, coords in samples]


def encode_he_cohort(samples, out_dir, geom: HeGeometry, *, encoder: str = "virchow2",
                     device: str | None = None, address: str | None = None,
                     max_attempts: int = 3, on_event=None):
    """embed every subarray, durably if an address is configured.

    the model is loaded ONCE per process and closed over, not per shard: a 631M
    parameter load is ~20 s and doing it 280 times is an hour of nothing. on the
    durable path the worker process holds it for the same reason, which is why
    the activity body is registered rather than serialised.
    """
    import numpy as np
    import pandas as pd

    from omicstra.dispatch import atomic_write, run_shards
    from omicstra.models.encoders import load

    dev = pick_device(device)
    model = load(encoder)
    if hasattr(model, "to"):
        model = model.to(dev)

    def one(shard, heartbeat):
        heartbeat(shard.id)
        coords = pd.read_parquet(shard.params["coords"])[["x", "y"]].to_numpy()
        vecs, _ = run_one_he(coords, shard.params["image"], geom,
                             encoder=encoder, device=str(dev), model=model)
        heartbeat(f"{shard.id} encoded")
        # a HANDLE, not a path. the temp name now keeps the .npy suffix so
        # np.save would leave it alone anyway, but writing through a handle means
        # this does not silently depend on that.
        def save(p):
            with p.open("wb") as fh:
                np.save(fh, vecs)

        atomic_write(shard.output, save)

    return run_shards(he_shards(samples, out_dir), one, address=address,
                      max_attempts=max_attempts, on_event=on_event)
