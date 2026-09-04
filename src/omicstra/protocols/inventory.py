"""the inventory chain - "what is this data".

eight questions, answered before anything asks whether the data is USABLE. that
is the eda chain's job, and it reads ROLES from here, never column names.

nothing here judges. inventory describes; the gate decides. so these emit
DiagnosticRecords - produced artifacts, no verdict - and each returns
`(record, ctx_updates)` so a later step reads what an earlier one found instead
of reopening the same file.

two things this chain exists to catch, both of which look fine until they are
measured:

  platform cannot be inferred. HEST stores Xenium with `array_col`/`array_row`/
  `in_tissue` - a spot schema - so column-sniffing reports a cell platform as a
  spot one, confidently. platform.json is the authority and must name every
  sample; an undeclared sample halts.

  a per-sample-unique key is not a cohort-unique key. original ST obs_names are
  `2x6`, `2x8` - identical in all 260 samples. the key is COMPOSITE.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from omicstra.protocols import Step
from omicstra.records import DiagnosticRecord

Ctx = dict

# module-level object cache. NOT in ctx: the tracer serialises every link's
# input, so anndata in ctx means reprs - and on a restricted cohort, values -
# leaving the machine. ctx carries paths; this holds what was opened.
_CACHE: dict[str, object] = {}


def _root(ctx: Ctx) -> Path:
    return Path(ctx["project_dir"]).expanduser()


def _sid(p: Path) -> str:
    """a sample id must be COHORT-unique.

    `p.parent.name` gives `C1`, `C2` on this layout - the subarray letter, which
    repeats under every slide. 281 files collapse to 6 ids and the whole
    per-sample story silently becomes a per-letter one.
    """
    if p.name == "selection.RData":
        return f"{p.parent.parent.name}_{p.parent.name}"      # CN1_C1
    return p.stem


def _read(p: Path):
    import anndata as ad
    if p.suffix == ".h5ad":
        return ad.read_h5ad(p)
    import pyreadr
    return pyreadr.read_r(str(p))


def _rec(step_id, status, result, decision, observed=None):
    """inventory emits DiagnosticRecord, not TransformRecord.

    these steps MEASURE and describe - method, observed, result, decision - and
    TransformRecord has none of those slots (it carries `produced`/`outputs` for
    artifacts). passing them there dropped every field silently, which is its own
    small lesson about writing to a schema you have not checked.
    """
    return DiagnosticRecord(step_id=step_id, status=status, result=result,
                            decision=decision, observed=observed or {})


def _index(o) -> list[str]:
    if hasattr(o, "obs_names"):
        return [str(i) for i in o.obs_names]
    return [str(i) for i in next(iter(o.values())).index]


def _matrix(o):
    X = o.X if hasattr(o, "X") else next(iter(o.values())).to_numpy()
    x = X[:200]
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


# -------------------------------------------------------------- 1 · files ---
def files(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    # where inputs live is DECLARED, not assumed. the project-home shape puts
    # them at <project>/data/inputs, but a cohort may point elsewhere - tnbc-92
    # is the in-repo fixture and its data sits at the package root.
    rel = ctx["params"].get("inputs_dir", "data/inputs")
    root = (_root(ctx) / rel).resolve()
    if not root.exists():
        return _rec("files", "fail", f"no data/inputs under {root.parent}",
                    "place the cohort's raw data there, or fix project_dir"), {}
    paths = sorted(root.rglob("*.h5ad")) or sorted(root.rglob("selection.RData"))
    ext = Counter(p.suffix for p in root.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in paths)
    return _rec("files", "pass" if paths else "fail",
                f"{len(paths)} sample files, {total/1e9:.2f} GB",
                "proceed" if paths else "no per-sample data found",
                {"n": len(paths), "bytes": total, "formats": dict(ext.most_common(6)),
                 "examples": [_sid(p) for p in paths[:4]]}), \
           {"files": [str(p) for p in paths], "sample_ids": [_sid(p) for p in paths]}


# ----------------------------------------------------------- 2 · platform ---
def platform(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    p = _root(ctx) / "platform.json"
    if not p.exists():
        return _rec("platform", "fail", "no platform.json",
                    "declare it. the platform is not inferable from the object - "
                    "HEST stores Xenium in a spot schema, so sniffing columns "
                    "reports a cell platform as a spot one, confidently"), {}
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    d = json.loads(raw)
    by_sample, defs = d.get("samples", {}), d.get("platforms", {})
    missing = [s for s in ctx["sample_ids"] if s not in by_sample]
    undefined = sorted(set(by_sample.values()) - set(defs))
    if missing or undefined:
        return _rec("platform", "fail",
                    f"{len(missing)} sample(s) undeclared, {len(undefined)} platform(s) undefined",
                    "every sample names a platform and every platform is defined. "
                    "an undeclared sample is a halt, not a guess",
                    {"undeclared": missing[:8], "undefined": undefined, "sha": sha}), {}
    counts = Counter(by_sample[s] for s in ctx["sample_ids"])
    return _rec("platform", "pass", f"{len(counts)} platform(s): {dict(counts)}",
                "heterogeneous - applicability resolves per sample"
                if len(counts) > 1 else "homogeneous",
                {"counts": dict(counts), "sha": sha,
                 "defs": {k: {"resolution": v.get("resolution_class"),
                              "pitch_um": v.get("spot_pitch_um"),
                              "scope": v.get("transcriptome_scope")} for k, v in defs.items()}}), \
           {"platform_by_sample": by_sample, "platform_defs": defs, "platform_decl_sha": sha}


# -------------------------------------------------------------- 3 · shape ---
def shape(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """reads each file ONCE and caches the object. `_objs` stays out of records."""
    paths = [Path(p) for p in ctx["files"]][: ctx["params"].get("max_samples", 8)]
    rows = []
    for p in paths:
        sid = _sid(p)
        try:
            o = _read(p)
            n_obs, n_var = (o.shape if hasattr(o, "shape") else next(iter(o.values())).shape)
            _CACHE[str(p)] = o
            rows.append({"sample": sid, "n_obs": int(n_obs), "n_var": int(n_var),
                         "platform": ctx["platform_by_sample"].get(sid)})
        except Exception as e:
            rows.append({"sample": sid, "error": f"{type(e).__name__}: {e}"})
    ok = [r for r in rows if "n_obs" in r]
    return _rec("shape", "pass" if ok else "fail", f"{len(ok)}/{len(paths)} readable",
                "proceed" if ok else "no sample could be read",
                {"per_sample": rows[:8],
                 "n_obs": [min(r["n_obs"] for r in ok), max(r["n_obs"] for r in ok)] if ok else None,
                 "n_var": [min(r["n_var"] for r in ok), max(r["n_var"] for r in ok)] if ok else None}), \
           {"paths_by_sid": {_sid(p): str(p) for p in paths}}


# ------------------------------------------------------------- 4 · coords ---
def coords(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """a DECLARED column that is absent is a fail - the declaration is wrong."""
    rows = []
    for sid, _p in ctx["paths_by_sid"].items():
        o = _CACHE[_p]
        pk = ctx["platform_by_sample"].get(sid)
        declared = (ctx["platform_defs"].get(pk) or {}).get("position_columns") or []
        if hasattr(o, "obsm"):
            has = "spatial" in o.obsm
            absent = [c for c in declared if c not in o.obs.columns]
            arr = np.asarray(o.obsm["spatial"]) if has else None
            rows.append({"sample": sid, "platform": pk, "obsm_spatial": has,
                         "declared": declared, "absent": absent,
                         "ok": bool(has and not absent),
                         "pixel_extent": [float(np.ptp(arr[:, 0])), float(np.ptp(arr[:, 1]))] if has else None})
        else:
            # a dict of frames: the declared columns live inside one of them,
            # not among the object names. checking the keys passes vacuously.
            objs = list(o.keys())
            cols = {f: list(v.columns) for f, v in o.items() if hasattr(v, "columns")}
            holder = next((f for f, cs in cols.items()
                           if declared and all(c in cs for c in declared)), None)
            absent = [] if holder else [c for c in declared
                                        if not any(c in cs for cs in cols.values())]
            rows.append({"sample": sid, "platform": pk, "objects": objs,
                         "declared": declared, "found_in": holder, "absent": absent,
                         "ok": bool(declared and holder)})
    bad = [r["sample"] for r in rows if not r["ok"]]
    return _rec("coords", "fail" if bad else "pass",
                f"{len(rows)-len(bad)}/{len(rows)} carry declared position",
                "the declaration is wrong for these samples" if bad else
                "a pixel extent is not a micron extent - scale comes from "
                "platform.spot_pitch_um, never from these numbers",
                {"per_sample": rows, "failing": bad}), {}


# ----------------------------------------------------------- 5 · join_key ---
def join_key(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """per-sample-unique is NOT cohort-unique. checked on (sample_id, index)."""
    rows, composite = [], set()
    collide = False
    for sid, _p in ctx["paths_by_sid"].items():
        o = _CACHE[_p]
        idx = _index(o)
        rows.append({"sample": sid, "n": len(idx),
                     "unique_within_sample": len(set(idx)) == len(idx),
                     "examples": idx[:2]})
        before = len(composite)
        composite |= {(sid, i) for i in idx}
        if len(composite) - before != len(set(idx)):
            collide = True
    within = [r["sample"] for r in rows if not r["unique_within_sample"]]
    total = sum(r["n"] for r in rows)
    return _rec("join_key", "fail" if (within or collide) else "pass",
                f"{len(rows)} samples, {len(composite)} distinct (sample_id, index) of {total}",
                "a non-unique key cannot join expression to image" if within or collide
                else "composite key is cohort-unique",
                {"per_sample": rows, "duplicated_within": within,
                 "image_side": "unchecked - the image index has not been compared "
                               "against this system"}), \
           {"join_key": {"system": "obs_names", "composite": ["sample_id", "obs_names"],
                         "scope": "cohort", "image_side": "unchecked"}}


# ------------------------------------------------------------- 6 · counts ---
def counts(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """TESTED per sample. an all-zero slice must not pass trivially."""
    rows = []
    for sid, _p in ctx["paths_by_sid"].items():
        o = _CACHE[_p]
        x = _matrix(o)
        nonzero = float(x.max()) > 0
        p = Path(ctx["paths_by_sid"][sid])
        rows.append({"sample": sid, "dtype": str(x.dtype), "max": float(x.max()),
                     "nonzero_slice": nonzero,
                     "integer_valued": bool(np.allclose(x, np.round(x))) and nonzero,
                     "sha256_1MB": hashlib.sha256(p.read_bytes()[:1_000_000]).hexdigest()[:16]})
    bad = [r["sample"] for r in rows if not r["integer_valued"]]
    return _rec("counts", "fail" if bad else "pass",
                f"{len(rows)-len(bad)}/{len(rows)} integer-valued",
                "find the raw matrix - a filename is not evidence of rawness"
                if bad else "raw integer counts",
                {"per_sample": rows, "not_integer": bad}), {"counts_raw": not bad}


# ------------------------------------------------------------- 7 · funnel ---
def funnel(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    declared = ctx["params"].get("declared_n")
    found = len(ctx["files"])
    delta = (found - declared) if declared else 0
    return _rec("funnel", "pass" if not delta else "fail",
                f"{found} present" + (f", {abs(delta)} {'extra' if delta>0 else 'missing'} vs {declared} declared" if delta else ""),
                "more files than the cohort claims is a finding too - an itemised "
                "funnel is the difference between a cohort and a convenience sample",
                {"declared": declared, "found": found, "delta": delta}), {}


# --------------------------------------------------------------- 8 · bind ---
def bind(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """conformance. binds concrete artifacts to the contract's abstract ROLES.

    this is the record eda reads. it is also the record that changes when the
    contract changes, which is what makes contract drift detectable.
    """
    from omicstra.settings import settings
    roles = json.loads((settings.resolve(settings.configs_dir) / "data_contract.json"
                        ).read_text())["roles"]["definitions"]
    recs = ctx["_records"]

    def ok(step): return recs.get(step, {}).get("status") == "pass"

    # start from the CONTRACT, not a literal. a role the contract declares and
    # this chain never mentions would otherwise pass by absence.
    bound = {r: {"bound": False, "why": "declared in the contract, not bound by this chain"}
             for r in roles}
    cohort = _root(ctx) / "cohort.json"
    decl = json.loads(cohort.read_text()) if cohort.exists() else {}
    subject_col = decl.get("subject_id_column")

    bound["key"]        = {"bound": ok("join_key"), "system": ctx.get("join_key")}
    bound["molecular"]  = {"bound": bool(ctx.get("counts_raw")), "source": "X", "tested": True}
    bound["platform"]   = {"bound": ok("platform"), "decl_sha": ctx.get("platform_decl_sha")}
    bound["spatial"]    = {"bound": ok("coords"), "scale_from": "platform.spot_pitch_um"}
    bound["confounder_axis"] = (
        {"bound": True, "column": subject_col, "source": "cohort.json"} if subject_col
        else {"bound": False, "why": "no cohort.json subject_id_column - declared, "
                                     "never guessed, same rule as platform"})
    unbound = [r for r, spec in roles.items()
               if spec.get("required") and not bound.get(r, {}).get("bound")]
    return _rec("bind", "fail" if unbound else "pass",
                f"{sum(1 for b in bound.values() if b['bound'])}/{len(roles)} roles bound"
                + (f"; REQUIRED unbound: {unbound}" if unbound else ""),
                "a protocol reading an unbound required role would be inventing data"
                if unbound else "conforms - eda may read roles, not columns",
                {"roles": bound, "unbound_required": unbound}), {"roles": bound}


INVENTORY_STEPS: list[Step] = [
    Step(id="files",    fn=files,    produces=frozenset({"files"})),
    Step(id="platform", fn=platform, produces=frozenset({"platform"}), requires=frozenset({"files"})),
    Step(id="shape",    fn=shape,    produces=frozenset({"shape"}),    requires=frozenset({"files", "platform"})),
    Step(id="coords",   fn=coords,   produces=frozenset({"coords"}),   requires=frozenset({"shape"})),
    Step(id="join_key", fn=join_key, produces=frozenset({"key"}),      requires=frozenset({"shape"})),
    Step(id="counts",   fn=counts,   produces=frozenset({"molecular"}), requires=frozenset({"shape"})),
    Step(id="funnel",   fn=funnel,   produces=frozenset({"funnel"}),   requires=frozenset({"files"})),
    Step(id="bind",     fn=bind,     produces=frozenset({"roles"}),
         requires=frozenset({"files", "platform", "shape", "coords", "key", "molecular", "funnel"})),
]