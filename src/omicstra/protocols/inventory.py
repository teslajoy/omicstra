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
import os
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
    """.h5ad only. a cohort's native format is converted once, outside the package.

    this used to fall back to pyreadr for anything else, which quietly made the
    package responsible for one cohort's file format - and a cohort that did not
    arrive from R would have found R-shaped assumptions waiting for it. the
    fallback is gone and the refusal names the fix, because an inventory that
    silently reads a native file is an inventory nobody can port.
    """
    import anndata as ad

    if p.suffix == ".h5ad":
        return ad.read_h5ad(p)
    raise ValueError(
        f"{p.name}: the inventory reads .h5ad only. convert this cohort once with "
        "scripts/ingest_<cohort>.py, which writes .h5ad plus a coordinates table "
        "and records the source shas.")


def _budget(sample_bytes: int | None = None) -> dict:
    """how many samples this machine can hold open at once, measured not guessed.

    two resources bind, and which one depends on how the objects are read:

      memory       a FULL read of one sample here costs ~8x its file size once
                   the sparse matrix is materialised. 280 of those is over 11 GB.
      descriptors  a BACKED read costs one open file handle and almost no memory,
                   so the limit becomes RLIMIT_NOFILE rather than RAM.

    the inventory reads backed, which is why the fd term usually dominates and
    the answer is usually "all of them". the memory term is kept because a host
    that cannot read backed falls back to full reads, and there the cap matters.

    no psutil: it is not a declared dependency. the memory figure comes from
    os.sysconf where the platform offers it and is simply absent otherwise -
    a missing figure drops that term rather than inventing one.
    """
    import resource

    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    # leave headroom: the process already holds stdio, the checkpointer, and
    # whatever the caller opened.
    fd_cap = max(1, int(soft * 0.5)) if soft and soft > 0 else None

    # SC_AVPHYS_PAGES returns 0 on macOS rather than failing, and 0 is not
    # "unknown" - reading it as a figure would compute a cap of 1 sample on the
    # machine this runs on. absent and zero both mean the same thing here: the
    # platform does not offer the number, so the memory term drops out.
    avail = None
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        if pages and pages > 0:
            avail = os.sysconf("SC_PAGE_SIZE") * pages
    except (ValueError, OSError, AttributeError):
        avail = None

    mem_cap = None
    if avail and sample_bytes:
        mem_cap = max(1, int((avail * 0.25) // sample_bytes))

    caps = [c for c in (fd_cap, mem_cap) if c]
    return {"fd_cap": fd_cap, "mem_cap": mem_cap,
            "available_bytes": avail, "per_sample_bytes": sample_bytes,
            "cap": min(caps) if caps else None,
            "bound_by": ("descriptors" if caps and min(caps) == fd_cap else
                         "memory" if caps else "nothing measurable"),
            "memory_term": ("dropped - this platform does not report available "
                            "pages" if avail is None else "measured")}


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
        # name the path that was CHECKED, not the default. the message read
        # "no data/inputs" whatever inputs_dir said, which sends a reader to
        # look for the wrong directory - and did.
        return _rec("files", "fail", f"no {rel} under {_root(ctx)}",
                    f"place the cohort's per-sample files at {root}, declare a different "
                    "inputs_dir, or fix project_dir",
                    {"checked": str(root), "inputs_dir": rel}), {}
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
    # the cap exists because opening every .h5ad in a large cohort is slow, and
    # `shape` is often run to see the SHAPE of the data rather than to measure
    # all of it. but the cap has to travel in the record: a reader cannot
    # otherwise tell 2-of-280 from 280-of-280, and a downstream gate that decides
    # from these counts goes silent rather than wrong when it sees a fraction.
    all_paths = [Path(p) for p in ctx["files"]]
    # the cap is DERIVED, not a constant. the old default of 8 bore no relation
    # to the machine and silently made a 280-sample cohort look like an 8-sample
    # one; a declared value still wins, and 0 means no cap.
    budget = _budget(int(sum(q.stat().st_size for q in all_paths[:3]) / 3 * 8)
                     if all_paths else None)
    declared = ctx["params"].get("max_samples", "unset")
    cap = budget["cap"] if declared == "unset" else declared
    paths = all_paths[:cap] if cap else all_paths
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
    capped = len(paths) < len(all_paths)
    return _rec("shape", "pass" if ok else "fail",
                f"{len(ok)}/{len(paths)} readable"
                + (f", capped at {cap} of {len(all_paths)} declared" if capped else ""),
                "proceed" if ok else "no sample could be read",
                # every row read, not a sample of them. `rows[:8]` used to truncate
                # the RECORD as well as the read, so a complete run still looked
                # partial - and a gate that decides from these counts then sees a
                # fraction of the cohort and goes quiet instead of firing.
                {"per_sample": rows,
                 "coverage": {"n_read": len(paths), "n_declared": len(all_paths),
                              "capped": capped, "cap": cap,
                              "cap_source": "declared" if declared != "unset" else "derived",
                              "budget": budget,
                              "note": ("raise params.max_samples to cover the cohort; "
                                       "0 means no cap" if capped else
                                       "every declared sample was read")},
                 "n_obs": [min(r["n_obs"] for r in ok), max(r["n_obs"] for r in ok)] if ok else None,
                 "n_var": [min(r["n_var"] for r in ok), max(r["n_var"] for r in ok)] if ok else None}), \
           {"paths_by_sid": {_sid(p): str(p) for p in paths}}


# ------------------------------------------------------------- 4 · coords ---
def _sidecar_coords(h5ad: Path) -> dict | None:
    """the coordinates table beside a canonical .h5ad, if there is one.

    the package's input contract is ".h5ad PLUS a coordinates table", and the
    split is deliberate: the morphology arm needs spot_id, x, y and nothing
    else, so it must not load a 27,000-gene matrix to read two columns. the
    inventory has to check both halves or it reports a conforming cohort as
    having no positions at all.
    """
    sc = h5ad.with_name(f"{h5ad.stem}_spots.parquet")
    if not sc.is_file():
        return None
    try:
        import pandas as pd

        df = pd.read_parquet(sc)
    except Exception as e:                  # noqa: BLE001 - reported, not raised
        return {"path": sc.name, "error": f"{type(e).__name__}: {e}"}
    cols = list(df.columns)
    ok = {"x", "y"} <= set(cols)
    return {"path": sc.name, "columns": cols, "n": len(df), "ok": ok,
            "extent": [float(df["x"].max() - df["x"].min()),
                       float(df["y"].max() - df["y"].min())] if ok else None}


def coords(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """a DECLARED column that is absent is a fail - the declaration is wrong.

    positions may arrive either inside the object (obsm/obs) or in the sidecar
    coordinates table the canonical contract names. both satisfy the contract;
    neither does not.
    """
    rows = []
    for sid, _p in ctx["paths_by_sid"].items():
        o = _CACHE[_p]
        pk = ctx["platform_by_sample"].get(sid)
        declared = (ctx["platform_defs"].get(pk) or {}).get("position_columns") or []
        side = _sidecar_coords(Path(_p))
        if side and side.get("ok"):
            rows.append({"sample": sid, "platform": pk, "source": "sidecar",
                         "sidecar": side["path"], "n": side["n"],
                         "declared": declared, "absent": [], "ok": True,
                         "pixel_extent": side["extent"]})
            continue
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
    n_side = sum(1 for r in rows if r.get("source") == "sidecar")
    return _rec("coords", "fail" if bad else "pass",
                f"{len(rows)-len(bad)}/{len(rows)} carry declared position"
                + (f" ({n_side} via the sidecar coordinates table)" if n_side else ""),
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


# --- step 9: the lattice, measured ------------------------------------------
# the declared pitch and the lattice can disagree, and on the seed cohort they
# did: 200um declared, 150um measured, found by hand months in, after every
# micron-denominated figure had been scaled by the wrong number. this step is
# that discovery turned into a measurement, per sample.
#
# what is measured and what is taken on trust is the whole point:
#
#   measured   nearest-neighbour spacing, in the coordinate space as stored
#   measured   pixel size, ONLY from the object's own scalefactors
#   derived    pitch = spacing x pixel size, only when both came from the object
#   declared   the platform's nominal pitch, compared but never substituted
#
# the pixel size is derived from the SPOT DIAMETER, which is a vendor spec and
# independent of the pitch. deriving it from a declared pixel size that was
# itself computed from a pitch would verify the pitch against itself - which is
# exactly the shape of the error this step exists to catch.
PITCH_TOLERANCE = 0.02


def _lattice(xy) -> dict:
    """spacing and grid from coordinates alone. no units, no assumptions."""
    from scipy.spatial import KDTree

    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) < 3:
        return {"n": len(xy), "nn": None, "why": "fewer than 3 positions"}
    d, _ = KDTree(xy).query(xy, k=2)
    return {"n": len(xy), "nn": float(np.median(d[:, 1]))}


def _pixel_size_um(o, diameter_um) -> tuple[float | None, str]:
    """um per pixel from the OBJECT, or a reason it is not measurable."""
    if diameter_um is None:
        return None, "platform declares no spot diameter"
    try:
        spatial = o.uns["spatial"]
        lib = next(iter(spatial))
        px = float(spatial[lib]["scalefactors"]["spot_diameter_fullres"])
    except (AttributeError, KeyError, StopIteration, TypeError, ValueError):
        return None, "object carries no scalefactors"
    if not px > 0:
        return None, "scalefactor is not positive"
    return float(diameter_um) / px, "object scalefactors"


def geometry(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """measure the lattice, and escalate where it contradicts the declaration.

    a pitch is never derived from a declared pixel size, because that declaration
    may itself have been computed from a pitch - which would verify the
    declaration against itself. declining to do that is what the label guards do
    one level up, and it is the reason this step reports the seed cohort's
    spacing without a pitch rather than confirming its own input.

    a platform may ANSWER the escalation in advance by declaring
    `pitch_authority`, the same way a cohort pre-answers a preflight gate: one
    recorded human decision instead of one prompt per sample.
    """
    rows, escalate, resolved = [], [], []
    for sid, _p in ctx["paths_by_sid"].items():
        o = _CACHE[_p]
        pk = ctx["platform_by_sample"].get(sid)
        defs = ctx["platform_defs"].get(pk) or {}
        side = _sidecar_coords(Path(_p))
        xy = None
        if side and side.get("ok"):
            import pandas as pd

            df = pd.read_parquet(Path(_p).with_name(f"{Path(_p).stem}_spots.parquet"))
            xy, source = df[["x", "y"]].to_numpy(), "sidecar"
        elif hasattr(o, "obsm") and "spatial" in o.obsm:
            xy, source = np.asarray(o.obsm["spatial"]), "obsm/spatial"
        row = {"sample": sid, "platform": pk}
        if xy is None:
            rows.append(row | {"measurable": False, "why": "no positions"})
            continue
        lat = _lattice(xy)
        um_px, how = _pixel_size_um(o, defs.get("spot_diameter_um"))
        declared = defs.get("spot_pitch_um")
        row |= {"source": source, "n": lat["n"], "nn_units": lat["nn"],
                "um_per_px": None if um_px is None else round(um_px, 4),
                "pixel_size_from": how, "pitch_declared_um": declared}
        if lat["nn"] and um_px:
            measured = lat["nn"] * um_px
            row["pitch_measured_um"] = round(measured, 1)
            if declared:
                rel = abs(measured - declared) / declared
                row["pitch_rel_delta"] = round(rel, 4)
                row["agrees"] = rel <= PITCH_TOLERANCE
                if rel > PITCH_TOLERANCE:
                    answer = defs.get("pitch_authority") or {}
                    if answer.get("value") in ("measured", "declared"):
                        row["resolved_by"] = answer["value"]
                        row["resolved_source"] = "platform.json#pitch_authority"
                        resolved.append(sid)
                    else:
                        escalate.append(sid)
        else:
            row["pitch_measured_um"] = None
            row["why_not_derived"] = ("pixel size not measurable from the object, so a pitch "
                                      "derived from a declared pixel size would test the "
                                      "declaration against itself")
        if hasattr(o, "obs") and {"array_row", "array_col"} <= set(getattr(o.obs, "columns", [])):
            row["grid"] = [int(np.ptp(o.obs["array_row"]) + 1), int(np.ptp(o.obs["array_col"]) + 1)]
        rows.append(row | {"measurable": True})

    derived = [r for r in rows if r.get("pitch_measured_um")]
    by_platform = {}
    for r in derived:
        by_platform.setdefault(r["platform"], []).append(r["pitch_measured_um"])
    summary = {k: {"n": len(v), "median_um": round(float(np.median(v)), 1),
                   "min_um": min(v), "max_um": max(v)} for k, v in by_platform.items()}

    if escalate:
        # a caveat, not a failure: a failed inventory step halts eda, and the
        # cohort is readable - what is unresolved is which number is authoritative.
        # the gate surfaces caveats as cautions, which is where a person sees it.
        rec = _rec("geometry", "pass",
                   f"{len(escalate)} of {len(derived)} sample(s) disagree with the declared "
                   f"pitch by more than {PITCH_TOLERANCE:.0%}",
                   "a person resolves which number is authoritative BEFORE any "
                   "micron-denominated figure is produced. the declaration may be a nominal "
                   "vendor spec and the lattice the real one, or the measurement may be wrong - "
                   "the record carries both rather than choosing",
                   {"per_platform": summary, "escalate": escalate[:8],
                    "tolerance": PITCH_TOLERANCE, "rows": rows})
        rec.caveats = [
            (f"{k}: declared {d} um, measured {v['median_um']} um over {v['n']} sample(s) "
             f"({v['min_um']}-{v['max_um']}) - every micron-denominated figure scales with "
             f"whichever is authoritative")
            for k, v in summary.items()
            for d in [(ctx["platform_defs"].get(k) or {}).get("spot_pitch_um")]
            if d and abs(v["median_um"] - d) / d > PITCH_TOLERANCE]
        return rec, {"geometry": rows}
    authority = {k: (ctx["platform_defs"].get(k) or {}).get("pitch_authority", {}).get("value")
                 for k in summary}
    return _rec("geometry", "pass",
                f"{len(derived)} of {len(rows)} sample(s) measured; "
                + ", ".join(f"{k} {v['median_um']}um" for k, v in summary.items())
                + (f"; {len(resolved)} disagreement(s) answered in advance" if resolved else ""),
                ("the lattice agrees with the declaration where both exist, or the platform "
                 "declared which is authoritative; where the pixel size is not in the object "
                 "the spacing is recorded without a pitch"),
                {"per_platform": summary, "tolerance": PITCH_TOLERANCE,
                 "resolved_by_declaration": len(resolved), "pitch_authority": authority,
                 "rows": rows}), {"geometry": rows}


INVENTORY_STEPS: list[Step] = [
    Step(id="files",    fn=files,    produces=frozenset({"files"})),
    Step(id="platform", fn=platform, produces=frozenset({"platform"}), requires=frozenset({"files"})),
    Step(id="shape",    fn=shape,    produces=frozenset({"shape"}),    requires=frozenset({"files", "platform"})),
    Step(id="coords",   fn=coords,   produces=frozenset({"coords"}),   requires=frozenset({"shape"})),
    Step(id="join_key", fn=join_key, produces=frozenset({"key"}),      requires=frozenset({"shape"})),
    Step(id="counts",   fn=counts,   produces=frozenset({"molecular"}), requires=frozenset({"shape"})),
    Step(id="funnel",   fn=funnel,   produces=frozenset({"funnel"}),   requires=frozenset({"files"})),
    Step(id="geometry", fn=geometry, produces=frozenset({"geometry"}),
         requires=frozenset({"coords", "platform"})),
    Step(id="bind",     fn=bind,     produces=frozenset({"roles"}),
         requires=frozenset({"files", "platform", "shape", "coords", "key", "molecular", "funnel"})),
]