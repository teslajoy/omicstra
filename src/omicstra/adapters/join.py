"""a cohort's files -> the arrays `JOIN_STEPS` takes, one sample at a time.

the adapter layer. it turns declared paths into per-unit frames and decides
nothing about them: which assays exist, where they live, how they are read and
which columns carry measurement are all the cohort's to declare.

why a LOADER REGISTRY rather than a format taxonomy
---------------------------------------------------
the seed cohort presents four shapes, and a reader that hardcoded them would be
shaped like that cohort:

    an .npy beside a _meta.tsv that holds its row index
    one parquet per sample, data columns by prefix, carrying a neighbour list
    one parquet for the whole cohort, filtered by a key column
    one delimited table for the whole cohort, grouped by a sample column

so a loader is resolved BY DECLARED NAME, exactly as `models/encoders.py`
resolves an encoder. a cohort naming a loader that does not exist is refused
with the list of those that do - which is a cheaper failure than a reader that
guesses from a file extension and succeeds on the wrong axis.

adding a fifth loader is a package change, and that is the honest cost: the
alternative is sniffing, and a sniffed format that reads a coordinate column as
a measurement produces vectors nobody can tell are wrong.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

LoaderFn = Callable[[dict, str, Path], Any]
LOADERS: dict[str, LoaderFn] = {}


class JoinDeclarationMissing(RuntimeError):
    """a field or a loader this cohort must declare is absent. never defaulted."""


def loader(name: str):
    def register(fn: LoaderFn) -> LoaderFn:
        LOADERS[name] = fn
        return fn
    return register


def _need(spec: dict, field: str, assay: str, why: str):
    v = spec.get(field)
    if v in (None, "", [], {}):
        raise JoinDeclarationMissing(
            f"assay {assay!r} declares loader {spec.get('loader')!r} and no "
            f"{field!r}. {why}")
    return v


# --------------------------------------------------------------- loaders ----
@loader("npy_with_meta")
def _npy_with_meta(spec, sample, root, *, assays=None):
    """an .npy of vectors beside a table that holds its ROW ORDER.

    the row index lives in a separate file, so the two must be read together or
    the vectors are anonymous - there is nothing in the .npy that says which
    unit row 0 belongs to.

    `index_from` names ANOTHER assay whose index this one shares. the seed
    cohort's per-unit arrays have no meta of their own - 280 .npy files, zero
    .tsv - and the script's own comment is "row order same as meta". that is a
    real dependency between two files and it is declared rather than assumed,
    because reading it wrong silently permutes every vector against its unit.
    """
    import numpy as np
    import pandas as pd

    d = root / _need(spec, "path", sample, "the directory holding the arrays")
    arr = np.load(d / f"{sample}.npy")

    if borrow := spec.get("index_from"):
        if not assays or borrow not in assays:
            raise JoinDeclarationMissing(
                f"this assay declares index_from={borrow!r}, which must be "
                "another assay loaded BEFORE it. declare it earlier in `assays`")
        idx = assays[borrow].index
        if arr.shape[0] != len(idx):
            raise JoinDeclarationMissing(
                f"{sample}: {arr.shape[0]} vectors against {len(idx)} rows in "
                f"{borrow!r}. a borrowed index must match exactly or the "
                "vectors are permuted against their units")
        return pd.DataFrame(arr, index=idx,
                            columns=[f"d{i}" for i in range(arr.shape[1])])

    suffix = spec.get("meta_suffix", "_meta.tsv")
    sep = spec.get("sep", "\t")
    meta = pd.read_csv(d / f"{sample}{suffix}", sep=sep, index_col=0)
    if arr.shape[0] != len(meta):
        raise JoinDeclarationMissing(
            f"{sample}: {arr.shape[0]} vectors against {len(meta)} meta rows. the "
            "meta table IS the row index, so a mismatch means the two files are "
            "not of the same run")
    cols = [f"d{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=meta.index, columns=cols)


@loader("parquet_per_sample")
def _parquet_per_sample(spec, sample, root, *, assays=None):
    """one parquet per sample, already indexed by unit."""
    import pandas as pd

    d = root / _need(spec, "path", sample, "the directory holding the parquets")
    pattern = spec.get("filename", "{sample}.parquet")
    p = d / pattern.format(sample=sample)
    if not p.is_file():
        raise JoinDeclarationMissing(f"no file at {p} for sample {sample!r}")
    df = pd.read_parquet(p)
    if prefix := spec.get("column_prefix"):
        keep = [c for c in df.columns if c.startswith(prefix)]
        if not keep:
            raise JoinDeclarationMissing(
                f"{p.name} has no column starting {prefix!r}; it has "
                f"{list(df.columns)[:6]}")
        return df[keep]
    return df


@loader("parquet_one_file")
def _parquet_one_file(spec, sample, root, *, assays=None):
    """one parquet for the whole cohort, filtered to this sample by a key column.

    `non_data_columns` is required and is not a convenience: a frame like this
    carries coordinates, ids and provenance beside the measurement, and
    selecting by dtype pulls a pixel coordinate into the vector. the seed
    cohort's build script had to name five such columns by hand.
    """
    import pandas as pd

    p = root / _need(spec, "path", sample, "the parquet file")
    key = _need(spec, "key_column", sample, "which column names the sample")
    unit = _need(spec, "unit_column", sample, "which column names the unit")
    non_data = _need(spec, "non_data_columns", sample,
                     "which columns are NOT measurement. inferring them by "
                     "dtype pulls coordinates into the vector")
    df = pd.read_parquet(p, filters=[(key, "==", sample)])
    if df.empty:
        raise JoinDeclarationMissing(
            f"{p.name} has no rows with {key}=={sample!r}")
    data = [c for c in df.columns if c not in set(non_data)]
    return df.set_index(unit)[data]


@loader("table_one_file")
def _table_one_file(spec, sample, root, *, assays=None):
    """one delimited table for the whole cohort, grouped by a sample column.

    `sample_key` exists because a cohort can carry two key spaces for one
    sample - the seed cohort's label tables are keyed by a slide position while
    its embeddings are keyed by the full sample id, and the build script
    stripped a prefix to cross between them. a declared transform is readable;
    a prefix strip buried in a helper is not.
    """
    import pandas as pd

    p = root / _need(spec, "path", sample, "the table file")
    key = _need(spec, "key_column", sample, "which column names the sample")
    unit = _need(spec, "unit_column", sample, "which column names the unit")
    sep = spec.get("sep", "\t")
    want = sample
    if (strip := spec.get("strip_prefix_before")) is not None:
        # "TNBC1_CN1_C1" -> "CN1_C1" when the label tables use a slide position
        want = sample.split(strip, 1)[1] if strip in sample else sample
    df = pd.read_csv(p, sep=sep)
    for c in (key, unit):
        if c not in df.columns:
            raise JoinDeclarationMissing(
                f"{p.name} has no {c!r} column; it has {list(df.columns)[:8]}")
    sub = df[df[key] == want]
    if sub.empty:
        return sub.set_index(unit)
    cols = spec.get("columns") or [c for c in sub.columns if c not in (key, unit)]
    return sub.set_index(unit)[cols]


# ------------------------------------------------------- the declaration ----
REQUIRED = {
    "assays": "which assays a unit is built from, each with a loader and a path",
    "neighbour_source": "which assay's declared neighbour list is authoritative. "
                        "two assays can disagree, so this is not inferred",
    "neighbour_column": "the column in that assay holding the neighbour list",
}


def declared(project_dir: str | Path) -> dict[str, Any]:
    """the cohort's `join_inputs`, read from project.json. refuses by field."""
    d = Path(project_dir)
    p = d / "project.json"
    if not p.is_file():
        raise JoinDeclarationMissing(
            f"no project.json in {d}. `omicstra init --project-dir {d}` scaffolds one")
    raw = json.loads(p.read_text())
    spec = raw.get("join_inputs")
    if not spec:
        raise JoinDeclarationMissing(
            f"{p} declares no `join_inputs`. the join reads assays this cohort "
            "names - there is no default set, because a default would be the "
            "seed cohort's four assays under the seed cohort's paths.\n  "
            + "\n  ".join(f"{k}: {why}" for k, why in sorted(REQUIRED.items())))

    missing = {k: why for k, why in REQUIRED.items() if not spec.get(k)}
    if missing:
        raise JoinDeclarationMissing(
            f"{p}#join_inputs is missing {sorted(missing)}:\n  "
            + "\n  ".join(f"{k}: {why}" for k, why in sorted(missing.items())))

    for name, a in (spec.get("assays") or {}).items():
        ldr = a.get("loader")
        if not ldr:
            raise JoinDeclarationMissing(
                f"assay {name!r} declares no loader. available: "
                f"{sorted(LOADERS)}")
        if ldr not in LOADERS:
            raise JoinDeclarationMissing(
                f"assay {name!r} declares loader {ldr!r}, which does not exist. "
                f"available: {sorted(LOADERS)}. adding one is a package change - "
                "the alternative is sniffing the format, and a sniffed format "
                "that reads a coordinate as a measurement is not detectable")
    if spec["neighbour_source"] not in (spec.get("assays") or {}):
        raise JoinDeclarationMissing(
            f"neighbour_source {spec['neighbour_source']!r} is not one of this "
            f"cohort's assays {sorted(spec.get('assays') or {})}")
    return spec


def samples(project_dir: str | Path) -> list[str]:
    """the samples every REQUIRED assay covers - the intersection, sorted.

    this is the `n_in` the join's first stage reports against, and it is
    computed here rather than in the chain because it is a property of the
    files: a sample absent from one assay is not a sample the join can build.
    """
    from omicstra.contracts.project import ProjectConfig

    spec = declared(project_dir)
    cfg = ProjectConfig.from_dir(project_dir)
    per_assay: dict[str, set[str]] = {}
    for name, a in spec["assays"].items():
        if not a.get("required_for_sample_set", True):
            continue
        root = cfg.path(a["path"])
        if a["loader"] == "npy_with_meta":
            per_assay[name] = {p.stem for p in Path(root).glob("*.npy")}
        elif a["loader"] == "parquet_per_sample":
            pat = a.get("filename", "{sample}.parquet")
            pre, _, suf = pat.partition("{sample}")
            per_assay[name] = {p.name[len(pre):len(p.name) - len(suf)]
                               for p in Path(root).glob(pat.format(sample="*"))}
        else:
            import pandas as pd

            key = a.get("key_column")
            col = pd.read_parquet(root, columns=[key]) if str(root).endswith(
                ".parquet") else pd.read_csv(root, sep=a.get("sep", "\t"),
                                             usecols=[key])
            per_assay[name] = set(col[key].astype(str).unique())
    if not per_assay:
        raise JoinDeclarationMissing(
            "no assay is declared `required_for_sample_set`, so there is no "
            "sample set to intersect")
    common = set.intersection(*per_assay.values())
    return sorted(common)


def arrays(project_dir: str | Path, sample: str) -> dict[str, Any]:
    """the ctx `JOIN_STEPS` reads, for ONE sample of one cohort."""
    from omicstra.contracts.project import ProjectConfig

    spec = declared(project_dir)
    cfg = ProjectConfig.from_dir(project_dir)
    root = Path(cfg.project_dir or project_dir)

    # insertion order is load order, so an assay that borrows another's index
    # must be declared after it. the refusal in `npy_with_meta` says so.
    assays: dict[str, Any] = {}
    for name, a in spec["assays"].items():
        assays[name] = LOADERS[a["loader"]](a, sample, root, assays=assays)

    src = spec["neighbour_source"]
    col = spec["neighbour_column"]
    raw = LOADERS[spec["assays"][src]["loader"]](
        {**spec["assays"][src], "column_prefix": None}, sample, root, assays=assays)
    if col not in raw.columns:
        raise JoinDeclarationMissing(
            f"the neighbour source {src!r} has no {col!r} column; it has "
            f"{list(raw.columns)[:8]}")
    neighbours = {u: list(v) for u, v in raw[col].items()}

    # a label's ROLE places it: a continuous signal is pooled over the
    # neighbourhood and reaches the chain as an assay; a discrete class is read
    # from the centre unit only and reaches it as a label. the caller does not
    # choose - pooling a discrete class is meaningless and the declaration is
    # where that is stated.
    labels: dict[str, Any] = {}
    for n, sp in (spec.get("labels") or {}).items():
        role = sp.get("role")
        if role not in ("pooled", "centre"):
            raise JoinDeclarationMissing(
                f"label {n!r} declares role={role!r}. it must be 'pooled' - a "
                "continuous signal averaged over the neighbourhood - or "
                "'centre', a discrete class read from the centre unit, which "
                "cannot be averaged at all")
        frame = LOADERS[sp["loader"]](sp, sample, root, assays=assays)
        if role == "pooled":
            assays[n] = frame
        else:
            labels[n] = frame

    subj = spec.get("subject") or {}
    subject_of, subject_labels = {}, {}
    if subj:
        frame = LOADERS[subj["loader"]](subj, sample, root, assays=assays)
        sid_col = subj.get("subject_column")
        if sid_col and sid_col in frame.columns and len(frame):
            sid = frame[sid_col].iloc[0]
            subject_of[sample] = sid
            subject_labels[sid] = {k: v for k, v in (subj.get("labels") or {}).items()}

    return {"sample": sample, "assays": assays, "neighbours": neighbours,
            "unit_labels": labels, "subject_of": subject_of,
            "subject_labels": subject_labels, "declared": spec}


__all__ = ["LOADERS", "REQUIRED", "JoinDeclarationMissing", "arrays",
           "declared", "loader", "samples"]
