"""a cohort's files -> the arrays `EVALUATE_STEPS` takes.

the adapter layer, not a contract and not a protocol: it turns declared paths
into matrices and decides nothing about them. without it the chain is a library
function - it takes arrays, MCP carries JSON, and nothing could hand it a
35,594 x 512 matrix over stdio.

it reads the RAW declaration rather than `ProjectConfig`'s fields, and that is
deliberate. three of the fields this needs carry the seed cohort's values as
package defaults:

    h2_label      = "mc_megacluster"
    supervision   = "mc_weights_niche"
    headline_view = "z_he"

a second cohort that declared none of them would inherit all three silently and
be scored against a label it does not have. the rule those defaults break is
the repo's own - a declaration is read, not guessed; absent means refuse with a
reason - and the numeric threshold lint cannot see them because they are
strings. so this reader asks the file, names the field that is missing, and
leaves `ProjectConfig` alone for the callers that already depend on it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# every field this reader needs, and what a cohort loses by not declaring it.
# the message is the whole value of the refusal: "h2_label is not declared" sends
# a reader to the right line, and a silent default does not.
REQUIRED = {
    "niches_dir": "where the per-sample join tables are. the labels and the "
                  "continuous signals come from them",
    "runs_dir": "where the fitted runs are. the views come from "
                "{run}/embeddings_test.parquet",
    "h2_label": "which label a clustering claim is scored against. the seed "
                "cohort's is a niche-level NMF class; another cohort's will not "
                "be, and scoring against the wrong one is a different claim",
    "headline_view": "which view carries the claim. a cohort with a circular "
                     "view on one side must say which side is clean",
    "supervision": "the signal that trains the arms. it must not appear among "
                   "the features, which is guard G6, and the guard cannot run "
                   "on an undeclared target",
    "seed": "the split and every estimator that takes one",
}


class DeclarationMissing(RuntimeError):
    """a field this cohort must declare is absent. never defaulted."""


def declared(project_dir: str | Path) -> dict[str, Any]:
    """the fields `EVALUATE_STEPS` needs, read from project.json.

    refuses with every missing field named at once rather than one per call: a
    cohort being scaffolded wants the whole list, and discovering them one
    exception at a time is six round trips.
    """
    d = Path(project_dir)
    p = d / "project.json"
    if not p.is_file():
        raise DeclarationMissing(
            f"no project.json in {d}. `omicstra init --project-dir {d}` scaffolds one")

    raw = json.loads(p.read_text())
    missing = {k: why for k, why in REQUIRED.items()
               if raw.get(k) in (None, "", [], {})}
    if missing:
        lines = "\n  ".join(f"{k}: {why}" for k, why in sorted(missing.items()))
        raise DeclarationMissing(
            f"{p} declares {len(REQUIRED) - len(missing)} of {len(REQUIRED)} "
            f"fields the evaluation chain needs. absent:\n  " + lines)
    return {k: raw[k] for k in REQUIRED} | {
        "project_id": raw.get("project_id"),
        "k_neighbors": raw.get("k_neighbors"),
    }


def _views(runs_path: Path, run: str, view: str):
    """the run's view matrix, and the frame it came from.

    the frame travels with it because the labels are joined on its own keys -
    a second read of the same parquet is a second chance for the two to be in
    different orders.
    """
    import numpy as np
    import pandas as pd

    p = runs_path / run / "embeddings_test.parquet"
    if not p.is_file():
        raise DeclarationMissing(
            f"no fitted run at {p}. `runs_dir` points at {runs_path}, and this "
            f"cohort's declared runs are what the grid fitted - a verdict needs "
            "one of them")
    df = pd.read_parquet(p)
    frame = df.reset_index().copy()
    if "spot_id" not in frame.columns:
        frame["spot_id"] = df.index.values
    if view not in frame.columns:
        raise DeclarationMissing(
            f"the run at {p} has no {view!r} column. this cohort declares "
            f"headline_view={view!r}; the run carries "
            f"{[c for c in frame.columns if c.startswith('z_')]}")
    return np.stack([np.asarray(v, np.float32) for v in frame[view]]), frame


def arrays(project_dir: str | Path, run: str, *, paired_view: str | None = None,
           signals: tuple[str, ...] = ()) -> dict[str, Any]:
    """the ctx `EVALUATE_STEPS` reads, assembled from one run of one cohort.

    `signals` is the caller's, not the cohort's: which continuous signals to
    test for association is a question, and a cohort does not declare the
    questions somebody will ask of it. an undeclared signal is refused by name
    rather than skipped.

    the dtype is float32 throughout, which is the dtype the published artifacts
    were computed at - and for KMeans that is part of the method, not a
    precision detail. `measures.evaluation.cluster_agreement` records it.
    """
    import numpy as np
    import pandas as pd

    from omicstra.contracts.project import ProjectConfig

    dec = declared(project_dir)
    cfg = ProjectConfig.from_dir(project_dir)
    runs_path, niches_path = cfg.path(dec["runs_dir"]), cfg.path(dec["niches_dir"])
    view = dec["headline_view"]

    he, frame = _views(runs_path, run, view)
    spaces = {view: he}
    paired: dict[str, Any] = {}
    if paired_view:
        if paired_view not in frame.columns:
            raise DeclarationMissing(
                f"the run has no {paired_view!r} column to pair {view!r} with")
        paired[view] = np.stack([np.asarray(v, np.float32)
                                 for v in frame[paired_view]])

    # the labels and signals live in the join tables, one per sample
    want = [dec["h2_label"], *signals]
    frames = []
    for sub in frame["subarray"].unique():
        jp = niches_path / f"{sub}.parquet"
        if not jp.is_file():
            continue
        nf = pd.read_parquet(jp).reset_index()
        nf["subarray"] = sub
        have = [c for c in want if c in nf.columns]
        frames.append(nf[["subarray", "spot_id", *have]])
    if not frames:
        raise DeclarationMissing(
            f"no join table for any sample of {run} under {niches_path}. "
            "`niches_dir` is where the labels come from")

    joined = pd.concat(frames, ignore_index=True)
    if absent := [c for c in want if c not in joined.columns]:
        raise DeclarationMissing(
            f"{absent} are not columns of the join tables under {niches_path}. "
            f"this cohort declares h2_label={dec['h2_label']!r}, and the signals "
            "asked for are the caller's")
    m = frame.merge(joined, on=["subarray", "spot_id"], how="left")

    label = dec["h2_label"]
    keep = m[label].notna()
    m = m[keep]
    idx = np.flatnonzero(keep.to_numpy())

    # a label that encodes its own absence as a VALUE is reported, never
    # silently scored. `build_niche_join.py:276` writes -1 for a unit with no
    # class and its comment says "eval code already filters mc_megacluster < 0";
    # the published eval filters `notna()` only, so 882 of this cohort's 35,594
    # units entered the published ARI as an extra class. the comment asserts a
    # filter that does not exist.
    #
    # the adapter does not take the decision - removing them would move a
    # published number - but a caller cannot score them unknowingly either.
    lab = m[label]
    sentinels = int((lab < 0).sum()) if lab.dtype.kind in "iuf" else 0

    subject = cfg.subject_id_column if hasattr(cfg, "subject_id_column") else None
    subject = subject or "patient_id"
    if subject not in m.columns:
        raise DeclarationMissing(
            f"the run carries no {subject!r} column, so there is no confounder "
            "axis to hold out by. every split and every confound guard is taken "
            "on it")

    sub_arr = m["subarray"].to_numpy()
    conf_arr = m[subject].to_numpy()

    # patient-level biology labels (H2-C), if declared
    bl = biology_labels(project_dir, sub_arr, conf_arr)
    bio_ctx: dict[str, Any] = {}
    bio_report: dict[str, Any] = {}
    if bl is not None:
        bio_ctx = bl["labels"]
        bio_report = bl["report"]

    # raw baseline cosine matrices for geometric faithfulness (rho)
    raw_bl = raw_baseline_cosines(project_dir, run, m)

    return {
        "sample_count": int(m["subarray"].nunique()),
        "n_units_total": len(frame),
        "n_units_labelled": len(m),
        "spaces": {k: v[idx] for k, v in spaces.items()},
        "paired": {k: v[idx] for k, v in paired.items()},
        "labels": {label: m[label].astype(int).to_numpy()},
        "label_sentinels": {label: sentinels},
        "label_classes": {label: int(m[label].nunique())},
        "label_note": (
            f"{sentinels} unit(s) carry a negative value for {label!r}, which the "
            "join writes for ABSENCE. they are kept because dropping them would "
            "move a published number, and reported because scoring an absence "
            "marker as a class is not a biological claim"
            if sentinels else ""),
        "subarray": sub_arr,
        "confounder": conf_arr,
        "biology_labels": bio_ctx or None,
        "biology_labels_report": bio_report or None,
        "scores": {s: m[s].to_numpy(dtype=float) for s in signals},
        "raw_baselines": raw_bl,
        "declared": dec,
        "run": run,
        "view": view,
    }


def biology_labels(project_dir: str | Path,
                   subarrays: Any,
                   patients: Any,
                   ) -> dict[str, Any] | None:
    """patient-level biology labels propagated to the unit (niche) level.

    reads the declaration from project.json's `biology_labels` block:
        labels_path   - path to the patient-level labels file (tsv)
        labels_key    - column in that file keying patients (e.g. patient_id)
        subject_key   - column on the niche frame that identifies the patient
        columns       - {output_alias: source_column} mapping

    returns {alias: niche_array} or None if no declaration exists. raises
    DeclarationMissing on structural errors (missing file, broken join).
    """
    import numpy as np
    import pandas as pd

    d = Path(project_dir)
    p = d / "project.json"
    if not p.is_file():
        return None
    raw = json.loads(p.read_text())
    bl = raw.get("biology_labels")
    if not bl or not isinstance(bl, dict):
        return None

    for field in ("labels_path", "labels_key", "subject_key", "columns"):
        if field not in bl or not bl[field]:
            raise DeclarationMissing(
                f"biology_labels is declared but {field!r} is absent. "
                "every field is required: labels_path, labels_key, "
                "subject_key, columns")

    labels_file = (d / bl["labels_path"]).resolve()
    if not labels_file.is_file():
        raise DeclarationMissing(
            f"biology_labels.labels_path points at {labels_file} which "
            "does not exist")

    lf = pd.read_csv(labels_file, sep="\t")
    lk = bl["labels_key"]
    if lk not in lf.columns:
        raise DeclarationMissing(
            f"biology_labels.labels_key={lk!r} is not a column of "
            f"{labels_file}. columns: {list(lf.columns)}")

    col_map: dict[str, str] = bl["columns"]
    missing_cols = [src for src in col_map.values() if src not in lf.columns]
    if missing_cols:
        raise DeclarationMissing(
            f"biology_labels.columns references {missing_cols} which are "
            f"not in {labels_file}. columns: {list(lf.columns)}")

    # build patient -> {alias: value} lookup from the labels file
    dup = lf[lf[lk].duplicated(keep=False)]
    if not dup.empty:
        raise DeclarationMissing(
            f"biology_labels file has {len(dup)} rows with duplicate "
            f"{lk!r} values: {sorted(dup[lk].unique()[:5])}...")

    pt_lookup: dict[Any, dict[str, str]] = {}
    for _, row in lf.iterrows():
        pid = row[lk]
        pt_lookup[pid] = {alias: str(row[src])
                          for alias, src in col_map.items()}

    # propagate to niche level via the subject (patient) column
    sub_arr = np.asarray(subarrays)
    pat_arr = np.asarray(patients)

    # validate: every subarray's niches must agree on patient_id
    unique_subs = np.unique(sub_arr)
    for s in unique_subs:
        mask = sub_arr == s
        pts_in_sub = np.unique(pat_arr[mask])
        if len(pts_in_sub) > 1:
            raise DeclarationMissing(
                f"subarray {s!r} maps to multiple patients: {pts_in_sub}. "
                "the join is inconsistent")

    # build per-niche arrays; track missing patients
    n = len(pat_arr)
    result: dict[str, Any] = {}
    missing_patients: set = set()
    for alias in col_map:
        arr = np.empty(n, dtype=object)
        for i in range(n):
            pid = pat_arr[i]
            entry = pt_lookup.get(pid)
            if entry is None:
                # try numeric coercion: patient_id may be int in one, str in other
                try:
                    entry = pt_lookup.get(int(pid))
                except (ValueError, TypeError):
                    pass
            if entry is None:
                missing_patients.add(pid)
                arr[i] = None
            else:
                arr[i] = entry[alias]
        result[alias] = arr

    report: dict[str, Any] = {
        "n_units": n,
        "n_patients": len(np.unique(pat_arr)),
        "n_labels_file": len(lf),
        "aliases": list(col_map.keys()),
        "sources": list(col_map.values()),
    }
    if missing_patients:
        report["missing_patients"] = sorted(str(p) for p in missing_patients)
        report["n_missing"] = len(missing_patients)

    return {"labels": result, "report": report}


def raw_baseline_cosines(project_dir: str | Path,
                         run: str,
                         filtered_frame,
                         ) -> dict[str, Any] | None:
    """raw baseline cosine matrices for geometric faithfulness (rho).

    loads raw features (virchow2_niche, novae_niche, gpath2vec_niche) from
    the niche parquets for the same niches that survived label filtering,
    mean-pools per subarray, and returns cosine matrices.

    returns None if the niche parquets lack the required feature columns.
    the returned cosines are in np.unique(subarray) order, matching the
    step's own subarray ordering.
    """
    import numpy as np
    import pandas as pd

    from omicstra.measures.biology import cosine_sim_matrix, mean_pool_l2

    d = Path(project_dir)
    raw = json.loads((d / "project.json").read_text())

    from omicstra.contracts.project import ProjectConfig
    cfg = ProjectConfig.from_dir(d)
    runs_path = cfg.path(raw["runs_dir"])
    niches_path = cfg.path(raw["niches_dir"])

    run_cfg_path = runs_path / run / "run_config.json"
    if not run_cfg_path.is_file():
        return None
    run_cfg = json.loads(run_cfg_path.read_text())
    st_features = tuple(run_cfg.get("st_features", ("novae_niche", "gpath2vec_niche")))

    unique_subs = np.unique(filtered_frame["subarray"])
    raw_he_pooled = []
    raw_st_pooled = []

    for sub in unique_subs:
        jp = niches_path / f"{sub}.parquet"
        if not jp.is_file():
            return None
        nf = pd.read_parquet(jp).reset_index()

        sub_spots = set(filtered_frame.loc[
            filtered_frame["subarray"] == sub, "spot_id"])
        if "spot_id" in nf.columns:
            mask = nf["spot_id"].isin(sub_spots)
        else:
            mask = nf.index.isin(sub_spots)
        nf = nf[mask]

        if len(nf) == 0:
            return None

        if "virchow2_niche" not in nf.columns:
            return None
        he = np.stack([np.asarray(v, np.float32) for v in nf["virchow2_niche"]])
        raw_he_pooled.append(mean_pool_l2(he))

        st_parts = []
        for feat in st_features:
            if feat not in nf.columns:
                return None
            arr = np.stack([np.asarray(v, np.float32) for v in nf[feat]])
            st_parts.append(arr)
        st = np.concatenate(st_parts, axis=1)
        raw_st_pooled.append(mean_pool_l2(st))

    R_he = np.stack(raw_he_pooled)
    R_st = np.stack(raw_st_pooled)

    return {
        "raw_he": cosine_sim_matrix(R_he),
        "raw_st": cosine_sim_matrix(R_st),
        "n_subarrays": len(unique_subs),
        "he_dim": int(R_he.shape[1]),
        "st_dim": int(R_st.shape[1]),
        "st_features": list(st_features),
    }


__all__ = ["REQUIRED", "DeclarationMissing", "arrays", "biology_labels",
           "raw_baseline_cosines", "declared"]
