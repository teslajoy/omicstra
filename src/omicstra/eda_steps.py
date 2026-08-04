"""EDA steps - each observes, compares against a declared criterion, emits a record.

every step takes AnnData and returns a DiagnosticRecord. AnnData is the contract:
cohort-specific loading is an adapter's job, so a step never learns a file format.
Visium, Xenium and MERFISH all have standard loaders; a bespoke cohort writes one
adapter and gets every step for free.

expected AnnData:
    .X                raw counts (steps that need raw say so)
    .var_names        gene identifiers
    .obsm["spatial"]  coordinates, one row per observation
    .obs[sample_key]  which sample each observation belongs to
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .records import ArtifactRef, DiagnosticRecord


def _dense(x) -> np.ndarray:
    return np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()


def _fmt(n: int) -> str:
    return f"{n:,}"


# --- count statistics ------------------------------------------------------
def count_statistics(adata, sample_key: str | None = None,
                     source: str | None = None) -> DiagnosticRecord:
    """depth, detection and sparsity. the floor everything else stands on."""
    t0 = time.time()
    X = adata.X
    total = np.asarray(X.sum(axis=1)).ravel()
    detected = np.asarray((X > 0).sum(axis=1)).ravel()
    nnz = X.nnz if hasattr(X, "nnz") else int((X != 0).sum())
    sparsity = 1.0 - nnz / (adata.n_obs * adata.n_vars)

    n_samples = adata.obs[sample_key].nunique() if sample_key and sample_key in adata.obs else 1
    integer_valued = bool(np.allclose(total, np.round(total)))

    obs = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_samples": int(n_samples),
        "median_total_per_obs": float(np.median(total)),
        "median_detected_per_obs": float(np.median(detected)),
        "sparsity": round(float(sparsity), 4),
        "values_are_integers": integer_valued,
    }
    return DiagnosticRecord(
        step_id="count_statistics",
        method="per-observation depth, detection and matrix sparsity",
        params={"sample_key": sample_key},
        scope=f"{_fmt(adata.n_obs)} observations x {_fmt(adata.n_vars)} features"
              + (f", {n_samples} samples" if n_samples > 1 else ""),
        observed=obs,
        criterion="counts must be integer-valued for count-based encoders",
        result=("integer counts" if integer_valued
                else "NOT integer-valued - values are normalised or transformed"),
        decision=("safe for count-based encoders" if integer_valued
                  else "resolve encoder_input_decision before any count-based encoder"),
        status="pass" if integer_valued else "fail",
        caveats=[] if integer_valued else
                ["a directory named 'raw' is not evidence; this was tested, not assumed"],
        inputs=[ArtifactRef(path=source)] if source else [],
        duration_s=round(time.time() - t0, 3),
    )


# --- marker expression -----------------------------------------------------
def marker_expression(adata, positive: dict[str, str], negative: dict[str, str],
                      min_pct_positive: float = 1.0,
                      max_pct_negative: float = 10.0) -> DiagnosticRecord:
    """are the markers this tissue should show present, and the ones it should not, absent.

    positive/negative map gene -> why we expect it. the 'why' travels into the
    record so the reader can judge the choice, not just the number.
    """
    t0 = time.time()
    obs: dict[str, Any] = {}
    missing: list[str] = []

    for group, genes in (("positive", positive), ("negative", negative)):
        for gene, expectation in genes.items():
            if gene not in adata.var_names:
                missing.append(gene)
                continue
            v = _dense(adata[:, gene].X)
            obs[gene] = {
                "group": group,
                "mean": round(float(v.mean()), 4),
                "pct_expressing": round(float((v > 0).mean() * 100), 2),
                "expected": expectation,
            }

    pos_ok = [g for g in positive if g in obs and obs[g]["pct_expressing"] >= min_pct_positive]
    neg_ok = [g for g in negative if g in obs and obs[g]["pct_expressing"] <= max_pct_negative]
    passed = len(pos_ok) == len([g for g in positive if g in obs]) and \
             len(neg_ok) == len([g for g in negative if g in obs])

    return DiagnosticRecord(
        step_id="marker_expression",
        method="fraction of observations with non-zero expression, per declared marker",
        params={"min_pct_positive": min_pct_positive, "max_pct_negative": max_pct_negative},
        scope=f"{_fmt(adata.n_obs)} observations, "
              f"{len(positive)} positive and {len(negative)} negative markers",
        observed=obs,
        criterion=f"positives >= {min_pct_positive}% expressing, "
                  f"negatives <= {max_pct_negative}%",
        result=f"{len(pos_ok)}/{len([g for g in positive if g in obs])} positives present, "
               f"{len(neg_ok)}/{len([g for g in negative if g in obs])} negatives absent",
        decision=("marker profile consistent with the declared tissue"
                  if passed else "marker profile inconsistent - check tissue identity or annotation"),
        status="pass" if passed else "fail",
        caveats=[f"not in var_names: {', '.join(missing)}"] if missing else [],
        duration_s=round(time.time() - t0, 3),
    )


# --- spatial autocorrelation -----------------------------------------------
def _knn_weights(coords: np.ndarray, k: int):
    from scipy.sparse import csr_matrix
    from scipy.spatial import KDTree
    tree = KDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    n = len(coords)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    W = csr_matrix((np.ones(n * k), (rows, cols)), shape=(n, n))
    rs = np.asarray(W.sum(axis=1)).ravel()
    rs[rs == 0] = 1
    return W.multiply(1.0 / rs[:, None])


def _morans_i(x: np.ndarray, W, n_perm: int, rng) -> dict[str, float]:
    z = x - x.mean()
    denom = float(z @ z)
    if denom == 0:
        return {"I": 0.0, "z": 0.0, "p": 1.0}
    n, S0 = len(x), W.sum()
    I = (n / S0) * float(z @ W.dot(z)) / denom
    perms = np.empty(n_perm)
    for i in range(n_perm):
        zp = rng.permutation(z)
        perms[i] = (n / S0) * float(zp @ W.dot(zp)) / float(zp @ zp)
    sd = perms.std()
    zscore = (I - perms.mean()) / sd if sd > 0 else 0.0
    from scipy import stats as sps
    return {"I": round(float(I), 4), "z": round(float(zscore), 1),
            "p": float(2 * (1 - sps.norm.cdf(abs(zscore))))}


def spatial_autocorrelation(adata, markers: list[str], sample_key: str | None = None,
                            k: int = 6, n_perm: int = 999, threshold: float = 0.3,
                            min_passing: int = 3, seed: int = 42,
                            max_samples: int | None = None) -> DiagnosticRecord:
    """is expression spatially structured, or randomly distributed?

    THE decision that changes the pipeline's shape: it routes the molecular
    encoder to a spatial or a non-spatial model.

    computed per sample and reported as a median, because a cohort-level routing
    decision must not rest on one sample. `n_samples` is in the record so a reader
    can see what it rested on.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    if sample_key and sample_key in adata.obs:
        samples = list(dict.fromkeys(adata.obs[sample_key]))
    else:
        samples = [None]
    if max_samples:
        samples = samples[:max_samples]

    per_sample: dict[str, dict[str, float]] = {}
    for s in samples:
        sub = adata if s is None else adata[adata.obs[sample_key] == s]
        if sub.n_obs < k + 1:
            continue
        W = _knn_weights(np.asarray(sub.obsm["spatial"], dtype=float), k)
        for g in markers:
            if g not in sub.var_names:
                continue
            per_sample.setdefault(g, {})[str(s)] = _morans_i(
                _dense(sub[:, g].X).astype(float), W, n_perm, rng)["I"]

    observed = {
        g: {"median_I": round(float(np.median(list(v.values()))), 4),
            "n_samples": len(v),
            "min_I": round(float(min(v.values())), 4),
            "max_I": round(float(max(v.values())), 4)}
        for g, v in per_sample.items()
    }
    ranked = sorted(observed.items(), key=lambda kv: -kv[1]["median_I"])
    top = ranked[:5]
    n_pass = sum(1 for _, v in top if v["median_I"] > threshold)
    passed = n_pass >= min_passing
    n_samples_used = len(samples)

    caveats = []
    if n_samples_used == 1:
        caveats.append(
            "computed on a single sample - a cohort-level routing decision should not rest on n=1")
    if sample_key and sample_key in adata.obs and max_samples:
        total_samples = adata.obs[sample_key].nunique()
        if max_samples < total_samples:
            caveats.append(f"limited to {max_samples} of {total_samples} samples by max_samples")

    top_str = ", ".join(f"{g} {v['median_I']:.3f}" for g, v in top[:3])
    result = f"{n_pass} of {len(top)} above threshold" + (f" (top: {top_str})" if top else "")

    return DiagnosticRecord(
        step_id="spatial_autocorrelation",
        method=f"Moran's I, k-NN spatial weights (k={k}), {n_perm}-permutation null, "
               f"median across samples",
        params={"k": k, "n_perm": n_perm, "threshold": threshold,
                "min_passing": min_passing, "seed": seed},
        scope=f"{_fmt(adata.n_obs)} observations across {n_samples_used} "
              f"sample{'s' if n_samples_used != 1 else ''}, {len(observed)} markers tested",
        observed=observed,
        criterion=f"median I > {threshold} for >= {min_passing} of the top 5 markers",
        result=result,
        decision=("spatial structure present -> route to a spatial encoder"
                  if passed else
                  "no spatial structure -> route to a non-spatial encoder"),
        status="pass" if passed else "fail",
        caveats=caveats,
        duration_s=round(time.time() - t0, 3),
    )

# --- batch structure -------------------------------------------------------
def batch_structure(adata, sample_key: str,
                    sample_meta: dict[str, dict[str, Any]],
                    technical: list[str], biological: list[str],
                    n_pcs: int = 10, max_silhouette: float = 0.1) -> DiagnosticRecord:
    """does technical origin explain more variation than biology does?

    pseudobulk per sample, PCA, then silhouette by each declared variable. a
    positive technical silhouette that exceeds the biological one means batch
    dominates and must be addressed before alignment - not corrected reflexively,
    which is why the criterion is comparative rather than absolute.
    """
    t0 = time.time()
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    samples = list(dict.fromkeys(adata.obs[sample_key]))
    if len(samples) < 3:
        return DiagnosticRecord(
            step_id="batch_structure", status="not_run",
            method="pseudobulk PCA + silhouette by declared variable",
            scope=f"{len(samples)} samples",
            criterion=f"technical silhouette <= biological, or both <= {max_silhouette}",
            result="fewer than 3 samples - batch structure is not assessable",
            decision="cannot assess; do not conclude absence of batch effect",
            caveats=["absence of evidence is not evidence of absence here"],
            duration_s=round(time.time() - t0, 3))

    # pseudobulk: mean expression per sample
    pb = np.vstack([np.asarray(adata[adata.obs[sample_key] == s].X.mean(axis=0)).ravel()
                    for s in samples])
    pb = np.log1p(pb)
    n_comp = min(n_pcs, len(samples) - 1, pb.shape[1])
    coords = PCA(n_components=n_comp, random_state=42).fit_transform(pb)

    observed: dict[str, Any] = {}
    for group, variables in (("technical", technical), ("biological", biological)):
        for v in variables:
            labels = [sample_meta.get(s, {}).get(v) for s in samples]
            keep = [i for i, l in enumerate(labels) if l is not None]
            vals = [labels[i] for i in keep]
            if len(set(vals)) < 2 or len(keep) < 3:
                observed[v] = {"group": group, "silhouette": None,
                               "note": "too few groups or samples"}
                continue
            observed[v] = {"group": group,
                           "silhouette": round(float(silhouette_score(coords[keep], vals)), 4),
                           "n_groups": len(set(vals))}

    tech = [o["silhouette"] for o in observed.values()
            if o["group"] == "technical" and o["silhouette"] is not None]
    bio = [o["silhouette"] for o in observed.values()
           if o["group"] == "biological" and o["silhouette"] is not None]
    max_tech = max(tech) if tech else None
    max_bio = max(bio) if bio else None

    if max_tech is None:
        passed, result = False, "no technical variable was assessable"
    else:
        passed = max_tech <= max_silhouette or (max_bio is not None and max_tech <= max_bio)
        result = (f"max technical silhouette {max_tech:+.3f}"
                  + (f", max biological {max_bio:+.3f}" if max_bio is not None else ""))

    return DiagnosticRecord(
        step_id="batch_structure",
        method=f"pseudobulk per sample, log1p, PCA({n_comp}), silhouette by declared variable",
        params={"n_pcs": n_comp, "max_silhouette": max_silhouette,
                "technical": technical, "biological": biological},
        scope=f"{len(samples)} samples pseudobulked over {_fmt(adata.n_obs)} observations",
        observed=observed,
        criterion=f"technical silhouette <= biological, or <= {max_silhouette}",
        result=result,
        decision=("no batch correction needed - technical origin does not dominate"
                  if passed else
                  "batch dominates; address before alignment rather than proceeding"),
        status="pass" if passed else "fail",
        duration_s=round(time.time() - t0, 3),
    )


def spatial_autocorrelation_streamed(load_sample_fn, sample_ids: list,
                                     markers: list[str], k: int = 6, n_perm: int = 999,
                                     threshold: float = 0.3, min_passing: int = 3,
                                     seed: int = 42, progress=None) -> DiagnosticRecord:
    """cohort-wide Moran's I without holding the cohort in memory.

    each sample is loaded, measured and discarded, so memory is flat regardless of
    cohort size. this is the one check the EDA notebooks never ran across samples -
    every other cohort-level statistic loops all of them, but Moran's I was built
    from a single subarray's weights matrix, which is why eda_summary.json carries
    it as an open risk rather than a finding.

    `load_sample_fn(sample_id) -> AnnData` keeps the adapter injected, so this is
    not specific to any cohort.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    per_sample: dict[str, dict[str, float]] = {}
    failed: list[str] = []

    for n, sid in enumerate(sample_ids, 1):
        try:
            a = load_sample_fn(sid)
            if a.n_obs < k + 1:
                failed.append(f"{sid}: too few observations")
                continue
            W = _knn_weights(np.asarray(a.obsm["spatial"], dtype=float), k)
            for g in markers:
                if g in a.var_names:
                    per_sample.setdefault(g, {})[str(sid)] = _morans_i(
                        _dense(a[:, g].X).astype(float), W, n_perm, rng)["I"]
            del a
        except Exception as e:
            failed.append(f"{sid}: {type(e).__name__}")
        if progress:
            progress(n, len(sample_ids), sid)

    observed = {
        g: {"median_I": round(float(np.median(list(v.values()))), 4),
            "n_samples": len(v),
            "q25": round(float(np.percentile(list(v.values()), 25)), 4),
            "q75": round(float(np.percentile(list(v.values()), 75)), 4),
            "min_I": round(float(min(v.values())), 4),
            "max_I": round(float(max(v.values())), 4)}
        for g, v in per_sample.items()
    }
    ranked = sorted(observed.items(), key=lambda kv: -kv[1]["median_I"])
    top = ranked[:5]
    n_pass = sum(1 for _, v in top if v["median_I"] > threshold)
    passed = n_pass >= min_passing
    n_used = len(sample_ids) - len(failed)
    top_str = ", ".join(f"{g} {v['median_I']:.3f}" for g, v in top[:3])
    result = f"{n_pass} of {len(top)} above threshold" + (f" (top: {top_str})" if top else "")

    return DiagnosticRecord(
        step_id="spatial_autocorrelation",
        method=f"Moran's I, k-NN spatial weights (k={k}), {n_perm}-permutation null, "
               f"computed per sample and reported as the cohort median",
        params={"k": k, "n_perm": n_perm, "threshold": threshold,
                "min_passing": min_passing, "seed": seed, "streamed": True},
        scope=f"{n_used} of {len(sample_ids)} samples, {len(observed)} markers",
        observed=observed,
        criterion=f"median I > {threshold} for >= {min_passing} of the top 5 markers",
        result=result,
        decision=("spatial structure present -> route to a spatial encoder" if passed
                  else "no spatial structure -> route to a non-spatial encoder"),
        status="pass" if passed else "fail",
        caveats=([f"{len(failed)} sample(s) not measured"] if failed else []),
        duration_s=round(time.time() - t0, 1),
    )
