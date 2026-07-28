"""
post-hoc 95% confidence intervals for H1 headline metrics (AUC, CKA).

reads each run's `embeddings_test.parquet`, recomputes AUC + CKA, and produces
a CI for each. does NOT retrain or touch align.py. writes per-run:
  - `metrics_h1_ci.json`   - scalars (AUC, CKA, CIs, methods)
  - `h1_eval_data.npz`     - raw arrays needed for downstream plotting

methods:
  AUC -> Hanley & McNeil (1982) closed-form SE  + nonparametric bootstrap on
         the (label, score) pairs (sanity check, plot-ready distribution).
  CKA -> nonparametric bootstrap over test niches (row resampling), B=200,
         seed=42. full-sample mean used for centering (sub-sample mean
         differs by O(1/sqrt(N)) for N=35,594 - negligible for CI width).

the .npz contains everything a pro bioinformatician plot needs:
  matched_cosines      (N,)         - diagonal of z_he @ z_st.T
  neg_sample_cosines   (~10N,)      - mismatched cosines sampled for AUC
  labels, scores       (11N,)       - feed direct to sklearn.roc_curve /
                                      precision_recall_curve
  auc_bootstrap        (B,)         - bootstrap AUC samples (for histogram /
                                      DeLong-equivalent comparison)
  cka_bootstrap        (B,)         - bootstrap CKA samples (for histogram)
  resample_seed        scalar       - reproducibility

supports: ROC curve overlays, PR curve overlays, cosine-distribution
histograms (matched vs mismatched), AUC + CKA forest plots with whiskers,
bootstrap CI validation plots, run-vs-run AUC delta tests.

per-run runtime: ~2-4 min. seed=42 throughout.

usage:
  python scripts/eval_h1_bootstrap_ci.py
  python scripts/eval_h1_bootstrap_ci.py --runs R4_v3 --n-bootstrap 500
"""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def hanley_mcneil_auc_ci(auc: float, n1: int, n0: int, alpha: float = 0.05):
    """closed-form 95% CI for ROC AUC. n1 = positives, n0 = negatives."""
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n1 - 1) * (q1 - auc * auc)
        + (n0 - 1) * (q2 - auc * auc)
    ) / (n1 * n0)
    se = float(np.sqrt(max(var, 0.0)))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return float(auc - z * se), float(auc + z * se), se


def cka_linear(Xc: np.ndarray, Yc: np.ndarray) -> float:
    """centered linear CKA (numerator weighted via Xc.T @ Yc, Frobenius)."""
    cross = Xc.T @ Yc
    num = float(np.sum(cross * cross))
    den = float(np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc))
    return num / (den + 1e-12)


def cka_bootstrap_fast(Xc: np.ndarray, Yc: np.ndarray, n_boot: int, seed: int):
    """B linear-CKA samples via weighted Gram products.

    bootstrap-resampling row indices is equivalent to multiplying each row by
    its multiplicity. so per replicate:
        A_b = Xc.T @ (w * Xc),  B_b = Xc.T @ (w * Yc),  C_b = Yc.T @ (w * Yc)
    where w = counts of each row in the resample. cost per replicate is
    3 matmuls of (D x N) @ (N x D) - tractable for D=512, N~35k.
    """
    rng = np.random.default_rng(seed)
    n = Xc.shape[0]
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # multiplicities = bincount on the resample index
        w = np.bincount(idx, minlength=n).astype(Xc.dtype)
        wX = Xc * w[:, None]
        A = Xc.T @ wX           # D x D, weighted Xc.T Xc
        B = Yc.T @ wX           # D x D, weighted Yc.T Xc -> ||.||_F = ||Xc.T Yc * w||
        C = Yc.T @ (Yc * w[:, None])
        num = float(np.sum(B * B))
        den = float(np.linalg.norm(A) * np.linalg.norm(C))
        out[b] = num / (den + 1e-12)
    return out


def evaluate_run(run_dir: Path, n_boot: int, seed: int) -> dict | None:
    emb = run_dir / "embeddings_test.parquet"
    if not emb.exists():
        return None
    df = pd.read_parquet(emb)
    z_he = np.stack(df["z_he"].to_numpy()).astype(np.float32)
    z_st = np.stack(df["z_st"].to_numpy()).astype(np.float32)
    n = z_he.shape[0]

    # --- AUC point estimate (replicates align.py:retrieval_metrics seed-0 neg sample) ---
    sim = z_he @ z_st.T
    matched = np.diag(sim).astype(np.float32).copy()
    mask = ~np.eye(n, dtype=bool)
    mismatched = sim[mask]
    rng = np.random.default_rng(0)
    neg = rng.choice(
        mismatched,
        size=min(len(matched) * 10, len(mismatched)),
        replace=False,
    ).astype(np.float32)
    del sim, mismatched  # ~5 GB freed
    labels = np.concatenate([np.ones(len(matched), dtype=np.int8),
                             np.zeros(len(neg), dtype=np.int8)])
    scores = np.concatenate([matched, neg])
    auc = float(roc_auc_score(labels, scores))
    n1, n0 = int(len(matched)), int(len(neg))
    auc_hm_lo, auc_hm_hi, auc_se = hanley_mcneil_auc_ci(auc, n1, n0)

    # --- AUC bootstrap (resample label/score pairs) ---
    boot_rng = np.random.default_rng(seed)
    auc_boots = np.empty(n_boot, dtype=np.float64)
    nlab = len(labels)
    for b in range(n_boot):
        idx = boot_rng.integers(0, nlab, size=nlab)
        auc_boots[b] = roc_auc_score(labels[idx], scores[idx])
    auc_boot_lo = float(np.percentile(auc_boots, 2.5))
    auc_boot_hi = float(np.percentile(auc_boots, 97.5))

    # --- CKA bootstrap (resample test niches) ---
    Xc = z_he - z_he.mean(0, keepdims=True)
    Yc = z_st - z_st.mean(0, keepdims=True)
    cka_point = cka_linear(Xc, Yc)
    cka_boots = cka_bootstrap_fast(Xc, Yc, n_boot=n_boot, seed=seed)
    cka_lo = float(np.percentile(cka_boots, 2.5))
    cka_hi = float(np.percentile(cka_boots, 97.5))

    # --- persist raw arrays for downstream plots ---
    np.savez_compressed(
        run_dir / "h1_eval_data.npz",
        matched_cosines=matched,
        neg_sample_cosines=neg,
        labels=labels,
        scores=scores,
        auc_bootstrap=auc_boots.astype(np.float32),
        cka_bootstrap=cka_boots.astype(np.float32),
        resample_seed=np.int32(seed),
    )

    return {
        "run": run_dir.name,
        "n_test": int(n),
        "auc": auc,
        "auc_ci_low": auc_hm_lo,
        "auc_ci_high": auc_hm_hi,
        "auc_se": auc_se,
        "auc_ci_method": "hanley_mcneil_1982",
        "auc_n_pos": n1,
        "auc_n_neg": n0,
        "auc_bootstrap_ci_low": auc_boot_lo,
        "auc_bootstrap_ci_high": auc_boot_hi,
        "cka": cka_point,
        "cka_ci_low": cka_lo,
        "cka_ci_high": cka_hi,
        "cka_n_bootstrap": int(n_boot),
        "cka_ci_method": f"nonparametric_bootstrap_seed={seed}",
        "plot_data": "h1_eval_data.npz",
    }


DEFAULT_RUNS = (
    "R1_v3,R2_v3,R3_v3,R4_v3,R5_v3,R6_v3,B1_v3,B2_v3,B3_v3,B4_v3"
)


@click.command()
@click.option("--runs-dir", default="runs/tnbc-92_v3", show_default=True)
@click.option("--runs", default=DEFAULT_RUNS, show_default=True)
@click.option("--n-bootstrap", default=200, show_default=True,
              help="bootstrap replicates for CKA. 200 = stable 2.5/97.5 percentiles.")
@click.option("--seed", default=42, show_default=True)
def main(runs_dir: str, runs: str, n_bootstrap: int, seed: int) -> None:
    root = Path(runs_dir)
    print(f"{'run':>8} | {'AUC [95% CI]':<24} | {'CKA [95% CI]':<24}")
    print("-" * 64)
    for run in (r.strip() for r in runs.split(",") if r.strip()):
        result = evaluate_run(root / run, n_boot=n_bootstrap, seed=seed)
        if result is None:
            print(f"{run:>8} | (no embeddings_test.parquet)")
            continue
        (root / run / "metrics_h1_ci.json").write_text(json.dumps(result, indent=2))
        auc_str = f"{result['auc']:.3f} [{result['auc_ci_low']:.3f}, {result['auc_ci_high']:.3f}]"
        cka_str = f"{result['cka']:.3f} [{result['cka_ci_low']:.3f}, {result['cka_ci_high']:.3f}]"
        print(f"{run:>8} | {auc_str:<24} | {cka_str:<24}")


if __name__ == "__main__":
    main()