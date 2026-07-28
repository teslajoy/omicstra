"""
post-hoc 95% confidence intervals for two scorecard panels:
  H2-A · mc_megacluster KMeans ARI (z_he)
  H2-C · biology / patient ratio (TIME z / patient_id z, z_he)

methods:
  H2-A -> patient-level nonparametric bootstrap. KMeans(k=14, n_init=10,
          seed=42) fit once on the full test set per run; bootstrap B=200
          resamples of the 14 test patients with replacement, then
          adjusted_rand_score on the (true, pred) labels of the resampled
          niche subset. uncertainty estimate = ARI variability under the
          patient-resampling distribution, model held fixed.
  H2-C -> delta-method (Fieller-style) CI on the ratio of two
          permutation-derived z-scores using the existing biology.parquet
          null_std as the standardizer. closed-form:
              SE(ratio) ≈ |ratio| × sqrt(1/z_TIME² + 1/z_patient²)
          95% CI = ratio ± 1.96 × SE. no new sampling; uses the perm null
          already embedded in biology.parquet.

writes: runs/tnbc-92_v3/eval/H2/metrics_h2_ci.json
  with 'h2a' (per-run ARI + CI) and 'h2c' (per-run ratio + CI).

usage:
  python scripts/eval_h2_bootstrap_ci.py
  python scripts/eval_h2_bootstrap_ci.py --runs R4_v3 --n-bootstrap 500
"""

import json
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _as_arr(v):
    return np.asarray(v, dtype=np.float32)


def join_mc(run_df: pd.DataFrame, niches_dir: Path) -> pd.DataFrame:
    """attach mc_megacluster from niche-join (same pattern as eval_h2_mc_coherence.py)."""
    df = run_df.reset_index().copy()
    if "spot_id" not in df.columns:
        df["spot_id"] = run_df.index.values
    frames = []
    for sub in df["subarray"].unique():
        p = niches_dir / f"{sub}.parquet"
        if not p.exists():
            continue
        nf = pd.read_parquet(p).reset_index()
        nf["subarray"] = sub
        frames.append(nf[["subarray", "spot_id", "mc_megacluster"]])
    if not frames:
        raise FileNotFoundError(f"no niche-join parquets under {niches_dir}")
    nj = pd.concat(frames, ignore_index=True)
    return df.merge(nj, on=["subarray", "spot_id"], how="left")


def h2a_one_run(run_dir: Path, niches_dir: Path, n_boot: int, seed: int) -> dict:
    emb = run_dir / "embeddings_test.parquet"
    if not emb.exists():
        return None
    rd = pd.read_parquet(emb)
    merged = join_mc(rd, niches_dir)
    mask = merged["mc_megacluster"].notna()
    merged = merged[mask].copy()
    if len(merged) == 0:
        return None

    z_he = np.stack([_as_arr(v) for v in merged["z_he"]])
    y_true = merged["mc_megacluster"].astype(int).values
    patient_ids = merged["patient_id"].values
    unique_pats = np.unique(patient_ids)

    # fit KMeans once on the full test set (matches mc_coherence.parquet protocol)
    km = KMeans(n_clusters=14, n_init=10, random_state=42)
    y_pred = km.fit_predict(z_he)
    ari_full = float(adjusted_rand_score(y_true, y_pred))

    # patient-level bootstrap
    rng = np.random.default_rng(seed)
    pat_to_idx = {p: np.where(patient_ids == p)[0] for p in unique_pats}
    boots = np.empty(n_boot, dtype=np.float64)
    n_pat = len(unique_pats)
    for b in range(n_boot):
        sampled = rng.choice(unique_pats, size=n_pat, replace=True)
        # gather niche indices from sampled patients (with multiplicity)
        idx_parts = [pat_to_idx[p] for p in sampled]
        idx_b = np.concatenate(idx_parts)
        boots[b] = adjusted_rand_score(y_true[idx_b], y_pred[idx_b])
    return {
        "run": run_dir.name,
        "n_test_niches": int(len(y_true)),
        "n_test_patients": int(n_pat),
        "ari": ari_full,
        "ari_ci_low": float(np.percentile(boots, 2.5)),
        "ari_ci_high": float(np.percentile(boots, 97.5)),
        "n_bootstrap": int(n_boot),
        "method": f"patient_bootstrap_seed={seed}_KMeans_k14_n_init10",
    }


def h2c_one_run(run_dir: Path) -> dict:
    """delta-method CI on TIME_z / patient_id_z using existing biology.parquet."""
    bio = run_dir / "eval" / "biology.parquet"
    if not bio.exists():
        return None
    df = pd.read_parquet(bio)
    df = df[df["view"] == "z_he"].set_index("label")
    if "TIME" not in df.index or "patient_id" not in df.index:
        return None
    z_t = float(df.loc["TIME", "z"])
    z_p = float(df.loc["patient_id", "z"])
    ratio = z_t / z_p
    # delta-method: each z has approx Var=1 under H0 (perm-std-normalized),
    # so SE(z1/z2) ≈ |ratio| * sqrt(1/z1² + 1/z2²)
    se = abs(ratio) * np.sqrt(1.0 / (z_t * z_t) + 1.0 / (z_p * z_p))
    ci_lo = float(ratio - 1.96 * se)
    ci_hi = float(ratio + 1.96 * se)
    return {
        "run": run_dir.name,
        "z_TIME": z_t,
        "z_patient": z_p,
        "ratio": float(ratio),
        "ratio_ci_low": ci_lo,
        "ratio_ci_high": ci_hi,
        "ratio_se": float(se),
        "method": "delta_method_unit_variance_perm_null",
    }


DEFAULT_RUNS = "R1_v3,R2_v3,R3_v3,R4_v3,R5_v3,R6_v3,B1_v3,B2_v3,B3_v3,B4_v3"


@click.command()
@click.option("--runs-dir", default="runs/tnbc-92_v3", show_default=True)
@click.option("--niches-dir", default="data/embeddings/niches_v3", show_default=True)
@click.option("--runs", default=DEFAULT_RUNS, show_default=True)
@click.option("--n-bootstrap", default=200, show_default=True,
              help="patient bootstrap replicates for H2-A ARI.")
@click.option("--seed", default=42, show_default=True)
def main(runs_dir: str, niches_dir: str, runs: str, n_bootstrap: int, seed: int) -> None:
    runs_dir_p = Path(runs_dir)
    niches_dir_p = Path(niches_dir)
    out_path = runs_dir_p / "eval" / "H2" / "metrics_h2_ci.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h2a, h2c = [], []
    print(f"{'run':>8} | {'H2-A ARI [95% CI]':<28} | {'H2-C ratio [95% CI]':<24}")
    print("-" * 70)
    for run in (r.strip() for r in runs.split(",") if r.strip()):
        run_dir = runs_dir_p / run
        a = h2a_one_run(run_dir, niches_dir_p, n_bootstrap, seed)
        c = h2c_one_run(run_dir)
        if a is not None:
            h2a.append(a)
            ari_str = f"{a['ari']:.3f} [{a['ari_ci_low']:.3f}, {a['ari_ci_high']:.3f}]"
        else:
            ari_str = "n/a"
        if c is not None:
            h2c.append(c)
            rat_str = f"{c['ratio']:.3f} [{c['ratio_ci_low']:.3f}, {c['ratio_ci_high']:.3f}]"
        else:
            rat_str = "n/a"
        print(f"{run:>8} | {ari_str:<28} | {rat_str:<24}")

    out = {
        "h2a": {r["run"]: r for r in h2a},
        "h2c": {r["run"]: r for r in h2c},
        "seed": int(seed),
        "n_bootstrap_h2a": int(n_bootstrap),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()