"""commit 3.5 - permutation null + BH-FDR calibration for H3 per-pathway CCA.

calibrates the option B / option A correlations from eval_h3_pathway_cca.py.
load-bearing because option B correlations are 0.70-0.90 across ALL runs and
pathways - that's overfit-noise capacity of a 512-d projection onto a scalar.
the null calibration is what distinguishes biological signal from regression
overfit.

fast perm protocol:
  - univariate CCA correlation = sqrt(R^2) of OLS regression of y on Z
  - R^2 = ||Q^T y_centered||^2 / ||y_centered||^2  where Z = QR (skinny QR)
  - per-perm work: 1 matmul Q^T y_shuf (~5ms for 45k x 512), ~1000 perms ~5-10s
  - amortize Q across 5 pathways within (run, view)

option B null: shuffle pathway score across all niches; refit; correlation.
option A null: shuffle pathway score within TRAIN sub-split; refit on shuffled
               train; evaluate on held-out test (test labels NOT shuffled).
               this calibrates "did training on real pathway labels produce a
               direction that transfers, vs training on random labels?"

writes:
  runs/tnbc-92/eval/H3/pathway_cca/perm_nulls.parquet  (240 rows; null mean/std + p_one_sided + p_two_sided + BH-FDR)
  runs/tnbc-92/eval/H3/pathway_cca/fdr_table.parquet   (per-option flat table with significance flags)
"""

from __future__ import annotations

import warnings
import click
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
# mutable defaults; @click.option in main() overrides for the v3 retrain.
RUNS_ROOT = ROOT / 'runs' / 'tnbc-92'
AUCELL_PATH = ROOT / 'data/embeddings/biological_signals/niche_aucell_5targets.parquet'
H3_DIR = RUNS_ROOT / 'eval' / 'H3' / 'pathway_cca'

DEFAULT_RUNS = ['R1', 'R2', 'R3', 'R4', 'R6', 'B1', 'B2', 'B3']
RUNS = DEFAULT_RUNS  # mutable; main() reassigns from --runs
VIEWS = ['z_he', 'z_st', 'z_mean']
PATHWAYS = ['TGF-beta_Signaling', 'Immune_System', 'ECM_Organization',
            'Cell_Cycle', 'Programmed_Cell_Death']
SEED = 42
N_PERMS = 1000


def _as_arr(cell) -> np.ndarray:
    return np.asarray(cell, dtype=np.float32)


def stack(col_series) -> np.ndarray:
    return np.stack([_as_arr(v) for v in col_series])


def load_run(run_id: str) -> pd.DataFrame:
    df = pd.read_parquet(RUNS_ROOT / run_id / 'embeddings_test.parquet').reset_index()
    df['niche_id'] = df['subarray'].astype(str) + '::' + df['spot_id'].astype(str)
    return df


def get_view(df: pd.DataFrame, view: str) -> np.ndarray:
    if view == 'z_mean':
        return 0.5 * (stack(df['z_he']) + stack(df['z_st']))
    return stack(df[view])


def patient_sub_split(patients: np.ndarray, n_test_patients: int = 3) -> tuple[np.ndarray, np.ndarray]:
    unique_pats = np.unique(patients)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_pats)
    test_pats = set(unique_pats[:n_test_patients].tolist())
    train_pats = set(unique_pats[n_test_patients:].tolist())
    return np.isin(patients, list(train_pats)), np.isin(patients, list(test_pats))


def ols_correlation(Q: np.ndarray, y_centered: np.ndarray) -> float:
    """correlation between OLS projection of y on col(Q) and y (Q has orthonormal cols)."""
    yn = np.linalg.norm(y_centered)
    if yn < 1e-12:
        return float('nan')
    r2 = float((Q.T @ y_centered).T @ (Q.T @ y_centered) / (yn * yn))
    return float(np.sqrt(max(0.0, r2)))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR. handles NaN by skipping."""
    p = np.asarray(pvals, dtype=np.float64)
    valid = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if valid.sum() == 0:
        return out
    n = int(valid.sum())
    order = np.argsort(p[valid])
    ranked = np.arange(1, n + 1)
    adj = p[valid][order] * n / ranked
    # cumulative minimum from the right
    for i in range(n - 2, -1, -1):
        if adj[i + 1] < adj[i]:
            adj[i] = adj[i + 1]
    adj = np.minimum(adj, 1.0)
    out_valid = np.empty(n)
    out_valid[order] = adj
    out[valid] = out_valid
    return out


def perm_option_B(Z: np.ndarray, y: np.ndarray, n_perms: int) -> np.ndarray:
    """null correlations from shuffling y across all rows; refit-via-Q each perm."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Q, _ = np.linalg.qr(Zc.astype(np.float64), mode='reduced')
    yc = y.astype(np.float64) - y.mean()
    yn = np.linalg.norm(yc)
    rng = np.random.default_rng(SEED)
    nulls = np.empty(n_perms)
    for i in range(n_perms):
        y_shuf = rng.permutation(yc)
        # |Q^T y_shuf| / ||y_shuf||
        qy = Q.T @ y_shuf
        r2 = float(qy @ qy) / float(yn * yn)
        nulls[i] = np.sqrt(max(0.0, r2))
    return nulls


def precompute_view_caches(Z: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> dict:
    """one-time-per-(run,view) caches reused across all 5 pathways and all perms."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Q_full, _ = np.linalg.qr(Zc.astype(np.float64), mode='reduced')

    Z_tr = Z[train_mask]; Z_te = Z[test_mask]
    Z_tr_c = Z_tr - Z_tr.mean(axis=0, keepdims=True)
    Z_te_c = Z_te - Z_te.mean(axis=0, keepdims=True)
    # cache pseudo-inverse once: shape (d, n_tr); reused 5 pathways x 1000 perms = 5000 times
    Z_tr_pinv = np.linalg.pinv(Z_tr_c.astype(np.float64))
    return {
        'Q_full': Q_full,
        'Z_tr_c': Z_tr_c,
        'Z_te_c': Z_te_c,
        'Z_tr_pinv': Z_tr_pinv,
    }


def perm_option_A_cached(cache: dict, y_tr: np.ndarray, y_te: np.ndarray, n_perms: int) -> np.ndarray:
    """null correlations from shuffling y in TRAIN; reuses cached Z_tr_pinv and Z_te_c."""
    Z_tr_pinv = cache['Z_tr_pinv']
    Z_te_c = cache['Z_te_c']
    y_te_c = y_te.astype(np.float64) - y_te.mean()
    yn_te = float(np.linalg.norm(y_te_c))
    if yn_te < 1e-12:
        return np.full(n_perms, np.nan)
    y_tr_c = y_tr.astype(np.float64) - y_tr.mean()
    rng = np.random.default_rng(SEED + 1)
    nulls = np.empty(n_perms)
    for i in range(n_perms):
        y_tr_shuf = rng.permutation(y_tr_c)
        w_shuf = Z_tr_pinv @ y_tr_shuf
        norm = float(np.linalg.norm(w_shuf))
        if norm < 1e-12:
            nulls[i] = float('nan')
            continue
        w_shuf = w_shuf / norm
        z_proj_te = Z_te_c @ w_shuf
        zn = float(np.linalg.norm(z_proj_te))
        if zn < 1e-12:
            nulls[i] = float('nan')
            continue
        nulls[i] = float(z_proj_te @ y_te_c) / (zn * yn_te)
    return nulls


@click.command()
@click.option('--runs-dir', type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help='parent dir holding {run_id}/embeddings_test.parquet. defaults to runs/tnbc-92/.')
@click.option('--h3-dir', type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help='dir holding per_pathway_cca.parquet (output of eval_h3_pathway_cca.py); '
                   'perm_nulls.parquet lands here too. defaults to {runs-dir}/eval/H3/pathway_cca.')
@click.option('--aucell-path', type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help='per-niche AUCell scores parquet.')
@click.option('--runs', 'runs_arg', default=None,
              help='comma-separated run ids (e.g. R1_v3,R2_v3,...). defaults to v1 8-run set.')
@click.option('--n-perms', type=int, default=N_PERMS, show_default=True,
              help='permutations per cell.')
def main(runs_dir, h3_dir, aucell_path, runs_arg, n_perms):
    global RUNS_ROOT, H3_DIR, AUCELL_PATH, RUNS, N_PERMS

    if runs_dir is not None:
        RUNS_ROOT = runs_dir.resolve()
    if aucell_path is not None:
        AUCELL_PATH = aucell_path.resolve()
    H3_DIR = h3_dir.resolve() if h3_dir is not None else RUNS_ROOT / 'eval' / 'H3' / 'pathway_cca'
    if runs_arg is not None:
        RUNS = [r.strip() for r in runs_arg.split(',') if r.strip()]
    N_PERMS = n_perms

    print(f'runs_dir: {RUNS_ROOT}')
    print(f'h3_dir:   {H3_DIR}')
    print(f'runs:     {RUNS}')
    print(f'n_perms:  {N_PERMS}')
    print('\nloading observed correlations from commit 3...')
    obs = pd.read_parquet(H3_DIR / 'per_pathway_cca.parquet')
    print(f'  {len(obs)} observed correlations')

    print('loading AUCell...')
    aucell = pd.read_parquet(AUCELL_PATH)

    print(f'\nrunning {N_PERMS} perms per cell (8 runs x 3 views x 5 pathways x 2 options)...\n')
    perm_rows = []
    for run_id in RUNS:
        print(f'  {run_id}: load + merge...')
        df = load_run(run_id)
        merged = df.merge(aucell, left_on='niche_id', right_index=True, how='left',
                          suffixes=('', '_au'))
        ok = merged[PATHWAYS].notna().all(axis=1)
        merged = merged[ok].reset_index(drop=True)
        patients = merged['patient_id'].astype(int).values
        train_mask, test_mask = patient_sub_split(patients, n_test_patients=3)

        for view in VIEWS:
            Z = get_view(merged, view)
            print(f'    {view}: precompute Q + Z_tr_pinv (one-time per view)...', flush=True)
            cache = precompute_view_caches(Z, train_mask, test_mask)
            Q_full = cache['Q_full']

            for pname in PATHWAYS:
                y = merged[pname].astype(np.float64).values

                # option B null - fast via Q
                yc = y - y.mean()
                rng = np.random.default_rng(SEED)
                yn = float(np.linalg.norm(yc))
                nulls_B = np.empty(N_PERMS)
                for i in range(N_PERMS):
                    y_shuf = rng.permutation(yc)
                    qy = Q_full.T @ y_shuf
                    r2 = float(qy @ qy) / (yn * yn)
                    nulls_B[i] = float(np.sqrt(max(0.0, r2)))

                # option A null - cached pinv
                y_tr = y[train_mask]; y_te = y[test_mask]
                nulls_A = perm_option_A_cached(cache, y_tr, y_te, N_PERMS)

                # fetch observed
                obs_row = obs[(obs.run_id == run_id) & (obs.view == view) &
                              (obs.pathway_name == pname)].iloc[0]
                obs_B = float(obs_row['corr_full'])
                obs_A = float(obs_row['corr_test'])

                # one-sided p (signed): proportion of perm >= observed (for positive obs)
                # two-sided: |perm| >= |obs|
                obs_B_sign = np.sign(obs_B) if not np.isnan(obs_B) else 0
                obs_A_sign = np.sign(obs_A) if not np.isnan(obs_A) else 0

                # nulls_B is always positive (sqrt of R^2); option B observed is too
                # (lstsq + project), so one-sided is the right test for option B
                p_one_B = float((nulls_B >= obs_B).mean()) if not np.isnan(obs_B) else float('nan')

                # option A nulls and observed can be signed
                valid_A = ~np.isnan(nulls_A)
                if valid_A.sum() > 0 and not np.isnan(obs_A):
                    p_one_A = float((nulls_A[valid_A] >= obs_A).mean())
                    p_two_A = float((np.abs(nulls_A[valid_A]) >= abs(obs_A)).mean())
                else:
                    p_one_A = p_two_A = float('nan')

                perm_rows.append({
                    'run_id': run_id, 'view': view, 'pathway_name': pname,
                    'obs_B': obs_B, 'obs_A': obs_A,
                    'null_B_mean': float(np.mean(nulls_B)),
                    'null_B_std': float(np.std(nulls_B)),
                    'null_B_p99': float(np.quantile(nulls_B, 0.99)),
                    'null_A_mean': float(np.mean(nulls_A[valid_A])) if valid_A.sum() else float('nan'),
                    'null_A_std': float(np.std(nulls_A[valid_A])) if valid_A.sum() else float('nan'),
                    'p_one_B': p_one_B,
                    'p_one_A': p_one_A, 'p_two_A': p_two_A,
                    'z_B': (obs_B - float(np.mean(nulls_B))) / float(np.std(nulls_B) + 1e-12)
                            if not np.isnan(obs_B) else float('nan'),
                    'z_A': (obs_A - float(np.mean(nulls_A[valid_A]))) /
                            float(np.std(nulls_A[valid_A]) + 1e-12)
                            if valid_A.sum() and not np.isnan(obs_A) else float('nan'),
                })
            print(f'    {view}: 5 pathways x 2 options x {N_PERMS} perms done.')

    df = pd.DataFrame(perm_rows)

    # BH-FDR across 120 cells per option
    df['fdr_B'] = bh_fdr(df['p_one_B'].values)
    df['fdr_A_one'] = bh_fdr(df['p_one_A'].values)
    df['fdr_A_two'] = bh_fdr(df['p_two_A'].values)

    df['sig_B_05'] = df['fdr_B'] < 0.05
    df['sig_A_one_05'] = df['fdr_A_one'] < 0.05
    df['sig_A_two_05'] = df['fdr_A_two'] < 0.05

    df.to_parquet(H3_DIR / 'perm_nulls.parquet', index=False)
    print(f'\nwrote {H3_DIR / "perm_nulls.parquet"}  ({len(df)} rows)')

    # ===== summary =====
    print('\n\n========================================')
    print(' OPTION B: observed vs null (regression capacity calibration)')
    print('========================================')
    print('\nz_he, observed vs null_B_mean (table is obs - null_mean = lift over chance fit):')
    he = df[df.view == 'z_he'].copy()
    he['lift_B'] = he['obs_B'] - he['null_B_mean']
    pv = he.pivot(index='run_id', columns='pathway_name', values='lift_B')
    print(pv.reindex(RUNS)[PATHWAYS].round(3).to_string())

    print('\nz_he, z_B (observed minus null_mean, in null-std units):')
    pv = he.pivot(index='run_id', columns='pathway_name', values='z_B')
    print(pv.reindex(RUNS)[PATHWAYS].round(1).to_string())

    print('\nz_he, BH-FDR option B (only cells with FDR < 0.05 marked):')
    sig_b = he[he.sig_B_05][['run_id', 'pathway_name', 'obs_B', 'null_B_mean', 'z_B', 'fdr_B']]
    print(sig_b.round(3).to_string(index=False) if len(sig_b) else 'NONE significant after BH-FDR')

    print('\n\n========================================')
    print(' OPTION A: held-out test correlation vs null (THE generalization test)')
    print('========================================')
    print('\nz_he, observed test corr (option A obs):')
    pv = he.pivot(index='run_id', columns='pathway_name', values='obs_A')
    print(pv.reindex(RUNS)[PATHWAYS].round(3).to_string())

    print('\nz_he, z_A (observed minus null_mean, in null-std units):')
    pv = he.pivot(index='run_id', columns='pathway_name', values='z_A')
    print(pv.reindex(RUNS)[PATHWAYS].round(1).to_string())

    print('\nz_he, BH-FDR option A two-sided (cells with FDR < 0.05):')
    sig_a = he[he.sig_A_two_05][['run_id', 'pathway_name', 'obs_A',
                                  'null_A_mean', 'z_A', 'fdr_A_two']]
    print(sig_a.round(3).to_string(index=False) if len(sig_a) else 'NONE significant after BH-FDR')

    # cumulative significant cells per run on option A
    print('\nn cells significant by option A (BH-FDR<0.05) per run, summed across views x pathways:')
    sig_pivot = df.groupby('run_id')['sig_A_two_05'].sum().sort_values(ascending=False)
    print(sig_pivot.to_string())


if __name__ == '__main__':
    main()