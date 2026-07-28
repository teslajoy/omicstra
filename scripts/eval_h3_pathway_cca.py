"""commit 3 - H3 per-pathway univariate CCA on 5 named Reactome pathways.

proposal H3 (p.5): "Reactome pathway embeddings computed via gpath2vec for five
cancer-relevant pathways (TGF-beta Signaling, Immune System, Extracellular Matrix
Organization, Cell Cycle, Programmed Cell Death) will correlate with specific
directions in the shared latent space via canonical correlation analysis"

implementation:
  - pathway signal: AUCell scores at niche level from
    data/embeddings/biological_signals/niche_aucell_5targets.parquet
    (286,250 niches, 100% coverage of the 5 pathways, all niches matched)
  - per (run, view, pathway): univariate CCA reduces to
        w        = lstsq(Z_centered, y_centered)          # canonical direction
        z_proj   = Z_centered @ w / ||w||
        corr     = pearsonr(z_proj, y_centered)
    closed-form, no sklearn.cross_decomposition.CCA machinery needed.

  - two CCA variants per cell:
      option B: fit on all test niches (proposal intent, matches existing
                cca_top_k(z_mean, gpath2vec_512d) protocol)
      option A: within-test patient sub-split 11/3, fit on 11p, eval on 3p
                (cross-patient generalization)

grid: 8 runs (R1,R2,R3,R4,R6,B1,B2,B3) x 3 views (z_he, z_st, z_mean) x 5 pathways
    = 120 cells, per option => 240 correlations.

pre-registered predictions: see memory `project_h3_pathway_cca_predictions.md`.
  R4 primary contender for Immune + TGF-beta; R1 strongest linear decoder;
  R3 floor; B1 wins cumulative correlation but loses per-pathway.

writes:
  runs/tnbc-92/eval/H3/pathway_cca/per_pathway_cca.parquet  (240 rows long-form)
  runs/tnbc-92/eval/H3/pathway_cca/canonical_directions.parquet  (120 x 512-d)
  runs/tnbc-92/eval/H3/pathway_cca/specificity_matrix.parquet  (5x5 per run/view)
"""

from __future__ import annotations

import json
import warnings
import click
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
# mutable defaults; @click.option in main() overrides at run time so v3 can
# target runs/tnbc-92_v3/ + a different out-dir without code changes.
RUNS_ROOT = ROOT / 'runs' / 'tnbc-92'
AUCELL_PATH = ROOT / 'data/embeddings/biological_signals/niche_aucell_5targets.parquet'
OUT_DIR = RUNS_ROOT / 'eval' / 'H3' / 'pathway_cca'

DEFAULT_RUNS = ['R1', 'R2', 'R3', 'R4', 'R6', 'B1', 'B2', 'B3']
RUNS = DEFAULT_RUNS  # mutable; main() reassigns from --runs
VIEWS = ['z_he', 'z_st', 'z_mean']
SEED = 42

PATHWAYS = {
    'TGF-beta_Signaling': 'R-HSA-170834',
    'Immune_System': 'R-HSA-168256',
    'ECM_Organization': 'R-HSA-1474244',
    'Cell_Cycle': 'R-HSA-1640170',
    'Programmed_Cell_Death': 'R-HSA-5357801',
}


def _as_arr(cell) -> np.ndarray:
    return np.asarray(cell, dtype=np.float32)


def stack(col_series) -> np.ndarray:
    return np.stack([_as_arr(v) for v in col_series])


def load_run(run_id: str) -> pd.DataFrame:
    df = pd.read_parquet(RUNS_ROOT / run_id / 'embeddings_test.parquet').reset_index()
    if 'spot_id' not in df.columns:
        # try index name
        raw = pd.read_parquet(RUNS_ROOT / run_id / 'embeddings_test.parquet')
        df = raw.reset_index()
    df['niche_id'] = df['subarray'].astype(str) + '::' + df['spot_id'].astype(str)
    return df


def get_view(df: pd.DataFrame, view: str) -> np.ndarray:
    if view == 'z_mean':
        z_he = stack(df['z_he'])
        z_st = stack(df['z_st'])
        return 0.5 * (z_he + z_st)
    return stack(df[view])


def univariate_cca(Z: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """univariate CCA: y is 1-d. returns (corr, canonical_direction_unit_norm).

    derivation: maximize corr(Zw, y) over w. closed-form:
      w_unscaled = (Z^T Z)^-1 Z^T y    (lstsq normal equations)
      w = w_unscaled / ||w_unscaled||  (unit-norm)
      z_proj = Z @ w
      corr = pearsonr(z_proj, y)
    """
    Zc = Z - Z.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    # lstsq with rcond=None uses default singular-value cutoff
    w, *_ = np.linalg.lstsq(Zc.astype(np.float64), yc.astype(np.float64), rcond=None)
    norm = float(np.linalg.norm(w))
    if norm < 1e-12:
        return float('nan'), np.zeros_like(w)
    w = w / norm
    z_proj = Zc @ w
    r, _ = pearsonr(z_proj, yc)
    return float(r), w


def patient_sub_split(patients: np.ndarray, n_test_patients: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """within-test patient sub-split. returns (train_mask, test_mask)."""
    unique_pats = np.unique(patients)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_pats)
    test_pats = set(unique_pats[:n_test_patients].tolist())
    train_pats = set(unique_pats[n_test_patients:].tolist())
    return np.isin(patients, list(train_pats)), np.isin(patients, list(test_pats))


def run_grid(aucell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    directions = []
    specificity_rows = []

    for run_id in RUNS:
        print(f'\n=== {run_id} ===')
        df = load_run(run_id)
        merged = df.merge(aucell, left_on='niche_id', right_index=True, how='left',
                          suffixes=('', '_au'))
        # keep niches with non-null pathway scores (all 5)
        path_cols = list(PATHWAYS.keys())
        ok = merged[path_cols].notna().all(axis=1)
        merged = merged[ok].reset_index(drop=True)
        patients = merged['patient_id'].astype(int).values
        train_mask, test_mask = patient_sub_split(patients, n_test_patients=3)
        n_full = len(merged)
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        # cache per-view embeddings
        views_data = {v: get_view(merged, v) for v in VIEWS}

        for view in VIEWS:
            Z = views_data[view]
            run_directions = {}  # pathway_name -> w (for specificity matrix)

            for pname, pid in PATHWAYS.items():
                y = merged[pname].astype(np.float64).values

                # option B: fit on all
                corr_B, w_B = univariate_cca(Z, y)

                # option A: fit on train sub-split, eval on test sub-split
                Z_tr, y_tr = Z[train_mask], y[train_mask]
                Z_te, y_te = Z[test_mask], y[test_mask]
                # fit canonical direction on train
                _, w_tr = univariate_cca(Z_tr, y_tr)
                # apply to test - mean-center test using TEST means (sigma-free w)
                Zc_te = Z_te - Z_te.mean(axis=0, keepdims=True)
                yc_te = y_te - y_te.mean()
                z_proj_te = Zc_te @ w_tr
                if len(yc_te) > 1 and np.std(z_proj_te) > 1e-12:
                    corr_A_test, _ = pearsonr(z_proj_te, yc_te)
                else:
                    corr_A_test = float('nan')
                # also report option A train-side correlation
                Zc_tr = Z_tr - Z_tr.mean(axis=0, keepdims=True)
                yc_tr = y_tr - y_tr.mean()
                z_proj_tr = Zc_tr @ w_tr
                if np.std(z_proj_tr) > 1e-12:
                    corr_A_train, _ = pearsonr(z_proj_tr, yc_tr)
                else:
                    corr_A_train = float('nan')

                rows.append({
                    'run_id': run_id, 'view': view,
                    'pathway_name': pname, 'pathway_id': pid,
                    'corr_full': corr_B,        # option B (proposal intent)
                    'corr_train': corr_A_train,  # option A train-side fit-correlation
                    'corr_test': corr_A_test,    # option A test-side generalization
                    'n_full': n_full, 'n_train': n_train, 'n_test': n_test,
                })
                directions.append({
                    'run_id': run_id, 'view': view, 'pathway_name': pname,
                    'direction': w_B.astype(np.float32),
                })
                run_directions[pname] = w_B

            # specificity matrix: 5x5 pairwise cosine between pathway canonical directions
            W = np.stack([run_directions[p] for p in PATHWAYS.keys()])
            # rows are unit-norm already; cosine = dot product
            S = W @ W.T
            for i, p_i in enumerate(PATHWAYS.keys()):
                for j, p_j in enumerate(PATHWAYS.keys()):
                    specificity_rows.append({
                        'run_id': run_id, 'view': view,
                        'pathway_i': p_i, 'pathway_j': p_j,
                        'cosine': float(S[i, j]),
                    })
            off_diag = S[np.triu_indices(5, k=1)]
            print(f'  {view}: 5 pathways done. specificity off-diag '
                  f'mean=|{np.abs(off_diag).mean():.3f}| max=|{np.abs(off_diag).max():.3f}|')

    return pd.DataFrame(rows), pd.DataFrame(directions), pd.DataFrame(specificity_rows)


@click.command()
@click.option('--runs-dir', type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help='parent dir holding {run_id}/embeddings_test.parquet. defaults to runs/tnbc-92/.')
@click.option('--out-dir', type=click.Path(file_okay=False, path_type=Path), default=None,
              help='where per_pathway_cca.parquet + specificity_matrix.parquet land. '
                   'defaults to {runs-dir}/eval/H3/pathway_cca.')
@click.option('--aucell-path', type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help='per-niche AUCell scores parquet. defaults to '
                   'data/embeddings/biological_signals/niche_aucell_5targets.parquet.')
@click.option('--runs', 'runs_arg', default=None,
              help='comma-separated run ids (e.g. R1_v3,R2_v3,...). defaults to v1 8-run set.')
def main(runs_dir, out_dir, aucell_path, runs_arg):
    global RUNS_ROOT, OUT_DIR, AUCELL_PATH, RUNS

    if runs_dir is not None:
        RUNS_ROOT = runs_dir.resolve()
    if aucell_path is not None:
        AUCELL_PATH = aucell_path.resolve()
    if out_dir is not None:
        OUT_DIR = out_dir.resolve()
    else:
        OUT_DIR = RUNS_ROOT / 'eval' / 'H3' / 'pathway_cca'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if runs_arg is not None:
        RUNS = [r.strip() for r in runs_arg.split(',') if r.strip()]

    print(f'runs_dir: {RUNS_ROOT}')
    print(f'out_dir:  {OUT_DIR}')
    print(f'aucell:   {AUCELL_PATH}')
    print(f'runs:     {RUNS}')
    print('\nloading AUCell...')
    aucell = pd.read_parquet(AUCELL_PATH)
    print(f'  {len(aucell):,} niches x {len(PATHWAYS)} pathways')

    print('\nrunning specificity sanity check on R4 and B1 first (z_he)...')
    # the sanity is folded into the main grid; we will inspect after

    df_main, df_dir, df_spec = run_grid(aucell)

    df_main.to_parquet(OUT_DIR / 'per_pathway_cca.parquet', index=False)
    print(f'\nwrote {OUT_DIR / "per_pathway_cca.parquet"}  ({len(df_main)} rows)')

    df_dir.to_parquet(OUT_DIR / 'canonical_directions.parquet', index=False)
    print(f'wrote {OUT_DIR / "canonical_directions.parquet"}  ({len(df_dir)} directions)')

    df_spec.to_parquet(OUT_DIR / 'specificity_matrix.parquet', index=False)
    print(f'wrote {OUT_DIR / "specificity_matrix.parquet"}  ({len(df_spec)} pairs)')

    # ===== summary tables =====
    print('\n\n========================================')
    print(' option B (fit on all test niches) - the proposal-intent result')
    print('========================================')
    print('\nz_he correlations:')
    pivot_he = df_main[df_main.view == 'z_he'].pivot(
        index='run_id', columns='pathway_name', values='corr_full')
    print(pivot_he.reindex(RUNS)[list(PATHWAYS.keys())].round(3).to_string())

    print('\nz_st correlations:')
    pivot_st = df_main[df_main.view == 'z_st'].pivot(
        index='run_id', columns='pathway_name', values='corr_full')
    print(pivot_st.reindex(RUNS)[list(PATHWAYS.keys())].round(3).to_string())

    print('\nz_mean correlations:')
    pivot_m = df_main[df_main.view == 'z_mean'].pivot(
        index='run_id', columns='pathway_name', values='corr_full')
    print(pivot_m.reindex(RUNS)[list(PATHWAYS.keys())].round(3).to_string())

    # winners per pathway per view
    print('\n\nwinners per pathway (z_he, option B):')
    for p in PATHWAYS.keys():
        s = pivot_he[p].abs()
        winner = s.idxmax()
        # top-3 runs by |corr| - run-id-agnostic (works for v1 R1.. and v3 R1_v3..)
        top3 = s.sort_values(ascending=False).head(3)
        detail = ', '.join(f'{rid}={pivot_he.loc[rid, p]:+.3f}' for rid in top3.index)
        print(f'  {p:<25} winner={winner}  |corr|={s.max():.3f}  ({detail})')

    # cumulative correlation across 5 pathways per run (sum of |corr|)
    print('\ncumulative |corr| across 5 pathways (z_he):')
    cum = pivot_he.abs().sum(axis=1).sort_values(ascending=False)
    print(cum.round(3).to_string())

    # option A cross-patient generalization summary
    print('\n========================================')
    print(' option A (cross-patient sub-split) - generalization test')
    print('========================================\n')
    pivot_A_he = df_main[df_main.view == 'z_he'].pivot(
        index='run_id', columns='pathway_name', values='corr_test')
    print('z_he, option-A held-out test correlations:')
    print(pivot_A_he.reindex(RUNS)[list(PATHWAYS.keys())].round(3).to_string())

    # option A vs option B divergence flag
    print('\noption A vs option B agreement (correlation between the two corr columns, per run):')
    for run_id in RUNS:
        run_subset = df_main[df_main.run_id == run_id]
        r_AB = run_subset[['corr_full', 'corr_test']].corr().iloc[0, 1]
        print(f'  {run_id}: r(option_B, option_A_test) = {r_AB:.3f}')

    # specificity off-diagonal summary (z_he)
    print('\n========================================')
    print(' specificity matrix off-diagonal |mean| (z_he)')
    print(' low values = distinct pathway directions; high values = aliased')
    print('========================================\n')
    spec_he = df_spec[df_spec.view == 'z_he']
    for run_id in RUNS:
        sub = spec_he[spec_he.run_id == run_id]
        S = sub.pivot(index='pathway_i', columns='pathway_j', values='cosine').values
        off = S[np.triu_indices(5, k=1)]
        print(f'  {run_id}: off-diag |mean| = {np.abs(off).mean():.3f}  '
              f'|max| = {np.abs(off).max():.3f}  range = '
              f'[{off.min():+.3f}, {off.max():+.3f}]')


if __name__ == '__main__':
    main()