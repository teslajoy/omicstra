"""H2 niche-level MC coherence: KMeans(k=14) -> ARI + silhouette vs
mc_labels.megacluster (Wang's per-spot 14-class NMF discrete labels).

per (run, view in {z_he, z_st, z_joint}): cluster_quality on niches with
non-null mc_megacluster.

reference: raw_he (virchow2_niche) and raw_st (novae_niche concat gpath2vec_niche)
on the same niche set used by R1 (assumed scope-defining).

writes runs/tnbc-92/eval/H2/mc_coherence.parquet + appends to summary.json
under per_run[run].mc_*.
"""

from __future__ import annotations

import json
import click
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
# mutable defaults; main() overrides from --runs-dir / --niches-dir / --runs
RUNS_ROOT = ROOT / 'runs' / 'tnbc-92'
NICHES_DIR = ROOT / 'data' / 'embeddings' / 'niches'
H2_DIR = RUNS_ROOT / 'eval' / 'H2'

N_MC = 14
SEED = 42
DEFAULT_RUNS = ['R1', 'R2', 'R3', 'R4', 'R6', 'B1', 'B2', 'B3']
RUNS = DEFAULT_RUNS
RAW_REF_RUN = 'R1'  # mutable; main() overrides from --raw-reference-run or RUNS[0]


def _as_arr(cell) -> np.ndarray:
    return np.asarray(cell, dtype=np.float32)


def cluster_quality(X: np.ndarray, y: np.ndarray, k: int = N_MC, sil_sample: int = 5000) -> dict:
    if len(np.unique(y)) < 2 or len(y) < k:
        return {'ari': float('nan'), 'silhouette': float('nan'), 'n': int(len(y))}
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    preds = km.fit_predict(X)
    ari = float(adjusted_rand_score(y, preds))
    n_sil = min(sil_sample, len(y))
    idx = np.random.default_rng(SEED).choice(len(y), n_sil, replace=False)
    sil = float(silhouette_score(X[idx], y[idx], sample_size=n_sil, random_state=SEED))
    return {'ari': ari, 'silhouette': sil, 'n': int(len(y))}


def load_run(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(RUNS_ROOT / run_id / 'embeddings_test.parquet')


def join_mc(run_df: pd.DataFrame) -> pd.DataFrame:
    """attach mc_megacluster + virchow2_niche + novae_niche + gpath2vec_niche from niche-join."""
    df = run_df.reset_index().copy()
    if 'spot_id' not in df.columns:
        # try index = spot_id case
        df['spot_id'] = run_df.index.values
    frames = []
    for sub in df['subarray'].unique():
        p = NICHES_DIR / f'{sub}.parquet'
        if not p.exists():
            continue
        nf = pd.read_parquet(p).reset_index()
        cols = ['spot_id', 'mc_megacluster']
        for opt in ['virchow2_niche', 'novae_niche', 'gpath2vec_niche']:
            if opt in nf.columns:
                cols.append(opt)
        nf['subarray'] = sub
        frames.append(nf[['subarray'] + cols])
    nj = pd.concat(frames, ignore_index=True)
    return df.merge(nj, on=['subarray', 'spot_id'], how='left')


def stack(col_series) -> np.ndarray:
    return np.stack([_as_arr(v) for v in col_series])


def run_one(run_id: str) -> dict:
    print(f'  {run_id}: load + join...')
    rd = load_run(run_id)
    merged = join_mc(rd)
    mask = merged['mc_megacluster'].notna()
    merged = merged[mask].copy()
    y = merged['mc_megacluster'].astype(int).values

    z_he = stack(merged['z_he'])
    z_st = stack(merged['z_st'])
    z_joint = np.concatenate([z_he, z_st], axis=1)

    print(f'  {run_id}: KMeans(k=14) on z_he ({len(y)} niches)...')
    out = {
        'run_id': run_id,
        'n_niches_with_mc': int(len(y)),
        'mc_z_he': cluster_quality(z_he, y),
        'mc_z_st': cluster_quality(z_st, y),
        'mc_z_joint': cluster_quality(z_joint, y),
    }
    return out


def run_raw_ref() -> dict:
    """raw_he + raw_st MC coherence on RAW_REF_RUN's test scope (same scope as archetype raw ref)."""
    print(f'  raw_he + raw_st reference ({RAW_REF_RUN} scope)...')
    rd = load_run(RAW_REF_RUN)
    merged = join_mc(rd)
    nan_he = merged['virchow2_niche'].isna() | merged['mc_megacluster'].isna()
    he_sub = merged[~nan_he].copy()
    y_he = he_sub['mc_megacluster'].astype(int).values
    x_he = stack(he_sub['virchow2_niche'])

    nan_st = (merged['novae_niche'].isna() | merged['gpath2vec_niche'].isna() |
              merged['mc_megacluster'].isna())
    st_sub = merged[~nan_st].copy()
    y_st = st_sub['mc_megacluster'].astype(int).values
    x_st = np.concatenate([stack(st_sub['novae_niche']),
                            stack(st_sub['gpath2vec_niche'])], axis=1)

    return {
        'raw_he': cluster_quality(x_he, y_he),
        'raw_st': cluster_quality(x_st, y_st),
    }


@click.command()
@click.option('--runs-dir', type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help='parent dir holding {run_id}/embeddings_test.parquet. defaults to runs/tnbc-92/.')
@click.option('--niches-dir', type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help='niche-join parquet dir (must have mc_megacluster column).')
@click.option('--h2-dir', type=click.Path(file_okay=False, path_type=Path), default=None,
              help='output dir. defaults to {runs-dir}/eval/H2.')
@click.option('--runs', 'runs_arg', default=None,
              help='comma-separated run ids (e.g. R1_v3,R2_v3,...).')
@click.option('--raw-reference-run', default=None,
              help='run whose test scope defines the raw_he/raw_st reference. defaults to RUNS[0].')
def main(runs_dir, niches_dir, h2_dir, runs_arg, raw_reference_run):
    global RUNS_ROOT, NICHES_DIR, H2_DIR, RUNS, RAW_REF_RUN
    if runs_dir is not None:
        RUNS_ROOT = runs_dir.resolve()
    if niches_dir is not None:
        NICHES_DIR = niches_dir.resolve()
    H2_DIR = h2_dir.resolve() if h2_dir is not None else RUNS_ROOT / 'eval' / 'H2'
    if runs_arg is not None:
        RUNS = [r.strip() for r in runs_arg.split(',') if r.strip()]
    RAW_REF_RUN = raw_reference_run or RUNS[0]

    H2_DIR.mkdir(parents=True, exist_ok=True)
    print(f'runs_dir   : {RUNS_ROOT}')
    print(f'niches_dir : {NICHES_DIR}')
    print(f'h2_dir     : {H2_DIR}')
    print(f'runs       : {RUNS}')
    print(f'raw ref    : {RAW_REF_RUN}')

    results = {'per_run': {}, 'raw_reference': None}
    print('\n=== H2 MC coherence ===')
    for r in RUNS:
        results['per_run'][r] = run_one(r)
    results['raw_reference'] = run_raw_ref()

    # flat parquet
    flat = []
    for r, d in results['per_run'].items():
        for view in ['mc_z_he', 'mc_z_st', 'mc_z_joint']:
            v = d[view]
            flat.append({
                'run_id': r, 'view': view.replace('mc_', ''),
                'ari': v['ari'], 'silhouette': v['silhouette'], 'n': v['n'],
            })
    for ref, v in results['raw_reference'].items():
        flat.append({
            'run_id': ref, 'view': '-',
            'ari': v['ari'], 'silhouette': v['silhouette'], 'n': v['n'],
        })
    fdf = pd.DataFrame(flat)
    fdf.to_parquet(H2_DIR / 'mc_coherence.parquet', index=False)
    print(f'\nwrote {H2_DIR / "mc_coherence.parquet"}')

    # merge into H2 summary.json under per_run[run].mc_*
    summ_path = H2_DIR / 'summary.json'
    summ = json.loads(summ_path.read_text())
    for r, d in results['per_run'].items():
        if r in summ['per_run']:
            summ['per_run'][r]['mc_z_he'] = d['mc_z_he']
            summ['per_run'][r]['mc_z_st'] = d['mc_z_st']
            summ['per_run'][r]['mc_z_joint'] = d['mc_z_joint']
            summ['per_run'][r]['n_niches_with_mc'] = d['n_niches_with_mc']
    summ['raw_reference']['raw_he_mc'] = results['raw_reference']['raw_he']
    summ['raw_reference']['raw_st_mc'] = results['raw_reference']['raw_st']
    summ_path.write_text(json.dumps(summ, indent=2))
    print(f'merged into {summ_path}')

    # side-by-side print with archetype
    print(f'\n=== side-by-side: archetype ARI vs MC ARI (z_he view) ===')
    print(f'{"run":<6}{"arch_ARI":>10}{"MC_ARI":>10}{"delta":>10}{"arch_sil":>10}{"MC_sil":>10}')
    for r in RUNS:
        a = summ['per_run'][r].get('z_he', {})
        m = summ['per_run'][r].get('mc_z_he', {})
        if not a or not m: continue
        delta = m['ari'] - a['ari']
        print(f'{r:<6}{a["ari"]:>10.3f}{m["ari"]:>10.3f}{delta:>+10.3f}'
              f'{a.get("silhouette",float("nan")):>10.3f}{m.get("silhouette",float("nan")):>10.3f}')
    rr = summ['raw_reference']
    print(f'{"raw_he":<6}{rr["raw_he"]["ari"]:>10.3f}{rr["raw_he_mc"]["ari"]:>10.3f}'
          f'{rr["raw_he_mc"]["ari"]-rr["raw_he"]["ari"]:>+10.3f}'
          f'{rr["raw_he"]["silhouette"]:>10.3f}{rr["raw_he_mc"]["silhouette"]:>10.3f}')
    print(f'{"raw_st":<6}{rr["raw_st"]["ari"]:>10.3f}{rr["raw_st_mc"]["ari"]:>10.3f}'
          f'{rr["raw_st_mc"]["ari"]-rr["raw_st"]["ari"]:>+10.3f}'
          f'{rr["raw_st"]["silhouette"]:>10.3f}{rr["raw_st_mc"]["silhouette"]:>10.3f}')


if __name__ == '__main__':
    main()
