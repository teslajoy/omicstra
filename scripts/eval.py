#!/usr/bin/env python
"""eval.py - hypothesis-specific evaluation on run outputs.

reads runs/tnbc-92/{run_id}/embeddings_test.parquet + metrics_h1_raw.json
joins with data/embeddings/niches/*.parquet for archetype, compartment, gpath2vec

usage:
    python scripts/eval.py --hypothesis H1 --runs R1 R2 R3 R4 B1 B2 B3
    python scripts/eval.py --hypothesis H2 --runs R1 R2 R3 R4 B1 B2 B3
    python scripts/eval.py --hypothesis H3 --runs R1 R4 B2

outputs:
    runs/tnbc-92/eval/{hypothesis}/summary.json     rollup across runs
    runs/tnbc-92/{run_id}/eval/{hypothesis}.json    per-run detail
    runs/tnbc-92/eval/{hypothesis}/compartment_cosine.parquet  (H2)
    runs/tnbc-92/eval/{hypothesis}/attention_by_compartment.parquet  (H3, R4 only)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
# mutable module-level defaults; main() overrides from --runs-dir / --niches-dir
# so the v3 retrain can target runs/tnbc-92_v3/ + data/embeddings/niches_v3/
# without touching every function's signature. functions read at call-time.
RUNS_ROOT = ROOT / 'runs' / 'tnbc-92'
NICHES_DIR = ROOT / 'data' / 'embeddings' / 'niches'

N_ARCHETYPES = 9
SEED = 42


def _rel_to_root(p: Path) -> Path | str:
    """defensive relative_to for printing/serialization. handles relative paths
    that don't resolve under ROOT (e.g. when --runs-dir is given as a relative
    string outside repo)."""
    pr = p.resolve()
    return pr.relative_to(ROOT) if pr.is_relative_to(ROOT) else pr


# ------------------------------------------------------------------------- #
# shared loaders
# ------------------------------------------------------------------------- #

def _as_arr(cell) -> np.ndarray:
    return np.asarray(cell, dtype=np.float32)


def load_run_embeddings(run_id: str) -> pd.DataFrame:
    """read embeddings_test.parquet and unpack z_he, z_st (+attention if present)."""
    path = RUNS_ROOT / run_id / 'embeddings_test.parquet'
    df = pd.read_parquet(path)
    return df


def z_matrix(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack([_as_arr(v) for v in df[col]])


def load_niches_for_spots(run_df: pd.DataFrame) -> pd.DataFrame:
    """join run's test niches with the full niche-join parquets to fetch
    raw features + gpath2vec + labels."""
    wanted = run_df.reset_index()[['subarray', 'spot_id']].drop_duplicates()
    frames = []
    for sub in wanted['subarray'].unique():
        p = NICHES_DIR / f'{sub}.parquet'
        if not p.exists():
            continue
        nf = pd.read_parquet(p)
        nf = nf.reset_index()
        nf['subarray'] = sub
        frames.append(nf)
    nj = pd.concat(frames, ignore_index=True)
    merged = wanted.merge(nj, on=['subarray', 'spot_id'], how='left')
    return merged


def split_probe(n_samples: int, seed: int = SEED):
    idx = np.arange(n_samples)
    return train_test_split(idx, test_size=0.2, random_state=seed)


# ------------------------------------------------------------------------- #
# H1 - already computed in align.py / align_classical.py
# ------------------------------------------------------------------------- #

def eval_h1(runs: list[str]) -> dict:
    """rollup the already-computed metrics_h1_raw.json across runs."""
    table = {}
    for r in runs:
        path = RUNS_ROOT / r / 'metrics_h1_raw.json'
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        table[r] = {
            'R@1': m['r1'], 'R@5': m['r5'], 'R@10': m['r10'], 'MRR': m['mrr'],
            'median_rank': m['median_rank'], 'alignment_gap': m['alignment_gap'],
            'AUC': m['auc'], 'CKA_before': m['cka_before'], 'CKA_after': m['cka_after'],
            'test_niches': m.get('test_niches'),
        }
    return {'per_run': table}


# ------------------------------------------------------------------------- #
# H2 - archetype coherence + per-compartment cosine
# ------------------------------------------------------------------------- #

def cluster_quality(X: np.ndarray, y: np.ndarray, k: int = N_ARCHETYPES,
                    silhouette_sample: int = 5000) -> dict:
    """K-means(k)->ARI vs y, plus silhouette on a random subsample."""
    if len(np.unique(y)) < 2 or len(y) < k:
        return {'ari': float('nan'), 'silhouette': float('nan')}
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    preds = km.fit_predict(X)
    ari = float(adjusted_rand_score(y, preds))
    n_sil = min(silhouette_sample, len(y))
    idx = np.random.default_rng(SEED).choice(len(y), n_sil, replace=False)
    sil = float(silhouette_score(X[idx], y[idx], sample_size=n_sil, random_state=SEED))
    return {'ari': ari, 'silhouette': sil}


def eval_h2_archetype_per_run(run_id: str) -> dict:
    df = load_run_embeddings(run_id)
    y = df['archetype'].values.astype(int)
    z_he = z_matrix(df, 'z_he')
    z_st = z_matrix(df, 'z_st')
    z_joint = np.concatenate([z_he, z_st], axis=1)
    return {
        'run_id': run_id,
        'n_test_niches': len(df),
        'z_he': cluster_quality(z_he, y),
        'z_st': cluster_quality(z_st, y),
        'z_joint': cluster_quality(z_joint, y),
    }


def eval_h2_compartment_cosine(run_id: str) -> pd.DataFrame:
    """mean matched-pair cosine (z_he · z_st) stratified by 18-class compartment."""
    df = load_run_embeddings(run_id)
    has_comp = df['compartment'].notna() & (df['compartment'] != '')
    df = df[has_comp]
    if len(df) == 0:
        return pd.DataFrame(columns=['run_id', 'compartment', 'n', 'mean_cosine'])
    z_he = z_matrix(df, 'z_he')
    z_st = z_matrix(df, 'z_st')
    cos = (z_he * z_st).sum(axis=1)
    out = pd.DataFrame({'compartment': df['compartment'].values, 'cos': cos})
    agg = out.groupby('compartment').agg(
        n=('cos', 'size'), mean_cosine=('cos', 'mean'), std=('cos', 'std'),
    ).reset_index()
    agg.insert(0, 'run_id', run_id)
    return agg


def eval_h2_raw_reference(reference_run: str = 'R1') -> dict:
    """raw-modality ARI + silhouette (same across all runs; computed once).

    uses reference run's test niches to define scope, joins to niche-join
    parquets to get raw virchow2_niche + concat-ST features. callers should
    pass the v3-aware reference (e.g. 'R1_v3') when targeting niches_v3.
    """
    df = load_run_embeddings(reference_run)
    y = df['archetype'].values.astype(int)
    merged = load_niches_for_spots(df)
    # align merged order back to df order
    merged = merged.set_index(['subarray', 'spot_id']).reindex(
        df.reset_index()[['subarray', 'spot_id']].apply(tuple, axis=1).tolist()
    ).reset_index()
    x_he = np.stack([_as_arr(v) for v in merged['virchow2_niche']])
    x_st = np.concatenate([
        np.stack([_as_arr(v) for v in merged['novae_niche']]),
        np.stack([_as_arr(v) for v in merged['gpath2vec_niche']]),
    ], axis=1)
    nan_rows = np.isnan(x_he).any(1) | np.isnan(x_st).any(1)
    x_he, x_st, y = x_he[~nan_rows], x_st[~nan_rows], y[~nan_rows]
    return {
        'raw_he': cluster_quality(x_he, y),
        'raw_st': cluster_quality(x_st, y),
    }


def eval_h2(runs: list[str], reference_run: str | None = None) -> dict:
    out = {'per_run': {}, 'raw_reference': None, 'compartment_cosine_runs': []}
    print('  H2: raw reference (once)...')
    out['raw_reference'] = eval_h2_raw_reference(reference_run or runs[0])
    for r in runs:
        print(f'  H2: {r} archetype clustering...')
        out['per_run'][r] = eval_h2_archetype_per_run(r)
        print(f'  H2: {r} compartment cosine...')
        out['compartment_cosine_runs'].append(eval_h2_compartment_cosine(r))
    out['compartment_cosine'] = pd.concat(
        out['compartment_cosine_runs'], ignore_index=True
    )
    # drop DataFrames from json-serializable summary
    out.pop('compartment_cosine_runs')
    return out


# ------------------------------------------------------------------------- #
# H3 - pathway coherence + patient probe + attention
# ------------------------------------------------------------------------- #

def cca_top_k(X: np.ndarray, Y: np.ndarray, k: int = 10, reg: float = 1e-4) -> dict:
    """closed-form CCA; returns top-k canonical correlations + sum."""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    N = Xc.shape[0]
    Cxx = (Xc.T @ Xc) / N + reg * np.eye(Xc.shape[1])
    Cyy = (Yc.T @ Yc) / N + reg * np.eye(Yc.shape[1])
    Cxy = (Xc.T @ Yc) / N

    def whiten(C):
        vals, vecs = np.linalg.eigh(C.astype(np.float64))
        vals = np.maximum(vals, reg)
        return vecs @ np.diag(vals ** -0.5) @ vecs.T

    M = whiten(Cxx) @ Cxy.astype(np.float64) @ whiten(Cyy)
    s = np.linalg.svd(M, compute_uv=False)
    top = s[:k].tolist()
    return {'top_k_canonical_correlations': top, 'top_k_sum': float(sum(top)), 'k': k}


def geometric_rho(X: np.ndarray, Y: np.ndarray, n_sample: int = 5000) -> float:
    """spearman rho between pairwise cosine sim matrices (upper triangle)."""
    idx = np.random.default_rng(SEED).choice(len(X), min(n_sample, len(X)), replace=False)
    Xn = X[idx] / (np.linalg.norm(X[idx], axis=1, keepdims=True) + 1e-12)
    Yn = Y[idx] / (np.linalg.norm(Y[idx], axis=1, keepdims=True) + 1e-12)
    Sx = Xn @ Xn.T
    Sy = Yn @ Yn.T
    iu = np.triu_indices(len(idx), k=1)
    rho, _ = spearmanr(Sx[iu], Sy[iu])
    return float(rho)


def patient_probe(X: np.ndarray, y_patient: np.ndarray, max_iter: int = 300) -> dict:
    """logistic regression probe for patient_id; accuracy lower = better disentanglement."""
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X.astype(np.float64))
    tr, te = split_probe(len(Xs))
    # sklearn 1.7+ removed multi_class param; lbfgs defaults to multinomial
    clf = LogisticRegression(max_iter=max_iter, solver='lbfgs', n_jobs=-1, C=1.0)
    clf.fit(Xs[tr], y_patient[tr])
    acc = float(clf.score(Xs[te], y_patient[te]))
    n_classes = int(len(np.unique(y_patient)))
    chance = 1.0 / n_classes
    return {'accuracy': acc, 'chance': chance, 'lift_over_chance': acc - chance,
            'n_classes': n_classes, 'n_test': int(len(te))}


def eval_h3_per_run(run_id: str, include_raw: bool = False,
                    raw_cache: dict | None = None) -> dict:
    df = load_run_embeddings(run_id)
    merged = load_niches_for_spots(df)
    # align merged to df order
    merged = merged.set_index(['subarray', 'spot_id']).reindex(
        df.reset_index()[['subarray', 'spot_id']].apply(tuple, axis=1).tolist()
    ).reset_index()
    gp = np.stack([_as_arr(v) for v in merged['gpath2vec_niche']])
    patient = df['patient_id'].values.astype(int)
    z_he = z_matrix(df, 'z_he')
    z_st = z_matrix(df, 'z_st')
    z_mean = 0.5 * (z_he + z_st)

    # drop any NaN rows in gpath2vec
    ok = ~np.isnan(gp).any(1)
    gp, patient = gp[ok], patient[ok]
    z_he, z_st, z_mean = z_he[ok], z_st[ok], z_mean[ok]

    out = {
        'run_id': run_id,
        'n_test_niches': int(len(patient)),
        'n_patients': int(len(np.unique(patient))),
    }

    # pathway coherence - CCA(z_mean, gpath2vec) + geometric rho
    out['cca_shared_vs_gpath2vec'] = cca_top_k(z_mean, gp, k=10)
    out['geom_rho_shared_vs_gpath2vec'] = geometric_rho(z_mean, gp)

    # patient probe on shared representations
    out['patient_probe_z_he'] = patient_probe(z_he, patient)
    out['patient_probe_z_st'] = patient_probe(z_st, patient)
    out['patient_probe_z_mean'] = patient_probe(z_mean, patient)

    if include_raw:
        if raw_cache is None:
            raw_cache = {}
        if 'raw_he' not in raw_cache:
            x_he = np.stack([_as_arr(v) for v in merged['virchow2_niche']])[ok]
            x_st = np.concatenate([
                np.stack([_as_arr(v) for v in merged['novae_niche']])[ok],
                gp,
            ], axis=1)
            raw_cache['raw_he'] = patient_probe(x_he, patient)
            raw_cache['raw_st'] = patient_probe(x_st, patient)
        out['patient_probe_raw_he'] = raw_cache['raw_he']
        out['patient_probe_raw_st'] = raw_cache['raw_st']

    return out


def eval_h3_attention_by_compartment(run_id: str) -> pd.DataFrame | None:
    """R4 cross_attn only - (N, 7) attention weights by compartment."""
    df = load_run_embeddings(run_id)
    if 'attention_weights' not in df.columns:
        return None
    attn = np.stack([_as_arr(v) for v in df['attention_weights']])   # (N, 7)
    comp = df['compartment'].values
    mask = pd.Series(comp).notna() & (pd.Series(comp) != '')
    attn, comp = attn[mask.values], comp[mask.values]
    if len(attn) == 0:
        return None
    out = pd.DataFrame(attn, columns=[f'pos_{i}' for i in range(attn.shape[1])])
    out['compartment'] = comp
    agg = out.groupby('compartment').agg(['mean', 'std', 'count']).reset_index()
    agg.insert(0, 'run_id', run_id)
    return agg


def eval_h3(runs: list[str]) -> dict:
    out = {'per_run': {}, 'attention_by_compartment_runs': []}
    raw_cache = {}
    for i, r in enumerate(runs):
        print(f'  H3: {r} ({i+1}/{len(runs)}) CCA + probes...')
        out['per_run'][r] = eval_h3_per_run(r, include_raw=(i == 0), raw_cache=raw_cache)
        attn = eval_h3_attention_by_compartment(r)
        if attn is not None:
            print(f'  H3: {r} attention-by-compartment saved')
            out['attention_by_compartment_runs'].append(attn)
    if out['attention_by_compartment_runs']:
        out['attention_by_compartment'] = pd.concat(
            out['attention_by_compartment_runs'], ignore_index=True,
        )
    out.pop('attention_by_compartment_runs')
    return out


# ------------------------------------------------------------------------- #
# main
# ------------------------------------------------------------------------- #

def main():
    global RUNS_ROOT, NICHES_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--hypothesis', required=True, choices=['H1', 'H2', 'H3'])
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--runs-dir', type=Path, default=None,
                    help='parent dir holding {run_id}/embeddings_test.parquet. '
                         'defaults to runs/tnbc-92/. use runs/tnbc-92_v3/ for v3.')
    ap.add_argument('--niches-dir', type=Path, default=None,
                    help='niche-join parquet dir. defaults to data/embeddings/niches/. '
                         'use data/embeddings/niches_v3/ for v3.')
    ap.add_argument('--raw-reference-run', default=None,
                    help='run whose test niches define the H2 raw_reference scope. '
                         'defaults to the first --runs entry.')
    args = ap.parse_args()

    if args.runs_dir is not None:
        RUNS_ROOT = args.runs_dir.resolve()
    if args.niches_dir is not None:
        NICHES_DIR = args.niches_dir.resolve()
    assert RUNS_ROOT.is_dir(), f'runs_dir not found: {RUNS_ROOT}'
    assert NICHES_DIR.is_dir(), f'niches_dir not found: {NICHES_DIR}'

    out_dir = RUNS_ROOT / 'eval' / args.hypothesis
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f'=== eval {args.hypothesis} on runs: {args.runs} ===')
    print(f'    runs_dir:   {_rel_to_root(RUNS_ROOT)}')
    print(f'    niches_dir: {_rel_to_root(NICHES_DIR)}')
    if args.hypothesis == 'H1':
        result = eval_h1(args.runs)
    elif args.hypothesis == 'H2':
        result = eval_h2(args.runs, reference_run=args.raw_reference_run)
        if 'compartment_cosine' in result and len(result['compartment_cosine']):
            result['compartment_cosine'].to_parquet(out_dir / 'compartment_cosine.parquet')
            result['compartment_cosine'] = f'written -> {_rel_to_root(out_dir)}/compartment_cosine.parquet'
    elif args.hypothesis == 'H3':
        result = eval_h3(args.runs)
        if 'attention_by_compartment' in result:
            att = result['attention_by_compartment']
            att.columns = [f'{a}_{b}' if b else a for a, b in att.columns]
            att.to_parquet(out_dir / 'attention_by_compartment.parquet')
            result['attention_by_compartment'] = f'written -> {_rel_to_root(out_dir)}/attention_by_compartment.parquet'

    # per-run detail
    for r, detail in result.get('per_run', {}).items():
        run_eval_dir = RUNS_ROOT / r / 'eval'
        run_eval_dir.mkdir(parents=True, exist_ok=True)
        with open(run_eval_dir / f'{args.hypothesis}.json', 'w') as f:
            json.dump(detail, f, indent=2, default=str)

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f'\ndone in {elapsed:.1f}s')
    print(f'summary: {_rel_to_root(out_dir / "summary.json")}')


if __name__ == '__main__':
    main()