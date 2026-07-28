#!/usr/bin/env python
"""align_classical.py - H1 classical linear-alignment baselines for tnbc-92 grid.

no training loop - fits on train patients, applies to test patients, saves the
same artifact schema as align.py so eval.py can treat all runs uniformly.

baselines implemented:
    B1  CCA: canonical correlation analysis, linear projection to shared_dim
    B2  Procrustes: PCA each modality to shared_dim, then orthogonal Procrustes
                    alignment (H&E rotated to match ST basis)
    B3  Unaligned: PCA each to shared_dim, L2-norm, no alignment (geometric null)

all baselines use the SAME train/val/test patient split as the corresponding
contrastive run (reads split.json from a reference run). this keeps H1 comparison
fair across trained and classical methods.

usage:
    python scripts/align_classical.py --baseline cca --reference-split runs/tnbc-92/R1/split.json --run-id B1
    python scripts/align_classical.py --baseline procrustes --reference-split runs/tnbc-92/R1/split.json --run-id B2
    python scripts/align_classical.py --baseline unaligned --reference-split runs/tnbc-92/R1/split.json --run-id B3

outputs per run at runs/tnbc-92/{run_id}/:
    run_config.json           # baseline name + source split
    split.json                # copy of reference split
    embeddings_test.parquet   # per-niche z_he, z_st (no attention weights)
    metrics_h1_raw.json       # R@K, MRR, gap, AUC, CKA (same schema as align.py)

scope: matched 260 subarrays from data/embeddings/niches/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# reuse loaders + metrics from align.py to stay schema-consistent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import (  # noqa: E402
    DEFAULT_NICHES_DIR, RUNS_ROOT, ROOT,
    discover_subarrays, load_subarray, cka_linear, retrieval_metrics,
)
import torch  # noqa: E402

ST_FEATURES = ('novae_niche', 'gpath2vec_niche')


def build_matrices(subs: list[dict]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """stack H&E niche and ST features across all subarrays, with per-niche metadata."""
    X_he = np.concatenate([s['x_he_niche'] for s in subs])
    X_st = np.concatenate([s['x_st'] for s in subs])
    meta_rows = []
    for s in subs:
        for sid, comp in zip(s['spot_id'], s['compartment']):
            meta_rows.append({
                'spot_id': sid,
                'subarray': s['subarray'],
                'patient_id': s['patient_id'],
                'archetype': s['archetype'],
                'compartment': comp,
            })
    meta = pd.DataFrame(meta_rows)
    return X_he, X_st, meta


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return (x / n).astype(np.float32)


def run_cca(
    X_he_train, X_st_train, X_he_test, X_st_test, shared_dim: int,
    reg: float = 1e-4,
):
    """closed-form CCA via SVD of whitened cross-covariance.

    equivalent to sklearn.CCA but O(seconds) instead of O(hours) at scale.
    regularizer `reg` stabilizes covariance inversion on ~180k high-dim data.
    """
    k = min(shared_dim, X_he_train.shape[1], X_st_train.shape[1], X_he_train.shape[0] - 1)
    Xc = X_he_train - X_he_train.mean(0, keepdims=True)
    Yc = X_st_train - X_st_train.mean(0, keepdims=True)
    N = Xc.shape[0]
    Cxx = (Xc.T @ Xc) / N + reg * np.eye(Xc.shape[1], dtype=np.float64)
    Cyy = (Yc.T @ Yc) / N + reg * np.eye(Yc.shape[1], dtype=np.float64)
    Cxy = (Xc.T @ Yc) / N

    # whiten: Cxx^{-1/2} via eigendecomp
    def whiten(C):
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, reg)
        return vecs @ np.diag(vals ** -0.5) @ vecs.T

    Wx = whiten(Cxx.astype(np.float64))
    Wy = whiten(Cyy.astype(np.float64))
    M = Wx @ Cxy.astype(np.float64) @ Wy
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    A = Wx @ U[:, :k]                         # (D_he, k) H&E projection
    B = Wy @ Vt[:k, :].T                      # (D_st, k) ST projection

    # apply to test
    mu_he = X_he_train.mean(0, keepdims=True)
    mu_st = X_st_train.mean(0, keepdims=True)
    z_he = (X_he_test - mu_he) @ A
    z_st = (X_st_test - mu_st) @ B
    return l2_normalize(z_he), l2_normalize(z_st), k


def run_procrustes(
    X_he_train, X_st_train, X_he_test, X_st_test, shared_dim: int,
):
    """PCA each modality to shared_dim, then solve orthogonal Procrustes.

    fits X_he_pca @ R ~ X_st_pca. reports test embeddings in the ST basis.
    """
    pca_he = PCA(n_components=shared_dim, random_state=42)
    pca_st = PCA(n_components=shared_dim, random_state=42)
    H_tr = pca_he.fit_transform(X_he_train)
    S_tr = pca_st.fit_transform(X_st_train)
    # orthogonal Procrustes: R = argmin ||H_tr R - S_tr||_F, R^T R = I
    U, _, Vt = np.linalg.svd(H_tr.T @ S_tr, full_matrices=False)
    R = U @ Vt
    H_te = pca_he.transform(X_he_test) @ R
    S_te = pca_st.transform(X_st_test)
    return l2_normalize(H_te), l2_normalize(S_te), shared_dim


def run_unaligned(
    X_he_train, X_st_train, X_he_test, X_st_test, shared_dim: int,
):
    """PCA each modality independently to shared_dim, L2-norm, no alignment.

    geometric null: what does cross-modal cosine look like when the only
    signal is per-modality covariance structure?
    """
    pca_he = PCA(n_components=shared_dim, random_state=42).fit(X_he_train)
    pca_st = PCA(n_components=shared_dim, random_state=42).fit(X_st_train)
    return (
        l2_normalize(pca_he.transform(X_he_test)),
        l2_normalize(pca_st.transform(X_st_test)),
        shared_dim,
    )


BASELINES = {'cca': run_cca, 'procrustes': run_procrustes, 'unaligned': run_unaligned}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, choices=list(BASELINES.keys()))
    ap.add_argument('--reference-split', required=True,
                    help='path to split.json from a contrastive run (e.g. runs/tnbc-92/R1/split.json)')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--shared-dim', type=int, default=512)
    ap.add_argument('--subset', type=int, default=None)
    ap.add_argument('--niches-dir', type=Path, default=None,
                    help='path to niche-join parquet dir. defaults to v1 '
                         '(data/embeddings/niches/). use niches_v3 for retrain.')
    ap.add_argument('--runs-root', type=Path, default=None,
                    help='parent dir for run output (default runs/tnbc-92/).')
    args = ap.parse_args()

    np.random.seed(42)  # PCA's randomized SVD is non-deterministic otherwise

    niches_dir = args.niches_dir if args.niches_dir is not None else DEFAULT_NICHES_DIR
    runs_root = args.runs_root if args.runs_root is not None else RUNS_ROOT
    assert niches_dir.is_dir(), f'niches_dir not found: {niches_dir}'

    run_dir = runs_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # copy reference split so eval treats all runs with identical held-out set
    split = json.loads(Path(args.reference_split).read_text())
    (run_dir / 'split.json').write_text(json.dumps(split, indent=2))
    train_p, val_p, test_p = set(split['train_patients']), set(split['val_patients']), set(split['test_patients'])

    # capture provenance from the niche-join manifest
    gpath_sha = None
    if (niches_dir / 'manifest.json').is_file():
        _man = json.loads((niches_dir / 'manifest.json').read_text())
        gpath_sha = _man.get('sources', {}).get('gpath2vec_sha256')

    cfg = {
        'run_id': args.run_id,
        'baseline': args.baseline,
        'shared_dim': args.shared_dim,
        'st_features': list(ST_FEATURES),
        'reference_split': args.reference_split,
        'niches_dir': str(niches_dir.relative_to(ROOT)
                          if niches_dir.is_relative_to(ROOT) else niches_dir),
        'gpath2vec_parquet_sha256': gpath_sha,
    }
    (run_dir / 'run_config.json').write_text(json.dumps(cfg, indent=2))

    paths = discover_subarrays(niches_dir)
    if args.subset is not None:
        paths = paths[:args.subset]
    print(f'loading {len(paths)} subarrays...')
    subs_raw = [load_subarray(p, ST_FEATURES) for p in paths]
    subs = [s for s in subs_raw if s['patient_id'] is not None and s['archetype'] is not None]
    dropped = len(subs_raw) - len(subs)
    if dropped:
        print(f'  dropped {dropped} subarrays with missing labels')

    train_subs = [s for s in subs if s['patient_id'] in train_p]
    test_subs = [s for s in subs if s['patient_id'] in test_p]
    # val is unused in classical baselines (no optim)

    X_he_tr, X_st_tr, _ = build_matrices(train_subs)
    X_he_te, X_st_te, meta_te = build_matrices(test_subs)
    print(f'train niches: {X_he_tr.shape[0]}, test niches: {X_he_te.shape[0]}')
    print(f'H&E dim: {X_he_tr.shape[1]}, ST dim: {X_st_tr.shape[1]}')

    runner = BASELINES[args.baseline]
    print(f'running baseline: {args.baseline} (shared_dim={args.shared_dim})')
    z_he, z_st, k_used = runner(X_he_tr, X_st_tr, X_he_te, X_st_te, args.shared_dim)
    print(f'shared_dim used: {k_used}')

    # save embeddings_test.parquet (same schema as align.py)
    emb_rows = []
    for i, row in meta_te.reset_index(drop=True).iterrows():
        emb_rows.append({
            'spot_id': row['spot_id'],
            'subarray': row['subarray'],
            'patient_id': row['patient_id'],
            'archetype': row['archetype'],
            'compartment': row['compartment'],
            'z_he': z_he[i].tolist(),
            'z_st': z_st[i].tolist(),
        })
    emb_df = pd.DataFrame(emb_rows).set_index('spot_id')
    emb_df.to_parquet(run_dir / 'embeddings_test.parquet')

    # H1 metrics
    metrics = retrieval_metrics(torch.from_numpy(z_he), torch.from_numpy(z_st))
    metrics['cka_before'] = cka_linear(X_he_te, X_st_te)
    metrics['cka_after'] = cka_linear(z_he, z_st)
    metrics['test_subarrays'] = len(test_subs)
    metrics['test_niches'] = int(len(emb_df))
    metrics['shared_dim_used'] = int(k_used)
    metrics['baseline'] = args.baseline
    (run_dir / 'metrics_h1_raw.json').write_text(json.dumps(metrics, indent=2))

    print(f'\nH1 metrics ({args.baseline}): '
          f'R@1={metrics["r1"]:.3f} R@5={metrics["r5"]:.3f} R@10={metrics["r10"]:.3f} '
          f'MRR={metrics["mrr"]:.3f} gap={metrics["alignment_gap"]:.3f} '
          f'CKA {metrics["cka_before"]:.3f}->{metrics["cka_after"]:.3f}')
    _rd = run_dir.resolve()
    print(f'artifacts in {_rd.relative_to(ROOT) if _rd.is_relative_to(ROOT) else _rd}')


if __name__ == '__main__':
    main()