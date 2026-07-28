#!/usr/bin/env python
"""eval_rank_matched_cka.py - rank-matched CKA control for the H1 fusion-axis verdict.

question this answers:
    R4 (cross-attn) wins H1 with CKA_after = 0.564 vs R1 (late) at 0.242.
    R4 also lives in ~3 effective dimensions (participation ratio, both streams).
    R1 lives in ~5 dims on z_he.
    is R4's CKA advantage "real" architectural work, or is it driven by
    compression (a tighter manifold in fewer dims naturally has higher CKA
    against any partner)?

how this answers it:
    project R1's z_he to its top-3 PCs (matching R4's effective rank).
    recompute linear CKA(R1_z_he_rank3, R1_z_st).
    compare to R4's CKA_after.

falsifier grid (committed BEFORE the number lands; do not relitigate post-hoc):
    R1@rank3 CKA >= R4 CKA (0.564):
        R4's H1 advantage is collapse-driven. cross-attention adds nothing
        R1 wouldn't do at matched rank. report should reframe to "R4 reaches
        higher CKA by compressing to ~3 dims; rank-matched comparison shows
        no architectural advantage over R1." flips R1-vs-R4 toward R1
        (preserves dim richness for free).
    R1@rank3 CKA materially below R4 (< 0.50):
        R4's cross-attention is doing real work beyond compression.
        existing trade-off framing holds.
    R1@rank3 CKA in [0.50, 0.56]:
        partial - R4 has some genuine advantage but most of the headline
        gap was compression. trade-off framing softens but holds.

sanity check:
    R4@rank3 CKA should be approximately R4's reported CKA_after (0.564),
    since R4 already lives in ~3 effective dims. if it differs by more
    than ~0.02, the rank-projection code has a bug and the R1@rank3
    number is unreliable.

usage:
    python scripts/eval_rank_matched_cka.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / 'runs' / 'tnbc-92'

R1_TARGET = 3   # match R4's effective rank
R4_TARGET = 3   # sanity check


def load_z(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(RUNS_ROOT / run_id / 'embeddings_test.parquet')
    z_he = np.stack([np.asarray(v, dtype=np.float64) for v in df['z_he']])
    z_st = np.stack([np.asarray(v, dtype=np.float64) for v in df['z_st']])
    return z_he, z_st


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """linear CKA between X (n, d_x) and Y (n, d_y), both centered.

    HSIC formulation collapses to:
        CKA = ||Y' X||_F^2 / (||X' X||_F * ||Y' Y||_F)
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    YtX = Yc.T @ Xc
    XtX = Xc.T @ Xc
    YtY = Yc.T @ Yc
    num = float(np.linalg.norm(YtX, ord='fro') ** 2)
    den = float(np.linalg.norm(XtX, ord='fro') * np.linalg.norm(YtY, ord='fro'))
    return num / den if den > 0 else float('nan')


def project_to_top_k(X: np.ndarray, k: int) -> np.ndarray:
    """project X onto its top-k principal components (centered SVD)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # use SVD on Xc; right singular vectors (V) are the PCs
    # Xc = U S V^T ; project = Xc @ V[:, :k]
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    V_k = vt[:k].T   # (d, k)
    return Xc @ V_k


def participation_ratio(X: np.ndarray) -> float:
    Xc = X - X.mean(0, keepdims=True)
    C = Xc.T @ Xc / max(len(Xc), 1)
    evals = np.linalg.eigvalsh(C)
    evals = evals[evals > 1e-12]
    return float((evals.sum() ** 2) / ((evals ** 2).sum() + 1e-20))


def run_one(run_id: str, k: int) -> dict:
    z_he, z_st = load_z(run_id)
    cka_full = linear_cka(z_he, z_st)
    z_he_k = project_to_top_k(z_he, k)
    cka_k = linear_cka(z_he_k, z_st)
    pr = participation_ratio(z_he)
    return {
        'run_id': run_id,
        'k': k,
        'n_test_niches': len(z_he),
        'pr_z_he': pr,
        'cka_full_dim': cka_full,
        f'cka_z_he_top{k}': cka_k,
    }


def verdict_string(r1_at_3: float, r4_full: float) -> str:
    if r1_at_3 >= r4_full:
        return ('VERDICT: collapse-driven. R4 has no architectural advantage over R1 '
                'at matched rank. report should reframe and flip recommendation toward R1.')
    if r1_at_3 < 0.50:
        return ('VERDICT: R4 architecture does real work. trade-off framing holds.')
    return ('VERDICT: partial. R4 has genuine advantage but most of the headline gap '
            'was compression. trade-off framing softens but holds.')


def main():
    print('=== rank-matched CKA control ===')
    print(f'(falsifier grid in docstring; do not relitigate post-hoc)\n')

    # load R4's reported CKA_after for the comparison baseline
    r4_h1 = json.loads((RUNS_ROOT / 'R4' / 'metrics_h1_raw.json').read_text())
    r4_cka_reported = float(r4_h1['cka_after'])
    r1_h1 = json.loads((RUNS_ROOT / 'R1' / 'metrics_h1_raw.json').read_text())
    r1_cka_reported = float(r1_h1['cka_after'])
    print(f'R4 reported CKA_after = {r4_cka_reported:.4f}  (target to match/exceed)')
    print(f'R1 reported CKA_after = {r1_cka_reported:.4f}  (full-dim baseline)\n')

    rows = []
    for run_id, k in [('R1', R1_TARGET), ('R4', R4_TARGET)]:
        print(f'  computing {run_id} CKA full-dim and top-{k} ...')
        rows.append(run_one(run_id, k))
    df = pd.DataFrame(rows)
    print()
    print(df.round(4).to_string(index=False))
    print()

    # sanity check first
    r4_full = float(df.loc[df['run_id'] == 'R4', 'cka_full_dim'].iloc[0])
    r4_at_3 = float(df.loc[df['run_id'] == 'R4', 'cka_z_he_top3'].iloc[0])
    sanity_gap = abs(r4_full - r4_at_3)
    print(f'SANITY CHECK: R4 full-dim CKA = {r4_full:.4f}, R4@rank3 CKA = {r4_at_3:.4f}, '
          f'gap = {sanity_gap:.4f}')
    if sanity_gap > 0.02:
        print(f'  WARNING: sanity gap > 0.02; rank-projection may be buggy. '
              f'R1@rank3 result below is UNRELIABLE.')
    else:
        print(f'  OK: gap within 0.02 - R4 already lives in ~3 dims, projection consistent.\n')

    # main test
    r1_at_3 = float(df.loc[df['run_id'] == 'R1', 'cka_z_he_top3'].iloc[0])
    print(f'MAIN TEST: R1@rank3 CKA = {r1_at_3:.4f}, R4 full-dim CKA = {r4_full:.4f}')
    print(f'  delta = {r1_at_3 - r4_full:+.4f}')
    print(f'  {verdict_string(r1_at_3, r4_full)}\n')

    out_path = RUNS_ROOT / 'eval' / 'compare' / 'rank_matched_cka.parquet'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f'saved -> {out_path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
