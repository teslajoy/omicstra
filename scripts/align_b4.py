#!/usr/bin/env python
"""align_b4.py - B4 strict-control: random-init late-fusion MLPs, no training.

The proposal H1 baselines: CCA (B1), late fusion concatenation, raw unaligned
concatenation. The strict version of baseline 2 is: independent late-fusion
MLPs with the SAME architecture as R1, but RANDOM Xavier init and NO training.
This isolates whether contrastive loss adds anything over a random projection
of the same architectural shape.

B4 architecture: byte-identical to R1's LateFusion (MLPBlock = LayerNorm,
Linear, ReLU, BatchNorm, Dropout, Linear, L2-norm), imported from align.py
so any architecture change to R1 propagates to B4 automatically.

writes the same artifact schema as align.py / align_classical.py so eval.py
treats all runs uniformly.

usage:
    python scripts/align_b4.py \\
      --reference-split runs/tnbc-92_v3/R1_v3/split.json \\
      --run-id B4_v3 \\
      --niches-dir data/embeddings/niches_v3 \\
      --runs-root runs/tnbc-92_v3
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

# reuse loaders + model code from align.py for cross-run consistency
sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import (  # noqa: E402
    DEFAULT_NICHES_DIR, RUNS_ROOT, ROOT,
    discover_subarrays, load_subarray, cka_linear, retrieval_metrics,
    LateFusion, _git_commit_sha,
)

ST_FEATURES = ('novae_niche', 'gpath2vec_niche')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference-split', required=True,
                    help='split.json from a contrastive run; B4 uses the same '
                         'held-out patients so H1 numbers are comparable.')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--niches-dir', type=Path, default=None,
                    help='niche-join parquet dir. defaults to v1 (data/embeddings/niches/).')
    ap.add_argument('--runs-root', type=Path, default=None)
    ap.add_argument('--shared-dim', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.3,
                    help='matches R1 default in projects/tnbc-92/alignment/config/R1.json')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--subset', type=int, default=None)
    args = ap.parse_args()

    # seed BEFORE building the model so Xavier init is deterministic
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    niches_dir = args.niches_dir if args.niches_dir is not None else DEFAULT_NICHES_DIR
    runs_root = args.runs_root if args.runs_root is not None else RUNS_ROOT
    assert niches_dir.is_dir(), f'niches_dir not found: {niches_dir}'

    run_dir = runs_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # copy reference split so eval treats B4 with identical held-out set
    split = json.loads(Path(args.reference_split).read_text())
    (run_dir / 'split.json').write_text(json.dumps(split, indent=2))
    test_p = set(split['test_patients'])

    # provenance from the niche-join manifest
    gpath_sha = None
    if (niches_dir / 'manifest.json').is_file():
        _man = json.loads((niches_dir / 'manifest.json').read_text())
        gpath_sha = _man.get('sources', {}).get('gpath2vec_sha256')

    cfg = {
        'run_id': args.run_id,
        'baseline': 'b4_random_init_late_fusion',
        'loss': 'none',
        'training_steps': 0,
        'fusion': 'late',
        'shared_dim': args.shared_dim,
        'dropout': args.dropout,
        'st_features': list(ST_FEATURES),
        'reference_split': args.reference_split,
        'niches_dir': str(niches_dir.relative_to(ROOT)
                          if niches_dir.is_relative_to(ROOT) else niches_dir),
        'runs_root': str(runs_root.relative_to(ROOT)
                         if runs_root.is_relative_to(ROOT) else runs_root),
        'gpath2vec_parquet_sha256': gpath_sha,
        'git_commit_sha': _git_commit_sha(),
        'seed': args.seed,
        'note': 'random-init Xavier MLPs, no contrastive training. controls for '
                'architecture-vs-loss attribution per proposal H1 baseline 2 (strict).',
    }
    (run_dir / 'run_config.json').write_text(json.dumps(cfg, indent=2))

    paths = discover_subarrays(niches_dir)
    if args.subset is not None:
        paths = paths[:args.subset]
    print(f'loading {len(paths)} subarrays...')
    subs_raw = [load_subarray(p, ST_FEATURES) for p in paths]
    # filter to test patients (B4 only needs test pool - no train/val)
    subs = [s for s in subs_raw
            if s['patient_id'] is not None and s['archetype'] is not None
            and s['patient_id'] in test_p]
    print(f'test subarrays: {len(subs)}')

    he_dim = subs[0]['x_he_niche'].shape[1]
    he_tile_dim = subs[0]['x_he_tokens'].shape[2]
    st_dim = subs[0]['x_st'].shape[1]
    print(f'H&E dim: {he_dim}, ST dim: {st_dim}, shared: {args.shared_dim}')

    # reseed immediately before model construction so init is bit-deterministic
    # (any RNG consumption above this line otherwise drifts the Xavier state)
    torch.manual_seed(args.seed)
    model = LateFusion(he_dim, st_dim, args.shared_dim, args.dropout)
    model.eval()

    rows, all_z_he, all_z_st, all_he, all_st = [], [], [], [], []
    with torch.no_grad():
        for sub in subs:
            x_he_niche = torch.from_numpy(sub['x_he_niche'])
            x_he_tokens = torch.from_numpy(sub['x_he_tokens'])
            x_st = torch.from_numpy(sub['x_st'])
            z_he, z_st, _ = model(x_he_niche, x_he_tokens, x_st)
            z_he_np = z_he.cpu().numpy().astype(np.float32)
            z_st_np = z_st.cpu().numpy().astype(np.float32)
            for i, sid in enumerate(sub['spot_id']):
                rows.append({
                    'spot_id': sid,
                    'subarray': sub['subarray'],
                    'patient_id': sub['patient_id'],
                    'archetype': sub['archetype'],
                    'compartment': sub['compartment'][i],
                    'z_he': z_he_np[i].tolist(),
                    'z_st': z_st_np[i].tolist(),
                })
            all_z_he.append(z_he_np)
            all_z_st.append(z_st_np)
            all_he.append(sub['x_he_niche'])
            all_st.append(sub['x_st'])

    emb_df = pd.DataFrame(rows).set_index('spot_id')
    emb_df.to_parquet(run_dir / 'embeddings_test.parquet')

    Z_he = np.concatenate(all_z_he)
    Z_st = np.concatenate(all_z_st)
    X_he = np.concatenate(all_he)
    X_st = np.concatenate(all_st)

    metrics = retrieval_metrics(torch.from_numpy(Z_he), torch.from_numpy(Z_st))
    metrics['cka_before'] = cka_linear(X_he, X_st)
    metrics['cka_after'] = cka_linear(Z_he, Z_st)
    metrics['test_subarrays'] = len(subs)
    metrics['test_niches'] = len(emb_df)
    metrics['baseline'] = 'b4_random_init_late_fusion'
    (run_dir / 'metrics_h1_raw.json').write_text(json.dumps(metrics, indent=2))

    print(f'\nB4 H1 metrics: R@1={metrics["r1"]:.4f} R@5={metrics["r5"]:.4f} '
          f'AUC={metrics["auc"]:.4f} gap={metrics["alignment_gap"]:.4f} '
          f'CKA {metrics["cka_before"]:.3f}->{metrics["cka_after"]:.3f}')
    _rd = run_dir.resolve()
    print(f'artifacts in {_rd.relative_to(ROOT) if _rd.is_relative_to(ROOT) else _rd}')


if __name__ == '__main__':
    main()
