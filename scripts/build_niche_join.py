#!/usr/bin/env python
"""build_niche_join.py

materialize per-subarray niche-join parquets for the tnbc-92 alignment grid.

each output row = one niche (keyed on spot_id = niche center). columns:
    spot_id              str (index)
    subarray             str (TNBC{id}_CN{x}_{pos})
    patient_id           int
    archetype            int (Wang 9-archetype, per-patient)
    compartment          str (Wang 18-category dominant_class, per-spot)
    virchow2_niche       list[float32] len=1280 (pre-pooled at extraction)
    virchow2_cell_tokens list[float32] len=7*1280=8960 (self + 6 neighbors stacked)
    novae_niche          list[float32] len=64 (mean-pool self + 6 neighbors, z-score per sub)
    gpath2vec_niche      list[float32] len=512
    tls                  float32 (mean-pool self + 6 neighbors, nan if missing)
    mc_weights_niche     list[float32] len=14 (mean-pool self + 6 neighbors)
    neighbor_spot_ids    list[str] len=6 (provenance)

scope: intersection of novae_niche_full, virchow2_niche, and gpath2vec_output coverage
(expected 260 subarrays).

output:
    data/embeddings/niches/{subarray_key}.parquet
    data/embeddings/niches/manifest.json

usage:
    python scripts/build_niche_join.py
    python scripts/build_niche_join.py --subarrays TNBC1_CN1_C1 TNBC3_CN2_C1  # smoke test
    python scripts/build_niche_join.py --force  # overwrite existing
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import hashlib

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
np.random.seed(42)  # future-proof against any numpy ops that drift; this script is currently deterministic

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / 'data' / 'embeddings'
BIO = EMB / 'biological_signals'
CLIN = ROOT / 'data' / 'inputs' / 'clinical'

VIRCHOW2_NICHE_DIR = EMB / 'virchow2_niche'
VIRCHOW2_CELL_DIR = EMB / 'virchow2_cell'
NOVAE_DIR = EMB / 'novae_niche_full'
# default to v1 location for back-compat; override with --gpath2vec-parquet
DEFAULT_GPATH_PARQ = BIO / 'gpath2vec_output' / 'full_cohort_tf_low' / 'cluster_embeddings.parquet'
MC_LABELS_TSV = BIO / 'mc_labels.tsv'
MC_WEIGHTS_TSV = BIO / 'mc_weights.tsv'
TLS_TSV = BIO / 'tls_scores.tsv'
MORPH_TSV = BIO / 'morphology_labels.tsv'
CLINICAL_RDS = CLIN / 'Clinical.RDS'

# default out dir for back-compat; override with --out-dir
DEFAULT_OUT_DIR = EMB / 'niches'

K_NEIGHBORS = 6  # self + 6 = 7 token niche
# columns to exclude from the embedding-column slice in gpath2vec parquets.
# v1 had {subarray, patient}; v3 has {subarray, patient, spot_id, pixel_x, pixel_y}.
GPATH_META_COLS = ('subarray', 'patient', 'spot_id', 'pixel_x', 'pixel_y')


def parse_niche_id(niche_id: str) -> tuple[str, str]:
    """split `{subarray}::{spot_id}` -> (subarray, spot_id).

    centralized per v3_phase2_plan.md step 1 gate so the format check has one
    canonical implementation. asserts on shape so a malformed key fails loud.
    """
    parts = niche_id.split('::', 1)
    assert len(parts) == 2, f'niche_id must be `{{subarray}}::{{spot_id}}`, got: {niche_id!r}'
    sub, sid = parts
    assert sub and sid, f'niche_id has empty subarray or spot_id: {niche_id!r}'
    return sub, sid


# unit asserts: v3 sample keys (subarray prefixed with TNBC{id}) + v1 sample keys
# (same format - v1 also used `{subarray}::{spot_id}` indexing). if either form
# regresses, this trips on import, not at run time.
assert parse_niche_id('TNBC10_CN5_D2::10x12') == ('TNBC10_CN5_D2', '10x12')
assert parse_niche_id('TNBC1_CN1_C1::2x12') == ('TNBC1_CN1_C1', '2x12')


def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(chunk), b''):
            h.update(c)
    return h.hexdigest()


def _git_commit_sha() -> str:
    """current git HEAD short sha, or 'unknown' if not in a git repo."""
    try:
        r = subprocess.run(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip()[:12] if r.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def load_archetypes() -> dict[int, int]:
    """extract per-patient spatial archetype (1-9) from Clinical.RDS via Rscript."""
    r = subprocess.run(
        ['Rscript', '-e', (
            f'x <- readRDS("{CLINICAL_RDS}"); '
            'write.table(data.frame(tnbc_id=x$ST_TNBC_ID, '
            'archetype=x[["Spatial archetypes_defined_on_ST_global_pseudobulk"]]), '
            'file=stdout(), sep="\\t", row.names=FALSE, quote=FALSE)'
        )],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f'Rscript failed: {r.stderr[:200]}'
    import io
    df = pd.read_csv(io.StringIO(r.stdout), sep='\t')
    df = df.dropna()
    return {int(row['tnbc_id']): int(row['archetype']) for _, row in df.iterrows()}


def discover_matched_subarrays(gpath_parq: Path) -> list[str]:
    """return subarrays present in all three modalities."""
    novae_keys = {
        p.name.replace('_novae_embeddings.parquet', '')
        for p in NOVAE_DIR.glob('*_novae_embeddings.parquet')
    }
    virchow_keys = {
        p.name.replace('.npy', '')
        for p in VIRCHOW2_NICHE_DIR.glob('*.npy')
    }
    gp = pd.read_parquet(gpath_parq, columns=['subarray'])
    gpath_keys = set(gp['subarray'].unique())
    return sorted(novae_keys & virchow_keys & gpath_keys)


def slide_pos(subarray_key: str) -> str:
    """TNBC1_CN1_C1 -> CN1_C1 (mc_labels/mc_weights key format)."""
    parts = subarray_key.split('_', 1)
    return parts[1]  # drop TNBC{id}_


def niche_mean_pool(
    per_spot_vals: pd.DataFrame,
    center: str,
    neighbors: list[str],
    columns: list[str],
) -> np.ndarray:
    """mean-pool numeric columns over center + neighbors. nan-safe.

    returns (len(columns),) array; nan if center + all neighbors missing.
    """
    keys = [center] + list(neighbors)
    sub = per_spot_vals.reindex(keys)[columns]
    return sub.mean(axis=0, skipna=True).values.astype(np.float32)


def process_subarray(
    subarray_key: str,
    archetype_map: dict[int, int],
    gpath_by_sub: dict[str, pd.DataFrame],
    mcw_by_sub: dict[str, pd.DataFrame],
    mclab_by_sub: dict[str, pd.DataFrame],
    tls_by_sub: dict[str, pd.DataFrame],
    morph_by_sub: dict[str, pd.DataFrame],
    patient_by_sub: dict[str, int],
) -> tuple[pd.DataFrame, dict]:
    """build one per-subarray niche-join DataFrame + stats."""
    sp = slide_pos(subarray_key)
    patient_id = patient_by_sub.get(sp)
    archetype = archetype_map.get(patient_id) if patient_id is not None else None

    # virchow2 niche (pre-pooled) + meta
    vn = np.load(VIRCHOW2_NICHE_DIR / f'{subarray_key}.npy')
    meta = pd.read_csv(
        VIRCHOW2_NICHE_DIR / f'{subarray_key}_meta.tsv',
        sep='\t', index_col=0,
    )
    # virchow2 cell (per-spot) - row order same as meta
    vc = np.load(VIRCHOW2_CELL_DIR / f'{subarray_key}.npy')
    assert vn.shape[0] == vc.shape[0] == len(meta), (
        f'{subarray_key}: virchow2 row count mismatch '
        f'(niche={vn.shape[0]}, cell={vc.shape[0]}, meta={len(meta)})'
    )
    spot_to_row = {sid: i for i, sid in enumerate(meta.index)}

    # novae niche (parquet with neighbor_spot_ids + 64d novae_*)
    nv = pd.read_parquet(NOVAE_DIR / f'{subarray_key}_novae_embeddings.parquet')
    novae_cols = [c for c in nv.columns if c.startswith('novae_')]
    # canonical spot set = intersection of virchow2 meta and novae parquet
    shared = [s for s in meta.index if s in nv.index]

    # per-spot lookup tables for this subarray
    gp_sub = gpath_by_sub.get(subarray_key)  # indexed by spot_id
    mcw_sub = mcw_by_sub.get(sp)             # indexed by spot_id (slide_pos key)
    mclab_sub = mclab_by_sub.get(sp)         # indexed by spot_id (slide_pos key)
    tls_sub = tls_by_sub.get(subarray_key)   # indexed by spot_id
    morph_sub = morph_by_sub.get(subarray_key)  # indexed by spot_id

    # novae rows as DataFrame for easy reindex-based neighbor pooling
    nv_vals = nv[novae_cols]

    rows = []
    n_missing_gp = n_missing_tls = n_missing_morph = n_missing_neighbor = 0
    n_missing_mc = 0
    for sid in shared:
        i = spot_to_row[sid]
        neighbors_raw = nv.loc[sid, 'neighbor_spot_ids']
        # only keep neighbors also present in virchow2 meta (boundary robustness)
        neighbors = [n for n in neighbors_raw if n in spot_to_row]
        if len(neighbors) < K_NEIGHBORS:
            n_missing_neighbor += 1
            # pad by repeating self so the 7-token stack stays fixed-size
            neighbors = list(neighbors) + [sid] * (K_NEIGHBORS - len(neighbors))

        # virchow2 cell tokens: self + 6 neighbors, stacked (7, 1280)
        token_rows = [i] + [spot_to_row[n] for n in neighbors]
        vc_tokens = vc[token_rows].astype(np.float32)  # (7, 1280)

        # novae niche: mean-pool self + 6 neighbors
        novae_niche = nv_vals.reindex([sid] + list(neighbors)).mean(axis=0).values.astype(np.float32)

        # gpath2vec (already per-niche)
        if gp_sub is not None and sid in gp_sub.index:
            gp_niche = gp_sub.loc[sid].values.astype(np.float32)
        else:
            n_missing_gp += 1
            gp_niche = np.full(512, np.nan, dtype=np.float32)

        # tls: mean-pool self + 6 neighbors
        if tls_sub is not None:
            tls_keys = [sid] + list(neighbors)
            tls_vals = tls_sub.reindex(tls_keys)['tls_score']
            if tls_vals.notna().any():
                tls_niche = float(tls_vals.mean(skipna=True))
            else:
                n_missing_tls += 1
                tls_niche = np.nan
        else:
            n_missing_tls += 1
            tls_niche = np.nan

        # mc_weights: mean-pool self + 6 neighbors
        if mcw_sub is not None:
            mcw_cols = [f'mc{k}' for k in range(1, 15)]
            mcw_niche = niche_mean_pool(mcw_sub, sid, neighbors, mcw_cols)
        else:
            mcw_niche = np.full(14, np.nan, dtype=np.float32)

        # compartment: per-spot dominant_18class on center only (not pooled)
        if morph_sub is not None and sid in morph_sub.index:
            compartment = str(morph_sub.loc[sid, 'dominant_18class'])
        else:
            n_missing_morph += 1
            compartment = None

        # mc_megacluster: Wang per-spot 14-class NMF discrete label, niche-level
        # biological clustering target (per commit 2.5 + evaluation_question_audit
        # finding #1 - the proper niche-level label, not the patient-pseudobulk
        # `archetype`). center spot only, not pooled - discrete labels can't be
        # mean-pooled meaningfully.
        if mclab_sub is not None and sid in mclab_sub.index:
            mc_megacluster = int(mclab_sub.loc[sid, 'megacluster'])
        else:
            n_missing_mc += 1
            mc_megacluster = -1  # sentinel; eval code already filters mc_megacluster < 0

        rows.append({
            'spot_id': sid,
            'subarray': subarray_key,
            'patient_id': patient_id,
            'archetype': archetype,
            'compartment': compartment,
            'mc_megacluster': mc_megacluster,
            'virchow2_niche': vn[i].astype(np.float32).tolist(),
            'virchow2_cell_tokens': vc_tokens.flatten().tolist(),  # 7*1280
            'novae_niche_raw': novae_niche.tolist(),  # z-scored below
            'gpath2vec_niche': gp_niche.tolist(),
            'tls': tls_niche,
            'mc_weights_niche': mcw_niche.tolist(),
            'neighbor_spot_ids': list(neighbors_raw)[:K_NEIGHBORS],
        })

    df = pd.DataFrame(rows).set_index('spot_id')

    # intersection split per v3_phase2_plan.md: drop niches with NaN gpath2vec_niche
    # (= niches not embedded by the gpath2vec build). v1 dropped ~0 rows here; v3
    # drops ~70k cohort-wide (no-significant-pathway niches). align.py would also
    # drop these at training time, but filtering at niche-join build time gives
    # smaller parquets + a clean intersection-split semantic.
    n_before_intersect = len(df)
    gp_mat = np.stack([np.asarray(v, np.float32) for v in df['gpath2vec_niche']])
    keep = ~np.isnan(gp_mat).any(axis=1)
    df = df.loc[keep]
    n_intersect_dropped = int(n_before_intersect - len(df))

    # z-score novae_niche per subarray (AFTER intersection filter so the std
    # reflects the niches that will actually feed alignment)
    novae_mat = np.stack(df['novae_niche_raw'].values)
    mu = novae_mat.mean(axis=0, keepdims=True)
    sigma = novae_mat.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    novae_z = (novae_mat - mu) / sigma
    df['novae_niche'] = [row.astype(np.float32).tolist() for row in novae_z]
    df = df.drop(columns=['novae_niche_raw'])

    n_mc_post_intersect = int((df['mc_megacluster'] >= 0).sum())
    stats = {
        'subarray': subarray_key,
        'patient_id': patient_id,
        'archetype': archetype,
        'n_niches': len(df),
        'n_niches_pre_intersection': n_before_intersect,
        'n_intersection_dropped_no_gpath2vec': n_intersect_dropped,
        'n_boundary_niches_padded': n_missing_neighbor,
        'n_missing_gpath2vec': n_missing_gp,
        'n_missing_mc_megacluster_pre_intersection': n_missing_mc,
        'n_with_mc_megacluster_post_intersection': n_mc_post_intersect,
        'n_missing_tls': n_missing_tls,
        'n_missing_morphology': n_missing_morph,
    }
    return df, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subarrays', nargs='*', default=None,
                    help='optional subset (for smoke tests)')
    ap.add_argument('--force', action='store_true',
                    help='overwrite existing parquets')
    ap.add_argument('--progress-every', type=int, default=10)
    ap.add_argument('--gpath2vec-parquet', type=Path, default=DEFAULT_GPATH_PARQ,
                    help='path to the gpath2vec niche-cluster embeddings '
                         'parquet (default: v1 location for back-compat). '
                         'parquet must have (subarray::spot_id) index and '
                         'columns = embedding-dim cols + {subarray, patient, '
                         'spot_id, pixel_x, pixel_y} (extras tolerated).')
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR,
                    help='output directory for per-subarray niche-join '
                         'parquets + manifest.json (default: data/embeddings/'
                         'niches/). use a different path for non-v1 builds '
                         '(e.g. data/embeddings/niches_v3/) to keep v1 intact.')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assert args.gpath2vec_parquet.is_file(), (
        f'gpath2vec parquet not found: {args.gpath2vec_parquet}')

    print('loading global label tables...')
    archetype_map = load_archetypes()
    print(f'  archetype map: {len(archetype_map)} patients')

    mcw = pd.read_csv(MC_WEIGHTS_TSV, sep='\t')
    patient_by_sub = (
        mcw.drop_duplicates('subarray')
           .set_index('subarray')['patient_id']
           .astype(int)
           .to_dict()
    )
    mcw_by_sub = {sub: g.set_index('spot_id') for sub, g in mcw.groupby('subarray')}
    print(f'  mc_weights: {len(mcw)} rows, {len(mcw_by_sub)} subarrays (slide_pos keys)')

    mclab = pd.read_csv(MC_LABELS_TSV, sep='\t')
    mclab_by_sub = {sub: g.set_index('spot_id') for sub, g in mclab.groupby('subarray')}
    print(f'  mc_labels: {len(mclab)} rows, {len(mclab_by_sub)} subarrays (slide_pos keys)')

    tls = pd.read_csv(TLS_TSV, sep='\t')
    tls_by_sub = {sub: g.set_index('spot_id') for sub, g in tls.groupby('subarray_id')}
    print(f'  tls: {len(tls)} rows, {len(tls_by_sub)} subarrays')

    morph = pd.read_csv(MORPH_TSV, sep='\t')
    morph_by_sub = {sub: g.set_index('spot_id') for sub, g in morph.groupby('subarray_id')}
    print(f'  morphology: {len(morph)} rows, {len(morph_by_sub)} subarrays')

    print(f'loading gpath2vec niche embeddings from '
          f'{args.gpath2vec_parquet.relative_to(ROOT) if args.gpath2vec_parquet.is_relative_to(ROOT) else args.gpath2vec_parquet}...')
    gpath_sha256 = _sha256_of_file(args.gpath2vec_parquet)
    print(f'  gpath2vec parquet sha256: {gpath_sha256}')
    gp = pd.read_parquet(args.gpath2vec_parquet)
    # tolerate v1 (cols={subarray, patient} + 512 emb) and v3 (cols + spot_id, pixel_x, pixel_y)
    gp_cols = [c for c in gp.columns if c not in GPATH_META_COLS]
    if 'spot_id' not in gp.columns:
        # v1 parquets carry only (subarray, patient) - derive spot_id from the
        # `{subarray}::{spot_id}` index. v3 stores spot_id directly.
        gp['spot_id'] = gp.index.to_series().map(lambda s: parse_niche_id(s)[1])
    gpath_by_sub = {
        sub: g.set_index('spot_id')[gp_cols]
        for sub, g in gp.groupby('subarray')
    }
    print(f'  gpath2vec: {len(gp)} niches, {len(gpath_by_sub)} subarrays, dim={len(gp_cols)}')
    del gp

    matched = discover_matched_subarrays(args.gpath2vec_parquet)
    if args.subarrays:
        matched = [s for s in matched if s in set(args.subarrays)]
    print(f'\nmatched subarrays to process: {len(matched)}')

    summary = []
    for i, subarray_key in enumerate(matched):
        out_path = args.out_dir / f'{subarray_key}.parquet'
        if out_path.exists() and not args.force:
            summary.append({'subarray': subarray_key, 'status': 'exists'})
            continue
        try:
            df, stats = process_subarray(
                subarray_key, archetype_map, gpath_by_sub,
                mcw_by_sub, mclab_by_sub, tls_by_sub, morph_by_sub, patient_by_sub,
            )
            df.to_parquet(out_path)
            stats['status'] = 'ok'
            summary.append(stats)
        except Exception as e:
            summary.append({
                'subarray': subarray_key, 'status': 'failed', 'error': str(e)[:120],
            })
            print(f'  [{i+1}/{len(matched)}] {subarray_key}: FAILED - {str(e)[:120]}')
            continue
        if (i + 1) % args.progress_every == 0:
            ok = sum(1 for s in summary if s['status'] == 'ok')
            print(f'  [{i+1}/{len(matched)}] ok={ok}')

    # manifest
    n_ok = sum(1 for s in summary if s['status'] == 'ok')
    n_exists = sum(1 for s in summary if s['status'] == 'exists')
    n_failed = sum(1 for s in summary if s['status'] == 'failed')
    gpath_rel = (args.gpath2vec_parquet.relative_to(ROOT)
                 if args.gpath2vec_parquet.is_relative_to(ROOT)
                 else args.gpath2vec_parquet)

    # aggregate post-intersection niche totals + mc_megacluster coverage
    ok_rows = [s for s in summary if s.get('status') == 'ok']
    n_niches_total = int(sum(s.get('n_niches', 0) for s in ok_rows))
    n_with_mc_total = int(sum(s.get('n_with_mc_megacluster_post_intersection', 0) for s in ok_rows))
    mc_coverage = (n_with_mc_total / n_niches_total) if n_niches_total else 0.0
    n_dropped_no_gpath = int(sum(s.get('n_intersection_dropped_no_gpath2vec', 0) for s in ok_rows))

    manifest = {
        'run_type': 'niche_join_table',
        'git_commit_sha': _git_commit_sha(),
        'n_subarrays_enumerated': len(matched),
        'n_ok': n_ok,
        'n_exists': n_exists,
        'n_failed': n_failed,
        'n_niches_total_post_intersection': n_niches_total,
        'n_niches_dropped_no_gpath2vec_coverage': n_dropped_no_gpath,
        'mc_label_join_coverage': mc_coverage,
        'mc_label_join_n_covered': n_with_mc_total,
        'columns': {
            'virchow2_niche_dim': 1280,
            'virchow2_cell_tokens_shape': [7, 1280],
            'novae_niche_dim': 64,
            'gpath2vec_niche_dim': len(gp_cols),
            'mc_weights_niche_dim': 14,
            'mc_megacluster_label_set': 'Wang per-spot 14-class NMF (mc_labels.tsv)',
            'novae_niche_normalization': 'z-score per subarray',
        },
        'sources': {
            'virchow2_niche': str(VIRCHOW2_NICHE_DIR.relative_to(ROOT)),
            'virchow2_cell': str(VIRCHOW2_CELL_DIR.relative_to(ROOT)),
            'novae_niche': str(NOVAE_DIR.relative_to(ROOT)),
            'gpath2vec': str(gpath_rel),
            'gpath2vec_sha256': gpath_sha256,
            'mc_weights': str(MC_WEIGHTS_TSV.relative_to(ROOT)),
            'mc_labels': str(MC_LABELS_TSV.relative_to(ROOT)),
            'tls': str(TLS_TSV.relative_to(ROOT)),
            'morphology': str(MORPH_TSV.relative_to(ROOT)),
            'archetype': str(CLINICAL_RDS.relative_to(ROOT)),
        },
        'out_dir': str(args.out_dir.relative_to(ROOT) if args.out_dir.is_relative_to(ROOT) else args.out_dir),
        'per_subarray': summary,
    }
    with open(args.out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\ndone: ok={n_ok}, exists={n_exists}, failed={n_failed}')
    _man = (args.out_dir / 'manifest.json').resolve()
    _rel = _man.relative_to(ROOT) if _man.is_relative_to(ROOT) else _man
    print(f'manifest: {_rel}')


if __name__ == '__main__':
    main()