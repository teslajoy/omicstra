#!/usr/bin/env python
"""eval_alignment_biology.py - port of gpath2vec biology vs patient validation
to alignment runs.

mirrors /Users/sanati/BForePC/gpath2vec/scripts/embedding_vs_rawea_biology.py
adapted to alignment outputs at niche resolution.

method:
  1. reconstruct the trained model from run_config.json + checkpoint.pt
     (or re-fit classical projection from the same train split)
  2. re-encode all 260 matched subarrays through the model
  3. aggregate to subarray-level mean embedding (L2-norm before+after)
  4. build subarray x subarray cosine matrices for {z_he, z_st, z_mean}
  5. compare each against raw baseline cosines (raw H&E niche, raw ST features)
     - rho_with_null: spearman vs shuffled-matrix null
  6. permutation tests for biology (MC_global, MC_tumor, TIME) and patient_id
     - 1e5 perms, one-tailed, bonferroni
  7. save per-run parquet for direct comparison across runs

usage:
    python scripts/eval_alignment_biology.py --run-id R1
    python scripts/eval_alignment_biology.py --run-id R4 --n-permutations 100000
    python scripts/eval_alignment_biology.py --run-id B1   # classical CCA refit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

import click

ROOT = Path(__file__).resolve().parents[1]
# mutable defaults; main() overrides from --runs-dir / --niches-dir for v3
RUNS_ROOT = ROOT / 'runs' / 'tnbc-92'
NICHES_DIR = ROOT / 'data' / 'embeddings' / 'niches'
CLINICAL = ROOT / 'data' / 'inputs' / 'clinical' / 'Clinical.xlsx'

# reuse model + classical builders
sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import Config, build_model, load_subarray, discover_subarrays  # noqa: E402
from align_classical import run_cca, run_procrustes, run_unaligned, build_matrices  # noqa: E402


# -------------------------------------------------------------------------- #
# label loader (same convention as gpath2vec script)
# -------------------------------------------------------------------------- #

def load_tnbc_labels(subarray_keys: list[str]) -> tuple[np.ndarray, dict]:
    """parse Clinical.xlsx supplementary table, propagate per-patient labels
    to subarrays. patient_id is the first underscore chunk of the subarray key
    (e.g. 'TNBC1_CN1_C1' -> 'TNBC1')."""
    clin = pd.read_excel(CLINICAL, sheet_name='Supplementary Data 1')
    clin['patient_id'] = 'TNBC' + clin['ST_TNBC_ID'].astype(str)
    cols = {
        'MC_global': 'Bareche_molecular_subtype_defined_on_global_pseudobulk',
        'MC_tumor':  'Bareche_molecular_subtype_defined_on_tumor_pseudobulk',
        'TIME':      'TIME_classes_expression_global_pseudobulk',
    }
    small = clin[['patient_id'] + list(cols.values())].rename(
        columns={v: k for k, v in cols.items()}
    )
    ptmap = {
        r.patient_id: {k: getattr(r, k) for k in cols}
        for r in small.itertuples(index=False)
    }
    patients = np.array([s.split('_')[0] for s in subarray_keys])
    labels = {
        k: np.array([str(ptmap.get(p, {}).get(k, 'NA')) for p in patients])
        for k in cols
    }
    return patients, labels


# -------------------------------------------------------------------------- #
# encoders: trained model OR classical projection
# -------------------------------------------------------------------------- #

ST_FEATURES = ('novae_niche', 'gpath2vec_niche')


def reconstruct_trained_model(run_id: str, sample_sub: dict) -> tuple[torch.nn.Module, Config]:
    cfg_path = RUNS_ROOT / run_id / 'run_config.json'
    ckpt_path = RUNS_ROOT / run_id / 'checkpoint.pt'
    cfg = Config.from_file(cfg_path)
    he_dim = sample_sub['x_he_niche'].shape[1]
    he_tile_dim = sample_sub['x_he_tokens'].shape[2]
    st_dim = sample_sub['x_st'].shape[1]
    model = build_model(cfg, he_dim, he_tile_dim, st_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    return model, cfg


def encode_trained(model, sub: dict) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        z_he, z_st, _ = model(
            torch.from_numpy(sub['x_he_niche']),
            torch.from_numpy(sub['x_he_tokens']),
            torch.from_numpy(sub['x_st']),
        )
    return z_he.cpu().numpy().astype(np.float32), z_st.cpu().numpy().astype(np.float32)


def fit_classical(run_id: str, all_subs: list[dict]) -> dict:
    """re-fit classical projection on the run's train patients, return projector."""
    split = json.loads((RUNS_ROOT / run_id / 'split.json').read_text())
    train_p = set(split['train_patients'])
    train_subs = [s for s in all_subs if s['patient_id'] in train_p]
    cfg = json.loads((RUNS_ROOT / run_id / 'run_config.json').read_text())
    baseline = cfg['baseline']
    shared_dim = cfg['shared_dim']
    X_he_tr, X_st_tr, _ = build_matrices(train_subs)
    return {'baseline': baseline, 'shared_dim': shared_dim,
            'X_he_train': X_he_tr, 'X_st_train': X_st_tr}


def encode_classical(state: dict, sub: dict) -> tuple[np.ndarray, np.ndarray]:
    runner = {'cca': run_cca, 'procrustes': run_procrustes,
              'unaligned': run_unaligned}[state['baseline']]
    z_he, z_st, _ = runner(
        state['X_he_train'], state['X_st_train'],
        sub['x_he_niche'], sub['x_st'], state['shared_dim'],
    )
    return z_he, z_st


# -------------------------------------------------------------------------- #
# subarray-level aggregation
# -------------------------------------------------------------------------- #

def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def subarray_mean_pool(per_niche: np.ndarray) -> np.ndarray:
    """L2-norm niches, mean, then L2-norm the mean. matches gpath2vec convention."""
    if per_niche.shape[0] == 0:
        return np.zeros(per_niche.shape[1], dtype=np.float32)
    return l2(l2(per_niche).mean(axis=0))


def cosine_matrix(M: np.ndarray) -> np.ndarray:
    Mn = l2(M)
    return np.clip(Mn @ Mn.T, -1.0, 1.0).astype(np.float32)


# -------------------------------------------------------------------------- #
# stats (lifted from gpath2vec script, unchanged math)
# -------------------------------------------------------------------------- #

def upper_triangle(arr):
    iu = np.triu_indices(arr.shape[0], k=1)
    return arr[iu], iu


def rho_with_null(emb_sim, ea_sim, n_perm, seed):
    flat_emb, _ = upper_triangle(emb_sim)
    flat_ea, _ = upper_triangle(ea_sim)
    obs_rho, _ = spearmanr(flat_emb, flat_ea)
    obs_pearson = float(np.corrcoef(flat_emb, flat_ea)[0, 1])
    rng = np.random.default_rng(seed)
    n = emb_sim.shape[0]
    null = np.empty(n_perm, dtype=np.float32)
    for i in range(n_perm):
        perm = rng.permutation(n)
        emb_perm = emb_sim[np.ix_(perm, perm)]
        flat_perm, _ = upper_triangle(emb_perm)
        r, _ = spearmanr(flat_perm, flat_ea)
        null[i] = r
    null_mean = float(null.mean())
    null_std = float(null.std())
    z = (float(obs_rho) - null_mean) / null_std if null_std > 0 else float('nan')
    p = float((null >= float(obs_rho)).sum() + 1) / (n_perm + 1)
    return {'rho': float(obs_rho), 'pearson': obs_pearson,
            'null_mean': null_mean, 'null_std': null_std,
            'z': z, 'p_value': p, 'n_pairs': int(len(flat_emb)),
            'n_permutations': int(n_perm)}


def _pair_indices_and_same_patient_mask(patients):
    iu = np.triu_indices(len(patients), k=1)
    same_pt = (patients[:, None] == patients[None, :])[iu]
    return iu, same_pt


def _biology_delta(sim_pairs, same_pt, labels, iu, valid_mask):
    same_lb = (labels[:, None] == labels[None, :])[iu]
    cross = (~same_pt) & valid_mask
    A = cross & same_lb
    B = cross & ~same_lb
    m_same = float(sim_pairs[A].mean()) if A.any() else float('nan')
    m_diff = float(sim_pairs[B].mean()) if B.any() else float('nan')
    return m_same - m_diff, int(A.sum()), int(B.sum()), m_same, m_diff


def _patient_delta(sim_pairs, patient_vec, iu):
    same_pt = (patient_vec[:, None] == patient_vec[None, :])[iu]
    m_same = float(sim_pairs[same_pt].mean()) if same_pt.any() else float('nan')
    m_diff = float(sim_pairs[~same_pt].mean()) if (~same_pt).any() else float('nan')
    return m_same - m_diff, int(same_pt.sum()), int((~same_pt).sum()), m_same, m_diff


def biology_perm_test(sim, patients, labels, n_perm, seed, return_null=False):
    iu, same_pt = _pair_indices_and_same_patient_mask(patients)
    sim_pairs = sim[iu]
    valid = ((labels != 'NA')[:, None] & (labels != 'NA')[None, :])[iu]
    obs, n_same, n_diff, m_same, m_diff = _biology_delta(sim_pairs, same_pt, labels, iu, valid)
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patients)
    patient_to_idx = pd.factorize(patients)[0]
    patient_label = np.array([labels[np.where(patients == p)[0][0]] for p in unique_patients])
    null = np.empty(n_perm, dtype=np.float32)
    for i in range(n_perm):
        perm = rng.permutation(len(unique_patients))
        shuffled = patient_label[perm][patient_to_idx]
        d, *_ = _biology_delta(sim_pairs, same_pt, shuffled, iu, valid)
        null[i] = d
    p = float((null >= obs).sum() + 1) / (n_perm + 1)
    if return_null:
        return obs, float(null.mean()), float(null.std()), p, n_same, n_diff, m_same, m_diff, null
    return obs, float(null.mean()), float(null.std()), p, n_same, n_diff, m_same, m_diff


def patient_perm_test(sim, patients, n_perm, seed, return_null=False):
    iu, _ = _pair_indices_and_same_patient_mask(patients)
    sim_pairs = sim[iu]
    obs, n_same, n_diff, m_same, m_diff = _patient_delta(sim_pairs, patients, iu)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float32)
    pv = patients.copy()
    for i in range(n_perm):
        rng.shuffle(pv)
        d, *_ = _patient_delta(sim_pairs, pv, iu)
        null[i] = d
    p = float((null >= obs).sum() + 1) / (n_perm + 1)
    if return_null:
        return obs, float(null.mean()), float(null.std()), p, n_same, n_diff, m_same, m_diff, null
    return obs, float(null.mean()), float(null.std()), p, n_same, n_diff, m_same, m_diff


# -------------------------------------------------------------------------- #
# main
# -------------------------------------------------------------------------- #

@click.command()
@click.option('--run-id', required=True)
@click.option('--runs-dir', type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help='parent dir holding {run_id}/. defaults to runs/tnbc-92/. use runs/tnbc-92_v3/ for v3.')
@click.option('--niches-dir', type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help='niche-join parquet dir. defaults to data/embeddings/niches/. use niches_v3 for v3.')
@click.option('--n-permutations', type=int, default=100000, show_default=True)
@click.option('--n-rho-null', type=int, default=1000, show_default=True)
@click.option('--seed', type=int, default=42, show_default=True)
def main(run_id, runs_dir, niches_dir, n_permutations, n_rho_null, seed):
    global RUNS_ROOT, NICHES_DIR
    if runs_dir is not None:
        RUNS_ROOT = runs_dir.resolve()
    if niches_dir is not None:
        NICHES_DIR = niches_dir.resolve()
    from types import SimpleNamespace
    args = SimpleNamespace(run_id=run_id, n_permutations=n_permutations,
                           n_rho_null=n_rho_null, seed=seed)

    run_dir = RUNS_ROOT / args.run_id
    out_dir = run_dir / 'eval'
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f'=== biology vs patient validation: {args.run_id} ===')
    print(f'  runs_dir: {RUNS_ROOT}  niches_dir: {NICHES_DIR}')

    # detect run kind
    run_cfg = json.loads((run_dir / 'run_config.json').read_text())
    is_classical = 'baseline' in run_cfg
    print(f'  type: {"classical" if is_classical else "trained"}')

    # read st_features from run config (R6 drops gpath2vec, etc.)
    st_features = tuple(run_cfg.get('st_features', ST_FEATURES))
    print(f'  st_features: {st_features}')

    # load all matched subarrays from the configured niche dir
    paths = discover_subarrays(NICHES_DIR)
    print(f'  loading {len(paths)} subarrays...')
    subs_raw = [load_subarray(p, st_features) for p in paths]
    subs = [s for s in subs_raw if s['patient_id'] is not None and s['archetype'] is not None]
    if len(subs) < len(subs_raw):
        print(f'    dropped {len(subs_raw)-len(subs)} subarrays missing patient_id/archetype')

    # build encoder
    if is_classical:
        state = fit_classical(args.run_id, subs)
        encode = lambda sub: encode_classical(state, sub)
    else:
        model, _ = reconstruct_trained_model(args.run_id, subs[0])
        encode = lambda sub: encode_trained(model, sub)

    # encode all subarrays, mean-pool to subarray level
    print('  encoding + subarray-mean-pooling...')
    keys, mean_he, mean_st, mean_raw_he, mean_raw_st = [], [], [], [], []
    for sub in subs:
        z_he, z_st = encode(sub)
        keys.append(sub['subarray'])
        mean_he.append(subarray_mean_pool(z_he))
        mean_st.append(subarray_mean_pool(z_st))
        # raw baselines: raw H&E niche + raw ST (novae+gpath2vec)
        mean_raw_he.append(subarray_mean_pool(sub['x_he_niche']))
        mean_raw_st.append(subarray_mean_pool(sub['x_st']))
    Z_he = np.stack(mean_he)
    Z_st = np.stack(mean_st)
    Z_mean = l2(0.5 * (Z_he + Z_st))
    R_he = np.stack(mean_raw_he)
    R_st = np.stack(mean_raw_st)

    # subarray cosine matrices
    cos_z_he = cosine_matrix(Z_he)
    cos_z_st = cosine_matrix(Z_st)
    cos_z_mean = cosine_matrix(Z_mean)
    cos_raw_he = cosine_matrix(R_he)
    cos_raw_st = cosine_matrix(R_st)

    # labels
    patients, labels_by_key = load_tnbc_labels(keys)
    print(f'  aligned: {len(keys)} subarrays, {len(set(patients))} patients, '
          f'biology labels: {list(labels_by_key.keys())}')

    # rho_with_null: each embedding view vs each raw baseline
    rho_rows = []
    print(f'\n  rho-with-null (n_rho_null={args.n_rho_null}):')
    for emb_name, emb_sim in [('z_he', cos_z_he), ('z_st', cos_z_st), ('z_mean', cos_z_mean)]:
        for raw_name, raw_sim in [('raw_he', cos_raw_he), ('raw_st', cos_raw_st)]:
            r = rho_with_null(emb_sim, raw_sim, args.n_rho_null, args.seed)
            r.update({'run_id': args.run_id, 'emb_view': emb_name, 'raw_baseline': raw_name})
            rho_rows.append(r)
            print(f'    {emb_name:>7} vs {raw_name:<8}: rho={r["rho"]:+.4f} '
                  f'null={r["null_mean"]:+.4f}±{r["null_std"]:.4f} '
                  f'z={r["z"]:+.2f} p={r["p_value"]:.3g}')

    # permutation tests on each EMBEDDING view + raw baselines (for ratio compare)
    # bonferroni across all biology+patient tests for this run
    n_tests_total = len(['z_he', 'z_st', 'z_mean', 'raw_he', 'raw_st']) * (len(labels_by_key) + 1)
    bonferroni_alpha = 0.05 / max(n_tests_total, 1)
    p_floor = 1.0 / (args.n_permutations + 1)
    print(f'\n  permutation tests (n={args.n_permutations}), p_floor={p_floor:.2e}, '
          f'bonferroni_alpha={bonferroni_alpha:.5f}')
    print(f'  {"emb_view":<10} {"label":<12} {"same":>7} {"diff":>7} '
          f'{"delta":>9} {"null_mean":>10} {"null_std":>9} {"z":>7} {"p":>10}')

    # run all tests, keeping full null arrays (seed=42 -> deterministic for paired tests)
    rows = []
    nulls = {}  # (view, label, test_type) -> np.ndarray (n_perm,)
    for view_name, view_sim in [('z_he', cos_z_he), ('z_st', cos_z_st),
                                 ('z_mean', cos_z_mean),
                                 ('raw_he', cos_raw_he), ('raw_st', cos_raw_st)]:
        for key, labs in labels_by_key.items():
            obs, nm, ns, p, n_same, n_diff, m_same, m_diff, null_arr = biology_perm_test(
                view_sim, patients, labs, args.n_permutations, args.seed, return_null=True,
            )
            z = obs / ns if ns > 0 else float('nan')
            rows.append({
                'run_id': args.run_id, 'view': view_name, 'label': key,
                'test_type': 'biology', 'mean_same': m_same, 'mean_diff': m_diff,
                'delta': obs, 'null_mean': nm, 'null_std': ns,
                'null_skew': float(pd.Series(null_arr).skew()),
                'null_kurt': float(pd.Series(null_arr).kurt()),
                'null_q05': float(np.quantile(null_arr, 0.05)),
                'null_q95': float(np.quantile(null_arr, 0.95)),
                'obs_percentile_in_null': float((null_arr < obs).mean()),
                'z': z, 'p_value': p, 'p_floor': p_floor,
                'n_same_pairs': n_same, 'n_diff_pairs': n_diff,
            })
            nulls[(view_name, key, 'biology')] = null_arr.astype(np.float32)
            print(f'  {view_name:<10} {key:<12} {m_same:7.3f} {m_diff:7.3f} '
                  f'{obs:+9.4f} {nm:+10.4f} {ns:9.4f} {z:+7.2f} {p:10.3g}')
        obs, nm, ns, p, n_same, n_diff, m_same, m_diff, null_arr = patient_perm_test(
            view_sim, patients, args.n_permutations, args.seed, return_null=True,
        )
        z = obs / ns if ns > 0 else float('nan')
        rows.append({
            'run_id': args.run_id, 'view': view_name, 'label': 'patient_id',
            'test_type': 'patient', 'mean_same': m_same, 'mean_diff': m_diff,
            'delta': obs, 'null_mean': nm, 'null_std': ns,
            'null_skew': float(pd.Series(null_arr).skew()),
            'null_kurt': float(pd.Series(null_arr).kurt()),
            'null_q05': float(np.quantile(null_arr, 0.05)),
            'null_q95': float(np.quantile(null_arr, 0.95)),
            'obs_percentile_in_null': float((null_arr < obs).mean()),
            'z': z, 'p_value': p, 'p_floor': p_floor,
            'n_same_pairs': n_same, 'n_diff_pairs': n_diff,
        })
        nulls[(view_name, 'patient_id', 'patient')] = null_arr.astype(np.float32)
        print(f'  {view_name:<10} {"patient_id":<12} {m_same:7.3f} {m_diff:7.3f} '
              f'{obs:+9.4f} {nm:+10.4f} {ns:9.4f} {z:+7.2f} {p:10.3g}')

    # BH-FDR per test-family: biology and patient separately (they have different nulls)
    try:
        from statsmodels.stats.multitest import multipletests
        df_rows = pd.DataFrame(rows)
        df_rows['q_bh'] = np.nan
        for test_type, fam in df_rows.groupby('test_type'):
            _, q, _, _ = multipletests(fam['p_value'].values, alpha=0.05, method='fdr_bh')
            df_rows.loc[fam.index, 'q_bh'] = q
        df_rows['significant_q05'] = df_rows['q_bh'] <= 0.05
    except ImportError:
        print('  WARN: statsmodels not available, skipping BH-FDR')
        df_rows = pd.DataFrame(rows)

    # persist main results + rho + full null distributions (zstd-compressed)
    df_rows.to_parquet(out_dir / 'biology.parquet')
    pd.DataFrame(rho_rows).to_parquet(out_dir / 'biology_rho.parquet')

    # nulls as wide parquet: one row per (view, label, test_type), columns = perm_0..perm_N
    # zstd compression reduces ~4x on the repeated-float null arrays
    null_records = []
    for (view, label, test_type), arr in nulls.items():
        rec = {'run_id': args.run_id, 'view': view, 'label': label, 'test_type': test_type}
        for i, v in enumerate(arr):
            rec[f'p{i}'] = v
        null_records.append(rec)
    try:
        pd.DataFrame(null_records).to_parquet(
            out_dir / 'biology_nulls.parquet', compression='zstd',
        )
    except Exception:
        # fallback if zstd not supported by pyarrow build
        pd.DataFrame(null_records).to_parquet(out_dir / 'biology_nulls.parquet')

    print(f'\n  saved:')
    print(f'    {(out_dir / "biology.parquet").relative_to(ROOT)}  (obs deltas + BH-FDR q)')
    print(f'    {(out_dir / "biology_rho.parquet").relative_to(ROOT)}  (geometric faithfulness)')
    print(f'    {(out_dir / "biology_nulls.parquet").relative_to(ROOT)}  (full {args.n_permutations}-perm null distributions, zstd)')
    print(f'\n  done in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()