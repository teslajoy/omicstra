"""noise-floor validation harness for gpath2vec niche embeddings.

answers "is this build tracking biology or noise?" via two falsifiable tests run
in parallel on N arms (any pkl pair/triple). this is methodology-validation code
for the paper, not a one-off - it gates which gpath2vec build is considered
publishable. lives in scripts/, not _scratch/, because the noise-floor read is
load-bearing for v3 (the canonical Fisher result) vs v1 (retired) vs v2 (the
noise-floor demonstration).

TEST 1 - MC coherence permutation z-test (niche-level)
    per arm, for each label in {mc_megacluster, patient_id}:
    delta_obs = mean[cos(i,j) | same label] - mean[cos(i,j) | diff label]
    null: shuffle labels (matched-null preserves cluster sizes) N_PERMS times
    report z = (delta_obs - mean_null) / std_null
    biology validation: high MC z = embedding encodes niche-level Wang 14-class biology
    leakage check: patient z too high (vs MC) = embedding is dominated by patient ID
    geometry sanity: same/diff mean cos ~0.88 across all labels = dim-collapse

TEST 2 - per-pathway internal consistency (subarray-level Spearman)
    per arm, for each of the 5 named Reactome targets:
    construct a 1-d per-niche pathway signal = cos(niche, mean(embedded members))
    compute Spearman(signal, niche AUCell score for that pathway) within each test subarray
    report mean +- SE across 38 test subarrays
    biology validation: if cos(niche, immune-rollup) correlates with the niche's actual
    immune AUCell, the embedding internally encodes pathway-specific signal
    (vs accidental dimensions that survive CCA by dimension count alone)

scope: R4's 45,661 held-out test niches (14 patients, 38 subarrays). cross-patient
test set ensures the read is honest for downstream alignment-time interpretation.

usage:
    # default 3-arm read (v1 arm A vs v3 madmean) into v3 eval dir:
    python scripts/eval_h3_gpath2vec_noise_floor.py

    # arbitrary arms via CLI; each --arm NAME PKL_PATH:
    python scripts/eval_h3_gpath2vec_noise_floor.py \\
        --arm v1_armA data/.../full_cohort_tf_low_arm_A/embeddings.pkl \\
        --arm v3_madmean data/.../fisher_madmean_low_embeddings.pkl \\
        --out-dir runs/tnbc-92/eval/H3/some_target_dir

outputs (parallel to other H3 evals):
    {out-dir}/noise_floor_mc.parquet      per-arm MC + patient z, geometry stats
    {out-dir}/noise_floor_aucell.parquet  per-arm per-pathway Spearman summary
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / 'data' / 'embeddings' / 'biological_signals'
RUNS = ROOT / 'runs' / 'tnbc-92'

# default arms (paper canon): v1 arm A (retired baseline) + v3 madmean (canonical)
V1_PKL = BIO / 'gpath2vec_output' / 'full_cohort_tf_low_arm_A' / 'embeddings.pkl'
V2_PKL = (ROOT / 'data' / 'embeddings' / 'gpath2vec'
          / 'aucell_level_all_topk50_dim128_e1_s1234' / 'full' / 'embeddings.pkl')
V3_PKL = (ROOT / 'data' / 'embeddings' / 'gpath2vec'
          / 'fisher_madmean_low_dim512_e5_s1234'
          / 'fisher_madmean_low_embeddings.pkl')
DEFAULT_ARMS = [('v1_armA', V1_PKL), ('v3_madmean', V3_PKL)]
AUCELL_5 = BIO / 'niche_aucell_5targets.parquet'
MC_LABELS = BIO / 'mc_labels.tsv'
R4_EMB = RUNS / 'R4' / 'embeddings_test.parquet'
OUT_DIR = RUNS / 'eval' / 'H3' / 'pathway_cca_gpath2vec_v3_fisher_madmean_low'

NAMED_TARGETS = {
    'TGF-beta_Signaling':    'R-HSA-170834',
    'Immune_System':         'R-HSA-168256',
    'ECM_Organization':      'R-HSA-1474244',
    'Cell_Cycle':            'R-HSA-1640170',
    'Programmed_Cell_Death': 'R-HSA-5357801',
}

# ============================================================
# load + parse
# ============================================================

def niche_key_to_id(k: str) -> str:
    s = k.rsplit('__', 1)[-1]
    return s[len('cluster_'):] if s.startswith('cluster_') else s


def load_emb(pkl_path: Path):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    rhsa = {k: np.asarray(v, np.float32) for k, v in d.items()
            if isinstance(k, str) and k.startswith('R-HSA-')}
    niche_dict = {}
    for k, v in d.items():
        if isinstance(k, str) and 'cluster_' in k:
            nid = niche_key_to_id(k)
            if nid not in niche_dict:
                niche_dict[nid] = np.asarray(v, np.float32)
    return rhsa, niche_dict


def load_test_niches():
    """45661 test niche ids + (patient_id, subarray) join keys."""
    e = pd.read_parquet(R4_EMB, columns=['subarray', 'patient_id', 'archetype'])
    e = e.reset_index().rename(columns={'spot_id': 'spot_id'})
    e['niche_id'] = e['subarray'].astype(str) + '::' + e['spot_id'].astype(str)
    return e


def load_mc_labels():
    """Wang per-spot 14-class mc_megacluster. key format gotcha per
    project_mc_weights_column_ambiguity.md: mc_labels.tsv uses CN1_C1 keys
    (no TNBC prefix); strip it from our TNBC{x}_CN..._C... subarray ids."""
    mc = pd.read_csv(MC_LABELS, sep='\t')
    # columns vary - find the 14-class column
    cand = [c for c in mc.columns if 'megacluster' in c.lower()]
    assert cand, f'mc_labels columns: {list(mc.columns)} - no megacluster col'
    mc_col = cand[0]
    print(f'  using mc col: {mc_col}')
    # rebuild niche_id from subarray + spot_id; account for the no-TNBC-prefix
    # format if columns name them 'subarray'/'spot_id' or similar
    sub_col = next((c for c in mc.columns if c.lower() in ('subarray', 'sample', 'slide_pos')), None)
    spot_col = next((c for c in mc.columns if c.lower() in ('spot_id', 'spot', 'barcode')), None)
    assert sub_col and spot_col, f'mc_labels cols: {list(mc.columns)}'
    return mc, mc_col, sub_col, spot_col


# ============================================================
# test 1: MC coherence permutation z-test
# ============================================================

def cosine_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def stratified_subsample(labels: np.ndarray, n_total: int, seed: int):
    """stratified subsample of labels - keeps class balance."""
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(labels, return_counts=True)
    # proportional sample size per class
    frac = n_total / len(labels)
    target = np.maximum(2, np.round(counts * frac).astype(int))
    out = []
    for u, t in zip(unique, target):
        idx = np.where(labels == u)[0]
        take = rng.choice(idx, size=min(t, len(idx)), replace=False)
        out.extend(take.tolist())
    return np.array(sorted(out))


def coherence_zscore(C: np.ndarray, labels: np.ndarray, n_perms: int, seed: int):
    """matched-null z-score on same-vs-diff cosine. C is square (n x n)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    # upper triangle pair mask (exclude diagonal)
    iu, ju = np.triu_indices(n, k=1)
    cos_pairs = C[iu, ju]

    def _delta(lbl):
        same = lbl[iu] == lbl[ju]
        ms = cos_pairs[same].mean()
        md = cos_pairs[~same].mean() if (~same).any() else np.nan
        return ms - md, int(same.sum()), int((~same).sum()), ms, md

    delta_obs, n_same, n_diff, mean_same, mean_diff = _delta(labels)
    null = np.empty(n_perms, np.float32)
    perm = labels.copy()
    for k in range(n_perms):
        rng.shuffle(perm)
        same_k = perm[iu] == perm[ju]
        # incremental delta: ms_k - md_k
        ms_k = cos_pairs[same_k].mean() if same_k.any() else 0.0
        md_k = cos_pairs[~same_k].mean() if (~same_k).any() else 0.0
        null[k] = ms_k - md_k

    z = (delta_obs - null.mean()) / (null.std() + 1e-12)
    # (1+sum)/(1+n) two-sided perm-p
    pval = (1 + (np.abs(null) >= abs(delta_obs)).sum()) / (1 + n_perms)
    return {
        'delta_obs': float(delta_obs),
        'mean_same': float(mean_same),
        'mean_diff': float(mean_diff),
        'n_same_pairs': n_same, 'n_diff_pairs': n_diff,
        'null_mean': float(null.mean()),
        'null_std': float(null.std()),
        'z': float(z), 'p_perm': float(pval),
    }


def run_mc_test(emb_dict: dict, test_df: pd.DataFrame,
                mc_label_df: pd.DataFrame, mc_col: str, sub_col: str, spot_col: str,
                arm_name: str, n_subsample: int = 4000, n_perms: int = 10000,
                seed: int = 42):
    """run MC + patient z-test on one arm. subsample stratified by mc_megacluster."""
    # join mc label by niche
    # niche_id = subarray::spot_id; mc_labels uses (sub_col, spot_col) but the
    # subarray format in mc_labels lacks the TNBC{n}_ prefix
    test_df = test_df.copy()
    test_df['mc_sub_key'] = test_df['subarray'].str.replace(r'^TNBC\d+_', '', regex=True)
    mc_join = mc_label_df.rename(columns={sub_col: 'mc_sub_key',
                                          spot_col: 'spot_id',
                                          mc_col: 'mc_megacluster'})[
        ['mc_sub_key', 'spot_id', 'mc_megacluster']]
    df = test_df.merge(mc_join, on=['mc_sub_key', 'spot_id'], how='inner')
    df = df.dropna(subset=['mc_megacluster'])
    df = df[df['niche_id'].isin(emb_dict)].copy()

    if len(df) < 1000:
        print(f'  [{arm_name}] only {len(df)} niches after join + emb intersect; '
              'check key format')
        return []

    print(f'  [{arm_name}] {len(df):,} niches with mc + emb (of {len(test_df):,} test)')

    # stratified subsample by MC label
    labels = df['mc_megacluster'].values.astype(int)
    idx = stratified_subsample(labels, n_subsample, seed)
    df_sub = df.iloc[idx].reset_index(drop=True)
    print(f'  [{arm_name}] subsampled {len(df_sub)} niches, '
          f'{df_sub["mc_megacluster"].nunique()} MC classes, '
          f'{df_sub["patient_id"].nunique()} patients')

    # build embedding matrix
    X = np.stack([emb_dict[nid] for nid in df_sub['niche_id']])
    C = cosine_matrix(X)

    out = []
    for label_name, lbl in (('mc_megacluster', df_sub['mc_megacluster'].values.astype(int)),
                            ('patient_id', df_sub['patient_id'].values.astype(int))):
        print(f'    [{arm_name}/{label_name}] running {n_perms} perms...')
        res = coherence_zscore(C, lbl, n_perms=n_perms, seed=seed)
        res.update({'arm': arm_name, 'label': label_name,
                    'n_niches_subsample': len(df_sub),
                    'n_classes': int(np.unique(lbl).size)})
        out.append(res)
        print(f'      z = {res["z"]:+.2f}   delta = {res["delta_obs"]:+.4f}   '
              f'(null mean {res["null_mean"]:+.4f} +- {res["null_std"]:.4f})')
    return out


# ============================================================
# test 2: per-pathway cos vs AUCell consistency
# ============================================================

def _descendants(child_of: dict, root: str) -> set:
    seen, st = set(), [root]
    while st:
        n = st.pop()
        for c in child_of.get(n, ()):
            if c not in seen:
                seen.add(c); st.append(c)
    return seen


def load_reactome_descendants():
    from collections import defaultdict
    ch = defaultdict(set)
    with open(ROOT / 'knowledge' / 'reactome' / 'ReactomePathwaysRelation.txt') as f:
        for ln in f:
            p = ln.strip().split('\t')
            if len(p) == 2 and p[0].startswith('R-HSA-'):
                ch[p[0]].add(p[1])
    return ch


def pathway_rollup_vector(rhsa: dict, child_of: dict, parent_id: str) -> np.ndarray | None:
    """mean of {parent if embedded} U {embedded descendants}, L2 normed.
    return None if no embedded member."""
    sub = _descendants(child_of, parent_id)
    ids = ([parent_id] if parent_id in rhsa else []) + sorted(sub & set(rhsa))
    if not ids:
        return None
    V = np.stack([rhsa[i] for i in ids])
    v = V.mean(axis=0)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype(np.float32)


def run_aucell_test(emb_niche: dict, emb_rhsa: dict, child_of: dict,
                    aucell_df: pd.DataFrame, test_df: pd.DataFrame, arm_name: str):
    """per-pathway Spearman within subarray, mean +- SE across test subarrays."""
    test_df = test_df.copy()
    # AUCell parquet has per-niche columns named <pathway>_z and raw <pathway>
    score_col_map = {
        'TGF-beta_Signaling':    'TGF-beta_Signaling_z',
        'Immune_System':         'Immune_System_z',
        'ECM_Organization':      'ECM_Organization_z',
        'Cell_Cycle':            'Cell_Cycle_z',
        'Programmed_Cell_Death': 'Programmed_Cell_Death_z',
    }
    # fallback: try unscored column names
    cols = aucell_df.columns
    for k in list(score_col_map):
        if score_col_map[k] not in cols:
            alt = next((c for c in cols if k in c), None)
            if alt is None:
                print(f'  [{arm_name}] AUCell missing pathway {k}, skipping')
                score_col_map.pop(k)
                continue
            score_col_map[k] = alt
    print(f'  [{arm_name}] using AUCell cols: {score_col_map}')

    auc = aucell_df.reset_index().rename(columns={aucell_df.index.name or 'index': 'niche_id'})
    if 'niche_id' not in auc.columns:
        # parquet may have niche_id as a regular column
        if 'subarray' in auc.columns and 'spot_id' in auc.columns:
            auc['niche_id'] = auc['subarray'].astype(str) + '::' + auc['spot_id'].astype(str)
    df = test_df.merge(auc[['niche_id'] + list(score_col_map.values())],
                       on='niche_id', how='inner')
    df = df[df['niche_id'].isin(emb_niche)].copy()
    print(f'  [{arm_name}] {len(df):,} niches with AUCell + emb (of {len(test_df):,})')

    rows = []
    for pname, parent_id in NAMED_TARGETS.items():
        if pname not in score_col_map:
            continue
        v = pathway_rollup_vector(emb_rhsa, child_of, parent_id)
        if v is None:
            print(f'    [{arm_name}/{pname}] no embedded members, skipping')
            continue
        # cos(niche, pathway rollup) per niche
        E = np.stack([emb_niche[nid] for nid in df['niche_id']])
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        sig = (En @ v).astype(np.float32)
        df_p = df.assign(signal=sig)
        score_col = score_col_map[pname]
        # Spearman per subarray
        rhos = []
        for sub_name, sub_df in df_p.groupby('subarray'):
            if len(sub_df) < 30:
                continue
            r, _ = spearmanr(sub_df['signal'], sub_df[score_col],
                             nan_policy='omit')
            if np.isfinite(r):
                rhos.append((sub_name, float(r), int(len(sub_df))))
        if not rhos:
            print(f'    [{arm_name}/{pname}] no valid subarrays')
            continue
        rho_arr = np.array([r[1] for r in rhos])
        sub_arr = [r[0] for r in rhos]
        n_arr = np.array([r[2] for r in rhos])
        rows.append({
            'arm': arm_name, 'pathway_name': pname, 'parent_id': parent_id,
            'n_subarrays': len(rhos),
            'mean_rho': float(rho_arr.mean()),
            'median_rho': float(np.median(rho_arr)),
            'se_rho': float(rho_arr.std(ddof=1) / np.sqrt(len(rho_arr))),
            'n_subarrays_positive': int((rho_arr > 0).sum()),
            'n_subarrays_signif_pos': int(((rho_arr > 0.1) & (n_arr > 100)).sum()),
            'sample_subarrays_rho': dict(zip(sub_arr[:3], rho_arr[:3].tolist())),
        })
        print(f'    [{arm_name}/{pname}] mean rho = {rho_arr.mean():+.3f} '
              f'+- {rho_arr.std(ddof=1)/np.sqrt(len(rho_arr)):.3f} '
              f'(n={len(rho_arr)} subarrays, {(rho_arr>0).sum()} positive)')
    return rows


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', nargs=2, metavar=('NAME', 'PKL'), action='append',
                    help='one arm spec: NAME PATH_TO_PKL. repeat for >=2 arms. '
                         'if omitted, uses default v1_armA + v3_madmean.')
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    ap.add_argument('--n-subsample', type=int, default=4000,
                    help='stratified subsample for the MC coherence test '
                         '(pair count = n_subsample^2 / 2)')
    ap.add_argument('--n-perms', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    arms = ([(name, Path(pkl)) for name, pkl in args.arm]
            if args.arm else DEFAULT_ARMS)
    assert len(arms) >= 2, 'need at least 2 arms for a head-to-head'

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'out: {args.out_dir.relative_to(ROOT)}')
    print(f'n_subsample={args.n_subsample} n_perms={args.n_perms} seed={args.seed}')

    print('\n=== loading embeddings + labels ===')
    loaded = {}
    for name, pkl in arms:
        rhsa, niche = load_emb(pkl)
        loaded[name] = (rhsa, niche)
        print(f'  {name}: {len(rhsa):,} R-HSA, {len(niche):,} niches '
              f'<- {pkl.relative_to(ROOT) if pkl.is_relative_to(ROOT) else pkl}')

    test_df = load_test_niches()
    print(f'  R4 test niches: {len(test_df):,}')
    mc_lbl, mc_col, sub_col, spot_col = load_mc_labels()
    print(f'  mc_labels: {len(mc_lbl):,} rows')
    auc_df = pd.read_parquet(AUCELL_5)
    print(f'  AUCell 5-pathway: {auc_df.shape}')
    ch_of = load_reactome_descendants()

    print('\n=== test 1: MC coherence (matched-null z-test) ===')
    mc_rows = []
    for name, (rhsa, niche) in loaded.items():
        mc_rows += run_mc_test(niche, test_df, mc_lbl, mc_col, sub_col, spot_col,
                               arm_name=name,
                               n_subsample=args.n_subsample, n_perms=args.n_perms,
                               seed=args.seed)
    mc_df = pd.DataFrame(mc_rows)
    out_mc = args.out_dir / 'noise_floor_mc.parquet'
    mc_df.to_parquet(out_mc, index=False)
    print(f'\nwrote {out_mc.relative_to(ROOT)}')
    print(mc_df[['arm', 'label', 'n_classes', 'delta_obs', 'null_mean',
                 'null_std', 'z', 'p_perm']].to_string(index=False))

    print('\n=== test 2: per-pathway cos vs AUCell (Spearman per subarray) ===')
    auc_rows = []
    for name, (rhsa, niche) in loaded.items():
        auc_rows += run_aucell_test(niche, rhsa, ch_of, auc_df, test_df, name)
    auc_out_df = pd.DataFrame(auc_rows)
    auc_clean = auc_out_df.drop(columns=['sample_subarrays_rho'], errors='ignore')
    out_auc = args.out_dir / 'noise_floor_aucell.parquet'
    auc_clean.to_parquet(out_auc, index=False)
    print(f'\nwrote {out_auc.relative_to(ROOT)}')
    print(auc_clean.to_string(index=False))


if __name__ == '__main__':
    main()