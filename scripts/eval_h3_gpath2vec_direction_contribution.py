"""per-niche / MC / patient contribution to gpath2vec H3 canonical directions.

answers "is the signal driving v3's CCA z-score carried by a few dominant
niches, or distributed across the cohort?" - the test the reviewer asks once
MC z is high but the cohort is heterogeneous. concentration metrics
(Gini, top-k mass) per (run, view, pathway):

  Gini  ~  0  ->  signal distributed uniformly across test niches
  Gini  ~  1  ->  signal concentrated in a few dominant niches
  top10pct_mass  ->  what fraction of the total |projection| comes from
                     the top 10% of niches by |projection|

per-MC and per-patient mean signed projection identifies which biological
classes / patients drive the direction.

reuses the CCA core (QR + SVD, top canonical direction `a`) from
eval_h3_pathway_cca_gpath2vec_v2.py. refits CCA on the full test set per cell
(option-B style fit-on-all; saturates magnitude, but the DIRECTION shape is
what we need for the concentration analysis).

usage:
    python scripts/eval_h3_gpath2vec_direction_contribution.py \\
        --embeddings-pkl data/embeddings/gpath2vec/.../embeddings.pkl \\
        --cca-dir runs/tnbc-92/eval/H3/pathway_cca_gpath2vec_v3_fisher_madmean_low

writes:
    {cca-dir}/direction_contribution.parquet
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / 'data' / 'embeddings' / 'biological_signals'
RUNS = ROOT / 'runs' / 'tnbc-92'

NAMED_TARGETS = {
    'TGF-beta_Signaling':    'R-HSA-170834',
    'Immune_System':         'R-HSA-168256',
    'ECM_Organization':      'R-HSA-1474244',
    'Cell_Cycle':            'R-HSA-1640170',
    'Programmed_Cell_Death': 'R-HSA-5357801',
}
RUN_IDS = ['R1', 'R2', 'R3', 'R4', 'R6', 'B1', 'B2', 'B3']
VIEWS = ['z_he', 'z_st', 'z_mean']


# ---------- helpers ---------------------------------------------------------

def niche_key_to_id(k: str) -> str:
    s = k.rsplit('__', 1)[-1]
    return s[len('cluster_'):] if s.startswith('cluster_') else s


def gini(x):
    """Gini coefficient of non-negative array. 0 = uniform, 1 = concentrated.
    standard formula via cumulative shares."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or x.sum() <= 0:
        return float('nan')
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def topk_mass(x, frac=0.10):
    """fraction of total |x| coming from the top frac of |x|."""
    x = np.sort(np.abs(np.asarray(x, np.float64)))[::-1]
    if x.sum() <= 0:
        return float('nan')
    k = max(1, int(np.ceil(frac * x.size)))
    return float(x[:k].sum() / x.sum())


def top_canonical_direction(Z, Y):
    """returns (top_corr, a, b) where a maps Z -> 1d canonical score, b maps Y."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Q_Z, R_Z = np.linalg.qr(Zc, mode='reduced')
    Q_Y, R_Y = np.linalg.qr(Yc, mode='reduced')
    U, S, Vt = np.linalg.svd(Q_Z.T @ Q_Y, full_matrices=False)
    top = float(np.clip(S[0], -1.0, 1.0))
    try:
        a = np.linalg.solve(R_Z, U[:, 0])
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(R_Z, U[:, 0], rcond=None)[0]
    try:
        b = np.linalg.solve(R_Y, Vt[0])
    except np.linalg.LinAlgError:
        b = np.linalg.lstsq(R_Y, Vt[0], rcond=None)[0]
    return top, a, b


# ---------- loaders ---------------------------------------------------------

def load_run_embeddings(run_id):
    p = RUNS / run_id / 'embeddings_test.parquet'
    df = pd.read_parquet(p)
    z_he = np.stack([np.asarray(v, np.float64) for v in df['z_he']])
    z_st = np.stack([np.asarray(v, np.float64) for v in df['z_st']])
    z_mean = (z_he + z_st) / 2.0
    z_mean = z_mean / (np.linalg.norm(z_mean, axis=1, keepdims=True) + 1e-12)
    niche_ids = (df['subarray'].astype(str) + '::' + df.index.astype(str)).values
    return df, niche_ids, {'z_he': z_he, 'z_st': z_st, 'z_mean': z_mean}


def load_mc_labels():
    mc = pd.read_csv(BIO / 'mc_labels.tsv', sep='\t')
    cand = [c for c in mc.columns if 'megacluster' in c.lower()]
    mc_col = cand[0]
    sub_col = next(c for c in mc.columns if c.lower() in ('subarray', 'slide_pos'))
    spot_col = next(c for c in mc.columns if c.lower() in ('spot_id', 'spot'))
    return mc, mc_col, sub_col, spot_col


def build_pathway_member_vectors(node_emb, pid, restrict_to=None):
    """{parent if embedded} U {embedded descendants}. returns ids, L2-normed matrix."""
    from collections import defaultdict
    ch = defaultdict(set)
    with open(ROOT / 'knowledge' / 'reactome' / 'ReactomePathwaysRelation.txt') as f:
        for ln in f:
            p = ln.strip().split('\t')
            if len(p) == 2 and p[0].startswith('R-HSA-'):
                ch[p[0]].add(p[1])

    def _desc(root):
        seen, st = set(), [root]
        while st:
            n = st.pop()
            for c in ch.get(n, ()):
                if c not in seen:
                    seen.add(c); st.append(c)
        return seen

    embedded = {k for k in node_emb if isinstance(k, str) and k.startswith('R-HSA-')}
    sub = _desc(pid)
    ids = []
    if pid in embedded and (restrict_to is None or pid in restrict_to):
        ids.append(pid)
    pool = (sub & embedded) if restrict_to is None else (sub & embedded & restrict_to)
    ids.extend(sorted(pool))
    if not ids:
        return [], np.zeros((0, 0))
    P = np.stack([np.asarray(node_emb[i], np.float64) for i in ids])
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    return ids, P


# ---------- main analysis ---------------------------------------------------

def analyze_cell(run_id, view, pathway_name, pid, node_emb, mc_lookup, restrict_to=None):
    """fit CCA on full test set; project niches onto direction; partition."""
    df_run, niche_ids, views = load_run_embeddings(run_id)
    Z = views[view]
    n = len(niche_ids)

    ids, P = build_pathway_member_vectors(node_emb, pid, restrict_to=restrict_to)
    if not ids:
        return None
    # Y[n, m] = cos(niche_emb_in_gpath2vec_space, member_j) for j in ids
    niche_keys_g = {niche_key_to_id(k): k for k in node_emb
                    if isinstance(k, str) and k.startswith('cluster_')}
    valid_mask = np.array([nid in niche_keys_g for nid in niche_ids])
    if valid_mask.sum() < 100:
        return None
    Z_v = Z[valid_mask]
    niche_ids_v = niche_ids[valid_mask]
    E = np.stack([np.asarray(node_emb[niche_keys_g[nid]], np.float64)
                  for nid in niche_ids_v])
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    Y = En @ P.T

    if Y.shape[1] < 2:
        return None  # single-member CCA degenerate
    top_corr, a, _ = top_canonical_direction(Z_v, Y)
    proj = Z_v @ a  # 1d per-niche canonical score
    abs_proj = np.abs(proj)

    # join niche metadata (mc, patient, subarray) for partitioning
    df_v = df_run[valid_mask.tolist() if hasattr(df_run, 'iloc') else valid_mask]
    df_v = df_v.reset_index().rename(columns={'spot_id': 'spot_id'}) \
        if df_v.index.name == 'spot_id' else df_v
    df_v['proj'] = proj
    df_v['abs_proj'] = abs_proj
    df_v['niche_id'] = niche_ids_v

    # mc_megacluster via mc_lookup (subarray_stripped, spot_id) -> mc
    df_v['mc_sub_key'] = df_v['subarray'].str.replace(r'^TNBC\d+_', '', regex=True)
    df_v = df_v.merge(mc_lookup, on=['mc_sub_key', 'spot_id'], how='left')

    # ---- concentration metrics ----
    g = gini(abs_proj)
    top10 = topk_mass(abs_proj, 0.10)
    top1 = topk_mass(abs_proj, 0.01)

    # ---- per-MC partition: which MC carries the (signed) direction ----
    by_mc = (df_v.dropna(subset=['mc_megacluster'])
             .groupby('mc_megacluster')['proj']
             .agg(['mean', 'std', 'count']).reset_index())
    by_mc = by_mc.sort_values('mean', key=lambda s: s.abs(), ascending=False)
    top_mc = by_mc.iloc[0]['mc_megacluster'] if len(by_mc) else None
    top_mc_mean = by_mc.iloc[0]['mean'] if len(by_mc) else float('nan')
    top_mc_n = int(by_mc.iloc[0]['count']) if len(by_mc) else 0

    # ---- per-patient partition: any single patient drive? ----
    by_pat = (df_v.groupby('patient_id')['proj']
              .agg(['mean', 'std', 'count']).reset_index())
    by_pat = by_pat.sort_values('mean', key=lambda s: s.abs(), ascending=False)
    top_pat = int(by_pat.iloc[0]['patient_id']) if len(by_pat) else None
    top_pat_mean = by_pat.iloc[0]['mean'] if len(by_pat) else float('nan')
    top_pat_n = int(by_pat.iloc[0]['count']) if len(by_pat) else 0

    return {
        'run_id': run_id, 'view': view,
        'pathway_name': pathway_name, 'pathway_id': pid,
        'n_niches': int(valid_mask.sum()),
        'set_size': len(ids),
        'top_canonical_corr_obs': top_corr,
        'gini_abs_proj': g,
        'top1pct_mass_frac': top1,
        'top10pct_mass_frac': top10,
        'top_mc': str(top_mc), 'top_mc_signed_mean': float(top_mc_mean),
        'top_mc_n_niches': top_mc_n,
        'top_pat_id': top_pat, 'top_pat_signed_mean': float(top_pat_mean),
        'top_pat_n_niches': top_pat_n,
        'n_mcs_observed': int(df_v['mc_megacluster'].dropna().nunique()),
        'n_patients_observed': int(df_v['patient_id'].nunique()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embeddings-pkl', type=Path, required=True)
    ap.add_argument('--cca-dir', type=Path, required=True,
                    help='dir containing per_pathway_cca.parquet for this arm; '
                         'output written to {cca-dir}/direction_contribution.parquet')
    ap.add_argument('--restrict-to-arm-a-frozen', action='store_true',
                    help='use the same arm-A frozen 395 hierarchy as the matched CCA')
    args = ap.parse_args()

    print(f'embeddings: {args.embeddings_pkl}')
    print(f'cca-dir   : {args.cca_dir}')

    print('\n=== loading ===')
    with open(args.embeddings_pkl, 'rb') as f:
        node_emb = pickle.load(f)
    print(f'  loaded {len(node_emb):,} keys from embeddings pkl')

    restrict_to = None
    if args.restrict_to_arm_a_frozen:
        _DAG = RUNS / 'eval' / 'H3' / 'pathway_cca' / 'dag_full'
        _dm = pd.read_parquet(_DAG / 'dag_metadata.parquet', columns=['node_id'])
        _pnc = pd.read_parquet(_DAG / 'per_node_cca.parquet', columns=['node_id'])
        restrict_to = set(_dm['node_id']) & set(_pnc['node_id'])
        print(f'  rank-matched mode: restricting to arm-A frozen '
              f'{len(restrict_to)} nodes')

    mc_lbl, mc_col, sub_col, spot_col = load_mc_labels()
    mc_lookup = mc_lbl.rename(columns={sub_col: 'mc_sub_key',
                                       spot_col: 'spot_id',
                                       mc_col: 'mc_megacluster'})[
        ['mc_sub_key', 'spot_id', 'mc_megacluster']]
    print(f'  mc labels: {len(mc_lookup):,} rows')

    print('\n=== sweeping (run, view, pathway) ===')
    rows = []
    for run_id in RUN_IDS:
        for view in VIEWS:
            for pname, pid in NAMED_TARGETS.items():
                res = analyze_cell(run_id, view, pname, pid, node_emb, mc_lookup,
                                   restrict_to=restrict_to)
                if res is None:
                    continue
                rows.append(res)
                print(f'  {run_id} {view:6s} {pname:24s} '
                      f'n={res["n_niches"]:5d} set={res["set_size"]:3d} '
                      f'r={res["top_canonical_corr_obs"]:.3f} '
                      f'gini={res["gini_abs_proj"]:.3f} '
                      f'top10%={res["top10pct_mass_frac"]:.3f} '
                      f'topMC={res["top_mc"]}({res["top_mc_signed_mean"]:+.3f})')

    df = pd.DataFrame(rows)
    out = args.cca_dir / 'direction_contribution.parquet'
    df.to_parquet(out, index=False)
    print(f'\nwrote {out.relative_to(ROOT)} ({len(df)} rows)')
    print('\n=== R4 z_he summary (the headline cell) ===')
    print(df[(df['run_id'] == 'R4') & (df['view'] == 'z_he')]
          [['pathway_name', 'n_niches', 'set_size', 'top_canonical_corr_obs',
            'gini_abs_proj', 'top1pct_mass_frac', 'top10pct_mass_frac',
            'top_mc', 'top_mc_signed_mean', 'top_pat_id', 'top_pat_signed_mean']]
          .to_string(index=False))


if __name__ == '__main__':
    main()