"""H3 multivariate per-pathway CCA using gpath2vec pathway embeddings.

proposal H3 (verbatim): "Reactome pathway embeddings computed via gpath2vec for five
cancer-relevant pathways (TGF-beta Signaling, Immune System, Extracellular Matrix
Organization, Cell Cycle, Programmed Cell Death) will correlate with specific directions
in the shared latent space via canonical correlation analysis, indicating preservation of
interpretable biological signal."

this is the proposal-faithful H3 ("shape B" multivariate). the AUCell variant
(eval_h3_pathway_cca.py) is the orthogonal robustness check.

per pathway:
  pathway set = {parent} U {embedded descendants}    (one 512-d gpath2vec vector each)
  for each niche, view-2 row = (niche_gpath2vec @ member_j) for j in set
                              = niche's similarity profile across the pathway set
  Z = aligned latent (z_he | z_st | z_mean), 512-d per niche
  Y = niche x set_size similarity matrix

  multivariate CCA(Z, Y) via QR + SVD:
    Z = Q_Z R_Z,  Y = Q_Y R_Y,  M = Q_Z^T Q_Y
    SVD: M = U S V^T  ->  top canonical correlation = S[0]
    canonical directions: a = R_Z^-1 U[:,0],  b = R_Y^-1 V[:,0]

  option B (fit-on-all-test-niches): S[0] on full test set.
  option A (11/3 patient sub-split): fit a,b on train; project test;
            test_corr = pearsonr(Z_te @ a, Y_te @ b).

  perm null: shuffle Y rows (option B) or Y_train rows (option A), refit, recompute.

circularity flag: scripts/align.py default st_features=('novae_niche','gpath2vec_niche'),
so R1/R2/R3/R4 consume gpath2vec on the ST side. z_he is clean across all 8 runs;
z_st and z_mean are partially circular for those four runs. flag in view_clean column.

inputs:
  --embeddings-pkl  pickle dict {node_id: 512-d ndarray}, niches + R-HSA-* pathways
                    defaults to v1_legacy for dry run; point at arm A output when ready
  --out-dir         default runs/tnbc-92/eval/H3/pathway_cca_gpath2vec
  --n-perms         default 500

outputs (parallel to AUCell H3 layout):
  pathway_axes.parquet         5 rows: name, id, parent_embedded, n_children_embedded, set_size
  per_pathway_cca.parquet      120 rows = 8 runs x 3 views x 5 pathways
                               cols: top_corr_full (B), top_corr_train, top_corr_test (A),
                                     null stats, p-values, z, BH-FDR, view_clean
  perm_nulls.parquet           long-form perm draws for diagnostics
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / 'data' / 'embeddings' / 'biological_signals'
REACTOME = ROOT / 'knowledge' / 'reactome'
RUNS = ROOT / 'runs' / 'tnbc-92'

DEFAULT_EMBEDDINGS = BIO / 'gpath2vec_output' / 'full_cohort_tf_low' / 'embeddings_v1_legacy.pkl'
DEFAULT_OUT = RUNS / 'eval' / 'H3' / 'pathway_cca_gpath2vec'

NAMED_TARGETS = {
    'TGF-beta_Signaling':    'R-HSA-170834',
    'Immune_System':         'R-HSA-168256',
    'ECM_Organization':      'R-HSA-1474244',
    'Cell_Cycle':            'R-HSA-1640170',
    'Programmed_Cell_Death': 'R-HSA-5357801',
}

RUN_IDS = ['R1', 'R2', 'R3', 'R4', 'R6', 'B1', 'B2', 'B3']
VIEWS = ['z_he', 'z_st', 'z_mean']
GPATH2VEC_ST_INPUT_RUNS = {'R1', 'R2', 'R3', 'R4'}
SEED = 42
N_PERMS = 500

# F1 reproducibility lock (see projects/tnbc-92/proposal_deviations.md, "H3 gpath2vec arm").
# the comparison to AUCell per_node_cca is only valid if both methods score the IDENTICAL
# pathway node set. AUCell used dag_metadata.parquet INTERSECT per_node_cca.node_id (the
# frozen 395-node set). this harness derives member sets from raw ReactomePathwaysRelation.txt;
# for arm A's 680 embedded pathways those two sources were verified IDENTICAL per parent on
# 2026-05-17 (TGF-beta R-HSA-170834 = 2, raw vs frozen 0 divergent nodes for all 5). these
# expected set sizes assert that equality holds at run time and fail loudly if the embedding
# set or Reactome snapshot ever drifts (the 111-vs-114 divergence class).
FROZEN_SET_SIZES = {
    'TGF-beta_Signaling': 2,   # parent R-HSA-170834 embedded + 1 embedded descendant (n=2, underpowered)
    'Immune_System': 74,
    'ECM_Organization': 14,
    'Cell_Cycle': 14,
    'Programmed_Cell_Death': 7,
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embeddings-pkl', type=Path, default=DEFAULT_EMBEDDINGS)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--n-perms', type=int, default=N_PERMS)
    return ap.parse_args()


# ---------- phase 1: build {parent + embedded children} per pathway -----------

def load_descendants():
    children_of = defaultdict(set)
    with open(REACTOME / 'ReactomePathwaysRelation.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2 and parts[0].startswith('R-HSA-'):
                children_of[parts[0]].add(parts[1])

    def descendants(root):
        seen, stack = set(), [root]
        while stack:
            n = stack.pop()
            for c in children_of.get(n, ()):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen
    return descendants


def build_pathway_sets(node_emb, descendants_of):
    """per parent: set of 512-d gpath2vec vectors = {parent if embedded} U {embedded descendants}.
    returns (axes_df, members_per_pathway dict: name -> (set_size, m x 512 matrix))."""
    embedded = {k for k in node_emb if isinstance(k, str) and k.startswith('R-HSA-')}
    rows = []
    members = {}
    for name, pid in NAMED_TARGETS.items():
        ids = []
        if pid in embedded:
            ids.append(pid)
        ids.extend(sorted(descendants_of(pid) & embedded))
        if not ids:
            raise RuntimeError(f'{pid} ({name}): no embedded parent or descendants')
        P = np.stack([node_emb[i].astype(np.float64) for i in ids])
        # L2-normalize so niche @ P^T is cosine similarity
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        members[name] = (ids, P)
        rows.append({
            'pathway_name': name,
            'pathway_id': pid,
            'parent_embedded': pid in embedded,
            'n_children_embedded': len(ids) - (1 if pid in embedded else 0),
            'set_size': len(ids),
        })
        print(f'  {name:24s} set_size={len(ids):4d}  parent_embedded={pid in embedded}')
    return pd.DataFrame(rows), members


# ---------- phase 2: build niche-side cosine tables per pathway ---------------

def niche_key_to_id(k):
    return k.rsplit('__', 1)[-1]


def build_niche_cosine_tables(node_emb, members):
    """returns dict niche_id -> per-pathway cosine vector, as a pandas df indexed by niche_id."""
    niche_keys = [k for k in node_emb if isinstance(k, str) and k.startswith('cluster_')]
    print(f'  niches in gpath2vec embedding: {len(niche_keys):,}')
    N = np.stack([node_emb[k].astype(np.float64) for k in niche_keys])
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    niche_ids = [niche_key_to_id(k) for k in niche_keys]

    # one cosine matrix per pathway, written separately because variable width
    cosine_per_pathway = {}
    for name, (ids, P) in members.items():
        C = N @ P.T                              # (n_niches, set_size)
        df = pd.DataFrame(C, index=niche_ids, columns=ids)
        df = df[~df.index.duplicated(keep='first')]
        df.index.name = 'niche_id'
        cosine_per_pathway[name] = df
    return cosine_per_pathway


# ---------- phase 3: multivariate CCA per (run, view, pathway) ----------------

def _as_arr(c):
    return np.asarray(c, dtype=np.float32)


def stack_col(s):
    return np.stack([_as_arr(v) for v in s])


def patient_sub_split(patients, n_test_pats=3):
    unique = np.unique(patients)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique)
    test_pats = set(unique[:n_test_pats].tolist())
    train_pats = set(unique[n_test_pats:].tolist())
    return np.isin(patients, list(train_pats)), np.isin(patients, list(test_pats))


def get_view(df, view):
    if view == 'z_mean':
        return 0.5 * (stack_col(df['z_he']) + stack_col(df['z_st']))
    return stack_col(df[view])


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=np.float64)
    valid = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if valid.sum() == 0:
        return out
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    adj = pv[order] * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        if adj[i + 1] < adj[i]:
            adj[i] = adj[i + 1]
    adj = np.minimum(adj, 1.0)
    out_valid = np.empty(n)
    out_valid[order] = adj
    out[valid] = out_valid
    return out


def top_canonical_corr(Z, Y):
    """multivariate CCA via QR + SVD. returns (top_corr, a_dir, b_dir).
    Z (n, d_z), Y (n, m).  a is (d_z,), b is (m,)."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Q_Z, R_Z = np.linalg.qr(Zc, mode='reduced')
    Q_Y, R_Y = np.linalg.qr(Yc, mode='reduced')
    M = Q_Z.T @ Q_Y
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
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


def top_canonical_corr_value_only(Z, Y):
    """faster path: only the top singular value (used inside perm loops)."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Q_Z, _ = np.linalg.qr(Zc, mode='reduced')
    Q_Y, _ = np.linalg.qr(Yc, mode='reduced')
    M = Q_Z.T @ Q_Y
    s_top = float(np.linalg.svd(M, full_matrices=False, compute_uv=False)[0])
    return float(np.clip(s_top, -1.0, 1.0))


# --- Q_Z-cached perm variants -------------------------------------------------
# permutations shuffle Y rows only; Z (hence Q_Z, R_Z) and the column means are
# invariant within a cell. caching them across perms is bit-identical to the
# as-written top_canonical_corr / _value_only (a = R_Z^-1 U, R_Z from fixed Z;
# Y[idx] has the same column means as Y, so Yc[idx] == (Y[idx] - Y.mean)).
# this mirrors the caching discipline already in eval_h3_pathway_cca_perms.py.

def top_canonical_corr_value_only_cached(Q_Z, Yc_perm):
    Q_Y, _ = np.linalg.qr(Yc_perm, mode='reduced')
    s_top = float(np.linalg.svd(Q_Z.T @ Q_Y, full_matrices=False, compute_uv=False)[0])
    return float(np.clip(s_top, -1.0, 1.0))


def canonical_dirs_cached(Q_Z, R_Z, Yc_perm):
    """a, b from precomputed Q_Z, R_Z (Z fixed) and a permuted, pre-centered Y."""
    Q_Y, R_Y = np.linalg.qr(Yc_perm, mode='reduced')
    U, _, Vt = np.linalg.svd(Q_Z.T @ Q_Y, full_matrices=False)
    try:
        a = np.linalg.solve(R_Z, U[:, 0])
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(R_Z, U[:, 0], rcond=None)[0]
    try:
        b = np.linalg.solve(R_Y, Vt[0])
    except np.linalg.LinAlgError:
        b = np.linalg.lstsq(R_Y, Vt[0], rcond=None)[0]
    return a, b


def run_cca_grid(cosine_per_pathway, n_perms):
    obs_rows = []
    perm_rows = []

    for run_id in RUN_IDS:
        print(f'\n  {run_id}: load embeddings + join cosine tables...')
        emb = pd.read_parquet(RUNS / run_id / 'embeddings_test.parquet').reset_index()
        emb['niche_id'] = emb['subarray'].astype(str) + '::' + emb['spot_id'].astype(str)

        for view in VIEWS:
            Z_full = get_view(emb, view).astype(np.float64)
            print(f'    {view}: Z shape {Z_full.shape}')

            for pname, pid in NAMED_TARGETS.items():
                cos_df = cosine_per_pathway[pname]
                # join: keep only niches present in both
                joined = emb[['niche_id', 'patient_id']].merge(
                    cos_df, left_on='niche_id', right_index=True, how='inner')
                if len(joined) == 0:
                    print(f'      {pname}: no overlap, skipping')
                    continue
                # align Z to joined niche order
                idx_in_emb = emb.set_index('niche_id').index
                joined_pos = idx_in_emb.get_indexer(joined['niche_id'])
                Z = Z_full[joined_pos]
                Y = joined[cos_df.columns].to_numpy(dtype=np.float64)
                patients = joined['patient_id'].astype(int).to_numpy()
                train_mask, test_mask = patient_sub_split(patients, n_test_pats=3)
                n_full = len(joined)
                n_tr = int(train_mask.sum())
                n_te = int(test_mask.sum())
                set_size = Y.shape[1]

                # option B observed: fit on all
                obs_B = top_canonical_corr_value_only(Z, Y)

                # option A observed: fit on train, eval on test
                Z_tr, Y_tr = Z[train_mask], Y[train_mask]
                Z_te, Y_te = Z[test_mask], Y[test_mask]
                _, a_tr, b_tr = top_canonical_corr(Z_tr, Y_tr)
                Z_te_c = Z_te - Z_te.mean(axis=0, keepdims=True)
                Y_te_c = Y_te - Y_te.mean(axis=0, keepdims=True)
                u_te = Z_te_c @ a_tr
                v_te = Y_te_c @ b_tr
                if u_te.std() > 0 and v_te.std() > 0:
                    obs_A = float(pearsonr(u_te, v_te)[0])
                else:
                    obs_A = float('nan')

                # train-side canonical correlation (for reporting / sanity)
                # by construction this equals the top singular value of M_train
                Z_tr_c = Z_tr - Z_tr.mean(axis=0, keepdims=True)
                Y_tr_c = Y_tr - Y_tr.mean(axis=0, keepdims=True)
                u_tr = Z_tr_c @ a_tr
                v_tr = Y_tr_c @ b_tr
                obs_train = float(pearsonr(u_tr, v_tr)[0]) if u_tr.std() > 0 and v_tr.std() > 0 else float('nan')

                # Q_Z-cached precompute (Z fixed across perms; bit-identical to as-written).
                # Z_tr_c / Y_tr_c already centered above; Y[idx] has Y's column means
                # so the centered permuted Y is just a row-permutation of the centered Y.
                Zc_full = Z - Z.mean(axis=0, keepdims=True)
                Qz_B, _ = np.linalg.qr(Zc_full, mode='reduced')
                Yc_full = Y - Y.mean(axis=0, keepdims=True)
                Qz_tr, Rz_tr = np.linalg.qr(Z_tr_c, mode='reduced')

                # perm null option B: shuffle Y rows, recompute top canonical
                rng_B = np.random.default_rng(SEED)
                nulls_B = np.empty(n_perms)
                for i in range(n_perms):
                    idx = rng_B.permutation(n_full)
                    nulls_B[i] = top_canonical_corr_value_only_cached(Qz_B, Yc_full[idx])

                # perm null option A: shuffle Y_train, refit, eval test
                rng_A = np.random.default_rng(SEED + 1)
                nulls_A = np.empty(n_perms)
                for i in range(n_perms):
                    idx_tr = rng_A.permutation(n_tr)
                    a_s, b_s = canonical_dirs_cached(Qz_tr, Rz_tr, Y_tr_c[idx_tr])
                    u_s = Z_te_c @ a_s
                    v_s = Y_te_c @ b_s
                    if u_s.std() > 0 and v_s.std() > 0:
                        nulls_A[i] = float(pearsonr(u_s, v_s)[0])
                    else:
                        nulls_A[i] = float('nan')

                view_clean = view == 'z_he' or run_id not in GPATH2VEC_ST_INPUT_RUNS
                null_A_valid = nulls_A[~np.isnan(nulls_A)]

                obs_rows.append({
                    'run_id': run_id, 'view': view,
                    'pathway_name': pname, 'pathway_id': pid,
                    'set_size': set_size, 'view_clean': view_clean,
                    'top_corr_full': obs_B,
                    'top_corr_train': obs_train,
                    'top_corr_test': obs_A,
                    'null_B_mean': float(nulls_B.mean()), 'null_B_std': float(nulls_B.std()),
                    'null_A_mean': float(null_A_valid.mean()) if null_A_valid.size else float('nan'),
                    'null_A_std':  float(null_A_valid.std())  if null_A_valid.size else float('nan'),
                    'p_one_B': float((nulls_B >= obs_B).mean()),
                    'p_two_A': (float((np.abs(null_A_valid) >= abs(obs_A)).mean())
                                if null_A_valid.size and not np.isnan(obs_A) else float('nan')),
                    'z_B': (obs_B - float(nulls_B.mean())) / (float(nulls_B.std()) + 1e-12),
                    'z_A': ((obs_A - float(null_A_valid.mean())) / (float(null_A_valid.std()) + 1e-12)
                            if null_A_valid.size and not np.isnan(obs_A) else float('nan')),
                    'n_full': n_full, 'n_train': n_tr, 'n_test': n_te,
                })
                perm_rows.append({
                    'run_id': run_id, 'view': view, 'pathway_name': pname,
                    'null_A': nulls_A.tolist(), 'null_B': nulls_B.tolist(),
                })
                print(f'      {pname:24s} set={set_size:3d}  obs_B={obs_B:.3f}  obs_A={obs_A:.3f}  '
                      f'z_A={obs_rows[-1]["z_A"]:.2f}  clean={view_clean}')

    df = pd.DataFrame(obs_rows)
    df['fdr_B'] = bh_fdr(df['p_one_B'].values)
    df['fdr_A'] = bh_fdr(df['p_two_A'].values)
    df['sig_B_05'] = df['fdr_B'] < 0.05
    df['sig_A_05'] = df['fdr_A'] < 0.05
    return df, pd.DataFrame(perm_rows)


# ---------- main --------------------------------------------------------------

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'embeddings: {args.embeddings_pkl}')
    print(f'out_dir:    {args.out_dir}')
    print(f'n_perms:    {args.n_perms}')

    print('\n=== phase 1: load embeddings + Reactome descendants ===')
    with open(args.embeddings_pkl, 'rb') as f:
        node_emb = pickle.load(f)
    n_path = sum(1 for k in node_emb if isinstance(k, str) and k.startswith('R-HSA-'))
    n_nic  = sum(1 for k in node_emb if isinstance(k, str) and k.startswith('cluster_'))
    print(f'  embedded pathways: {n_path:,}   embedded niches: {n_nic:,}')
    desc = load_descendants()

    print('\n=== phase 2: build {parent+children} sets per pathway ===')
    axes_df, members = build_pathway_sets(node_emb, desc)

    # F1 lock: assert raw-relation member sets == frozen AUCell-comparable 395-node set sizes
    actual = {n: len(ids) for n, (ids, _) in members.items()}
    if actual != FROZEN_SET_SIZES:
        raise RuntimeError(
            'F1 LOCK FAILED — gpath2vec member sets diverge from the frozen AUCell-comparable '
            f'node set.\n  expected (proposal_deviations.md): {FROZEN_SET_SIZES}\n  got: {actual}\n'
            'the head-to-head vs AUCell per_node_cca is INVALID until reconciled. do not '
            'interpret results; re-pin the hierarchy in proposal_deviations.md first.')
    print(f'  F1 lock OK: member set sizes == frozen {FROZEN_SET_SIZES}')

    axes_df.to_parquet(args.out_dir / 'pathway_axes.parquet', index=False)
    print(f'wrote {args.out_dir / "pathway_axes.parquet"}')

    print('\n=== phase 3: niche-vs-set cosine tables ===')
    cos = build_niche_cosine_tables(node_emb, members)

    print('\n=== phase 4: multivariate CCA grid (option A + option B, perm null, BH-FDR) ===')
    df, perm_df = run_cca_grid(cos, args.n_perms)
    df.to_parquet(args.out_dir / 'per_pathway_cca.parquet', index=False)
    perm_df.to_parquet(args.out_dir / 'perm_nulls.parquet', index=False)
    print(f'\nwrote {args.out_dir / "per_pathway_cca.parquet"}  ({len(df)} rows)')
    print(f'wrote {args.out_dir / "perm_nulls.parquet"}')

    print('\n=== summary: option A (cross-patient) significant cells, z_he view ===')
    z_he = df[df['view'] == 'z_he']
    s_A = z_he.groupby('run_id')['sig_A_05'].sum().reindex(RUN_IDS)
    for r, n in s_A.items():
        flag = ' <- gpath2vec ST input' if r in GPATH2VEC_ST_INPUT_RUNS else ''
        print(f'  {r}: {int(n)}/5 pathways sig{flag}')

    print('\nsummary: option B (fit-on-all) significant cells, z_he view:')
    s_B = z_he.groupby('run_id')['sig_B_05'].sum().reindex(RUN_IDS)
    for r, n in s_B.items():
        print(f'  {r}: {int(n)}/5 pathways sig')


if __name__ == '__main__':
    main()