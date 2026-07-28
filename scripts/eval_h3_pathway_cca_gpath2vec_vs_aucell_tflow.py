"""H3 controlled head-to-head: gpath2vec vs AUCell, matched regime.

WHY THIS EXISTS
The shipped AUCell H3 input (niche_aucell_5targets.parquet) is NOT a valid
comparator for gpath2vec arm A. Verified in niche_aucell.py: those 5 columns are
AUCell scored against gene-set UNIONS built by BFS over the FULL Reactome gmt
(all levels, high-level included), on full normalized expression, no TF filter.
gpath2vec arm A is Fisher EA on the top-100 TF-filtered genes vs LOW-level
pathways, embedded, rolled to 5 parents. Four stacked confounds (universe,
gene input, gene filter, parent construction) make any "gpath2vec vs AUCell"
read from that pairing uninterpretable.

WHAT THIS DOES
Removes the two largest confounds (all-level -> low; gene-union -> per-pathway)
and isolates the representation step. Both methods are scored on:
  - the IDENTICAL member set: v2's {parent if embedded} U {embedded low
    descendants} (frozen sizes asserted) INTERSECTED with the pathways AUCell
    actually scored in niche_aucell_low_level.parquet (829 low-level). The
    shared set is used for BOTH methods, so neither is evaluated on members the
    other lacks.
  - the IDENTICAL niches (inner join on subarray::spot_id),
  - the IDENTICAL aligned latent Z (z_he/z_st/z_mean per run),
  - the IDENTICAL CCA protocol: global seeded patient sub-split, option A
    (cross-patient) primary + option B diagnostic, Q_Z-cached perm null,
    (1+sum)/(1+n) perm-p, view_clean circularity quarantine. These primitives
    are replicated here (not imported) so this script does not depend on the
    in-flight v2 run.

niche-side per (run, view, pathway):
  gpath2vec:  Y[n, j] = cos(niche_gpath2vec_vec, member_j_vec)        (as v2)
  aucell:     Y[n, j] = AUCell(niche, member_j)  from the low-level parquet
Same Z, same split, same null -> the ONLY difference is the representation.

RESIDUAL CONFOUND (stated, not hidden)
AUCell here used full pathway gene memberships of low-level pathways; gpath2vec
EA additionally restricted pathway genes to the TF-low gene universe. That is
the one remaining (smallest) difference. A perfect-match AUCell would re-run
decoupler AUCell against gpath2vec.ea.filter_pathways(level='low',
gene_filter=tf_genes) gene sets on niche_counts_cache. decoupler is not in this
env and that recompute is heavy; the recipe is in RECOMPUTE_TF below. This
script removes the dominant confounds and is the runnable controlled test.

outputs (out-dir):
  provenance.json
  pathway_sets.parquet      per parent: v2_set_size, aucell_available,
                            shared_size, coverage_in_aucell
  per_pathway_cca.parquet   rows = 8 runs x 3 views x 5 pathways x 2 methods
  headtohead.parquet        wide: per (run,view,pathway) gpath2vec vs aucell z_A
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
DEFAULT_AUCELL = BIO / 'niche_aucell_low_level.parquet'
DEFAULT_OUT = RUNS / 'eval' / 'H3' / 'pathway_cca_gpath2vec_vs_aucell_tflow'

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
MIN_TESTABLE_SET = 5
# same lock as v2: the gpath2vec member-set sizes BEFORE intersecting AUCell.
FROZEN_SET_SIZES = {
    'TGF-beta_Signaling': 2, 'Immune_System': 74, 'ECM_Organization': 14,
    'Cell_Cycle': 14, 'Programmed_Cell_Death': 7,
}

RECOMPUTE_TF = """
true-TF AUCell (optional, heavier, needs decoupler+anndata in an env):
  from gpath2vec.ea import filter_pathways
  gm = filter_pathways(level='low', gene_filter=tf_genes, min_genes=3)
  net = gm.explode('genes').rename(columns={'stId':'source','genes':'target'})
  # AnnData from niche_counts_cache (normalize_total 1e4 -> log1p, as
  # niche_aucell.py), dc.mt.aucell(adata, net=net, tmin=3), then feed the
  # resulting niche x TF-low matrix into this script via --aucell-low-parquet.
"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embeddings-pkl', type=Path, required=True,
                    help='REQUIRED. gpath2vec arm A embeddings pickle. defines '
                         'the member sets and the gpath2vec niche vectors. '
                         'no default: prevents an accidental v1_legacy run.')
    ap.add_argument('--aucell-low-parquet', type=Path, default=DEFAULT_AUCELL,
                    help='niche x low-level-pathway AUCell scores '
                         '(default: the existing niche_aucell_low_level.parquet)')
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--n-perms', type=int, default=N_PERMS)
    ap.add_argument('--n-test-pats', type=int, default=3)
    ap.add_argument('--allow-legacy', action='store_true')
    return ap.parse_args()


def sha256_of(path: Path, buf=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(buf), b''):
            h.update(c)
    return h.hexdigest()


def check_provenance(emb_path, auc_path, allow_legacy):
    emb_path = emb_path.resolve()
    auc_path = auc_path.resolve()
    for p in (emb_path, auc_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    if ('legacy' in emb_path.name) and not allow_legacy:
        raise RuntimeError(
            f'REFUSING legacy embeddings: {emb_path}\nv1_legacy is the '
            'irreproducible first-FDR artifact. pass --allow-legacy only for '
            'an explicitly-labelled diagnostic.')
    prov = {
        'embeddings_path': str(emb_path), 'embeddings_sha256': sha256_of(emb_path),
        'aucell_path': str(auc_path), 'aucell_sha256': sha256_of(auc_path),
        'seed': SEED,
    }
    print(f'  embeddings: {emb_path}\n  aucell    : {auc_path}')
    return prov


def load_descendants():
    ch = defaultdict(set)
    with open(REACTOME / 'ReactomePathwaysRelation.txt') as f:
        for ln in f:
            p = ln.strip().split('\t')
            if len(p) == 2 and p[0].startswith('R-HSA-'):
                ch[p[0]].add(p[1])

    def desc(root):
        seen, st = set(), [root]
        while st:
            n = st.pop()
            for c in ch.get(n, ()):
                if c not in seen:
                    seen.add(c)
                    st.append(c)
        return seen
    return desc


# ---- CCA primitives (replicated from v2; self-contained on purpose) ---------

def perm_p(nulls, obs):
    n = np.asarray(nulls, np.float64)
    n = n[~np.isnan(n)]
    if n.size == 0 or np.isnan(obs):
        return float('nan')
    return float((1 + np.sum(n >= obs)) / (1 + n.size))


def perm_p_two(nulls, obs):
    n = np.asarray(nulls, np.float64)
    n = n[~np.isnan(n)]
    if n.size == 0 or np.isnan(obs):
        return float('nan')
    return float((1 + np.sum(np.abs(n) >= abs(obs))) / (1 + n.size))


def bh_fdr(pv):
    p = np.asarray(pv, np.float64)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return out
    q = p[ok]
    m = len(q)
    o = np.argsort(q)
    adj = q[o] * m / np.arange(1, m + 1)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    v = np.empty(m)
    v[o] = adj
    out[ok] = v
    return out


def top_cc(Z, Y):
    Zc = Z - Z.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Qz, Rz = np.linalg.qr(Zc, mode='reduced')
    Qy, Ry = np.linalg.qr(Yc, mode='reduced')
    U, S, Vt = np.linalg.svd(Qz.T @ Qy, full_matrices=False)
    a = np.linalg.lstsq(Rz, U[:, 0], rcond=None)[0]
    b = np.linalg.lstsq(Ry, Vt[0], rcond=None)[0]
    return float(np.clip(S[0], -1, 1)), a, b


def top_cc_val(Z, Y):
    Zc = Z - Z.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Qz, _ = np.linalg.qr(Zc, mode='reduced')
    Qy, _ = np.linalg.qr(Yc, mode='reduced')
    return float(np.clip(np.linalg.svd(Qz.T @ Qy, compute_uv=False)[0], -1, 1))


def dirs_cached(Qz, Rz, Ycp):
    Qy, Ry = np.linalg.qr(Ycp, mode='reduced')
    U, _, Vt = np.linalg.svd(Qz.T @ Qy, full_matrices=False)
    a = np.linalg.lstsq(Rz, U[:, 0], rcond=None)[0]
    b = np.linalg.lstsq(Ry, Vt[0], rcond=None)[0]
    return a, b


def val_cached(Qz, Ycp):
    Qy, _ = np.linalg.qr(Ycp, mode='reduced')
    return float(np.clip(np.linalg.svd(Qz.T @ Qy, compute_uv=False)[0], -1, 1))


def global_test_patients(n_test):
    pats = set()
    for r in RUN_IDS:
        p = RUNS / r / 'embeddings_test.parquet'
        if p.is_file():
            pats |= set(pd.read_parquet(p, columns=['patient_id'])
                        ['patient_id'].astype(int).tolist())
    u = np.array(sorted(pats))
    rng = np.random.default_rng(SEED)
    rng.shuffle(u)
    test = set(u[:n_test].tolist())
    print(f'  global held-out patients (seed {SEED}): {sorted(test)} of {len(u)}')
    return test


def cca_cell(Z, Y, train_mask, test_mask, n_perms):
    """one (run,view,pathway,method) cell. returns dict of stats."""
    n_full = len(Z)
    n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
    obs_B = top_cc_val(Z, Y)
    obs_A = float('nan')
    nulls_A = np.full(n_perms, np.nan)
    if n_tr >= 3 and n_te >= 3 and Y.shape[1] >= 1:
        Ztr, Ytr = Z[train_mask], Y[train_mask]
        Zte, Yte = Z[test_mask], Y[test_mask]
        _, a, b = top_cc(Ztr, Ytr)
        Ztc = Zte - Zte.mean(0, keepdims=True)
        Ytc = Yte - Yte.mean(0, keepdims=True)
        ua, va = Ztc @ a, Ytc @ b
        if ua.std() > 0 and va.std() > 0:
            obs_A = float(pearsonr(ua, va)[0])
        Ztrc = Ztr - Ztr.mean(0, keepdims=True)
        Ytrc = Ytr - Ytr.mean(0, keepdims=True)
        Qz, Rz = np.linalg.qr(Ztrc, mode='reduced')
        rng = np.random.default_rng(SEED + 1)
        for i in range(n_perms):
            a_s, b_s = dirs_cached(Qz, Rz, Ytrc[rng.permutation(n_tr)])
            us, vs = Ztc @ a_s, Ytc @ b_s
            if us.std() > 0 and vs.std() > 0:
                nulls_A[i] = float(pearsonr(us, vs)[0])
    Zc = Z - Z.mean(0, keepdims=True)
    QzB, _ = np.linalg.qr(Zc, mode='reduced')
    Yc = Y - Y.mean(0, keepdims=True)
    rngB = np.random.default_rng(SEED)
    nB = np.array([val_cached(QzB, Yc[rngB.permutation(n_full)])
                   for _ in range(n_perms)])
    nA = nulls_A[~np.isnan(nulls_A)]
    return {
        'n_full': n_full, 'n_train': n_tr, 'n_test': n_te,
        'top_corr_test': obs_A, 'top_corr_full': obs_B,
        'p_two_A': perm_p_two(nA, obs_A), 'p_one_B': perm_p(nB, obs_B),
        'z_A': ((obs_A - nA.mean()) / (nA.std() + 1e-12)
                if nA.size and not np.isnan(obs_A) else float('nan')),
        'z_B': (obs_B - nB.mean()) / (nB.std() + 1e-12),
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print('=== provenance ===')
    prov = check_provenance(args.embeddings_pkl, args.aucell_low_parquet,
                            args.allow_legacy)

    print('\n=== member sets (v2-identical) INTERSECT AUCell-low ===')
    with open(args.embeddings_pkl, 'rb') as f:
        emb = pickle.load(f)
    embedded = {k for k in emb if isinstance(k, str) and k.startswith('R-HSA-')}
    desc = load_descendants()

    auc = pd.read_parquet(args.aucell_low_parquet)
    auc['niche_id'] = (auc['subarray'].astype(str) + '::'
                       + auc['spot_id'].astype(str))
    # AUCell columns are "R-HSA-xxxx|Name" -> map stId to column
    stid_to_col = {}
    for c in auc.columns:
        if isinstance(c, str) and c.startswith('R-HSA-'):
            stid_to_col[c.split('|', 1)[0]] = c
    auc_stids = set(stid_to_col)

    members, set_rows = {}, []
    for name, pid in NAMED_TARGETS.items():
        v2_ids = ([pid] if pid in embedded else []) + sorted(
            desc(pid) & embedded)
        shared = [s for s in v2_ids if s in auc_stids]
        if name in FROZEN_SET_SIZES and len(v2_ids) != FROZEN_SET_SIZES[name]:
            raise RuntimeError(
                f'member-set lock failed for {name}: got {len(v2_ids)} '
                f'expected {FROZEN_SET_SIZES[name]} (Reactome/embedding drift)')
        members[name] = shared
        set_rows.append({
            'pathway_name': name, 'pathway_id': pid,
            'v2_set_size': len(v2_ids), 'aucell_available': len(auc_stids),
            'shared_size': len(shared),
            'coverage_in_aucell': len(shared) / len(v2_ids) if v2_ids else 0.0,
            'testable': len(shared) >= MIN_TESTABLE_SET,
        })
        print(f'  {name:24s} v2={len(v2_ids):3d} shared_with_aucell={len(shared):3d}'
              f'  {"TESTABLE" if len(shared) >= MIN_TESTABLE_SET else "UNDERPOWERED"}')
    sets_df = pd.DataFrame(set_rows)
    (args.out_dir / 'provenance.json').write_text(json.dumps(prov, indent=2))
    sets_df.to_parquet(args.out_dir / 'pathway_sets.parquet', index=False)

    # gpath2vec niche vectors (L2-normed), keyed by subarray::spot
    print('\n=== gpath2vec niche vectors ===')
    nkeys = [k for k in emb if isinstance(k, str) and k.startswith('cluster_')]
    Nmat = np.stack([emb[k].astype(np.float64) for k in nkeys])
    Nmat /= (np.linalg.norm(Nmat, axis=1, keepdims=True) + 1e-12)
    g_niche = pd.DataFrame(
        Nmat, index=[k.rsplit('__', 1)[-1] for k in nkeys])
    g_niche = g_niche[~g_niche.index.duplicated(keep='first')]
    # member pathway vectors (L2-normed)
    memvec = {}
    for name, ids in members.items():
        if not ids:
            continue
        P = np.stack([emb[i].astype(np.float64) for i in ids])
        P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        memvec[name] = P
    del emb

    test_pats = global_test_patients(args.n_test_pats)

    print('\n=== CCA grid: both methods, shared member set ===')
    rows = []
    for run_id in RUN_IDS:
        ep = RUNS / run_id / 'embeddings_test.parquet'
        if not ep.is_file():
            continue
        e = pd.read_parquet(ep).reset_index()
        e['niche_id'] = e['subarray'].astype(str) + '::' + e['spot_id'].astype(str)
        for view in VIEWS:
            if view == 'z_mean':
                Zf = 0.5 * (np.stack([np.asarray(v, np.float64) for v in e['z_he']])
                            + np.stack([np.asarray(v, np.float64) for v in e['z_st']]))
            else:
                Zf = np.stack([np.asarray(v, np.float64) for v in e[view]])
            view_clean = view == 'z_he' or run_id not in GPATH2VEC_ST_INPUT_RUNS
            for name, ids in members.items():
                if len(ids) < 1:
                    continue
                cols = [stid_to_col[s] for s in ids]
                a_sub = auc[['niche_id'] + cols].drop_duplicates('niche_id')
                # join on niches present in run, gpath2vec, and aucell
                base = e[['niche_id', 'patient_id']].merge(
                    a_sub, on='niche_id', how='inner')
                base = base[base['niche_id'].isin(g_niche.index)]
                if len(base) < 10:
                    continue
                pos = e.set_index('niche_id').index.get_indexer(base['niche_id'])
                Z = Zf[pos]
                pats = base['patient_id'].astype(int).to_numpy()
                te = np.isin(pats, list(test_pats))
                tr = ~te
                Y_auc = base[cols].to_numpy(np.float64)
                Y_g = g_niche.loc[base['niche_id']].to_numpy() @ memvec[name].T
                for method, Y in (('gpath2vec', Y_g), ('aucell_lowmatched', Y_auc)):
                    st = cca_cell(Z, Y, tr, te, args.n_perms)
                    st.update({'run_id': run_id, 'view': view,
                               'pathway_name': name, 'method': method,
                               'shared_size': len(ids), 'view_clean': view_clean})
                    rows.append(st)
                print(f'  {run_id} {view:6s} {name:22s} shared={len(ids):3d} '
                      f'g.z_A={rows[-2]["z_A"]:+.2f} auc.z_A={rows[-1]["z_A"]:+.2f} '
                      f'clean={view_clean}')

    df = pd.DataFrame(rows)
    # FDR within testable, clean, option A, per method
    df['fdr_A'] = np.nan
    for m in df['method'].unique():
        sel = df[(df['method'] == m) & (df['view'] == 'z_he')
                 & (df['shared_size'] >= MIN_TESTABLE_SET)]
        if len(sel):
            df.loc[sel.index, 'fdr_A'] = bh_fdr(sel['p_two_A'].values)
    df['sig_A_05'] = df['fdr_A'] < 0.05
    df.to_parquet(args.out_dir / 'per_pathway_cca.parquet', index=False)

    # head-to-head wide (z_he, clean only)
    h = df[(df['view'] == 'z_he')].pivot_table(
        index=['run_id', 'pathway_name', 'shared_size'],
        columns='method', values='z_A').reset_index()
    h.to_parquet(args.out_dir / 'headtohead.parquet', index=False)

    print('\n=== HEAD-TO-HEAD (z_he, option A z_A; per pathway, no aggregate) ===')
    print('  set_size shown; TGF-beta/sparse pathways are underpowered by '
          'construction, not a method verdict.')
    for _, r in h.sort_values(['pathway_name', 'run_id']).iterrows():
        g = r.get('gpath2vec', float('nan'))
        a = r.get('aucell_lowmatched', float('nan'))
        tag = '' if r['shared_size'] >= MIN_TESTABLE_SET else '  [UNDERPOWERED]'
        win = 'gpath2vec' if (g == g and a == a and g > a) else (
            'aucell' if (g == g and a == a) else 'NA')
        print(f'  {r["run_id"]:3s} {r["pathway_name"]:22s} '
              f'n={int(r["shared_size"]):3d}  g={g:+.2f}  auc={a:+.2f}  '
              f'-> {win}{tag}')
    print('\nresidual confound reminder: AUCell here used full low-level '
          'pathway memberships; gpath2vec EA additionally restricted to the '
          'TF-low gene universe. dominant confounds (all-level, gene-union) '
          'are removed; this is the runnable controlled test. true-TF recompute'
          ' recipe in module docstring RECOMPUTE_TF.')


if __name__ == '__main__':
    main()