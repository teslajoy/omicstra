"""H3 multivariate per-pathway CCA using gpath2vec pathway embeddings (v2).

proposal H3 (verbatim): "Reactome pathway embeddings computed via gpath2vec for five
cancer-relevant pathways (TGF-beta Signaling, Immune System, Extracellular Matrix
Organization, Cell Cycle, Programmed Cell Death) will correlate with specific directions
in the shared latent space via canonical correlation analysis, indicating preservation of
interpretable biological signal."

this is the proposal-faithful H3 ("shape B" multivariate). the AUCell variant
(eval_h3_pathway_cca.py) is the orthogonal robustness check.

------------------------------------------------------------------------------
WHAT CHANGED vs eval_h3_pathway_cca_gpath2vec.py (and why)

1. PROVENANCE LOCK (was: set-size only). The old "F1 lock" asserted member-set
   SIZES, which are identical for v1_legacy and arm A (same 680-pathway universe)
   -> it passed for either file and could silently run on the irreproducible
   first-FDR v1 embeddings. v2 records the embeddings path + sha256 into
   provenance.json + pathway_axes.parquet, and HARD-FAILS on a *_v1_legacy*.pkl
   path unless --allow-legacy is explicitly passed. There is no default embeddings
   path: --embeddings-pkl is required, so no accidental legacy run.

2. REPORTING INTEGRITY. The old summary printed a flat "X/5 sig", conflating
   "untestable by construction" (TGF-beta set_size=2; Cell Cycle 11% subtree
   coverage) with "tested and null". v2 reports PER-(run,pathway) with set_size
   + coverage_frac + z_A + sig all visible; it emits NO aggregate "K/5 sig"
   headline (that aggregate is exactly what hides degeneracy). set_size<
   MIN_TESTABLE_SET is flagged UNDERPOWERED and excluded from the FDR family,
   but coverage is NOT thresholded: measured arm-A coverage is ECM 74%,
   Immune 34%, PCD 17%, TGF-beta 17%, Cell Cycle 11% -- any COVERAGE_MIN
   separating PCD (17%) from Cell Cycle (11%) is a knife-edge arbitrary knob
   (the F1 anti-pattern). Instead coverage_frac rides in per_pathway_cca.parquet
   and every per-pathway line, so the reader judges representativeness from raw
   numbers and the writeup must state it per pathway. Only ECM/Immune are
   well-covered; PCD/CellCycle/TGF-beta are sparse and must be reported as such.

3. OPERATIONALIZATION is recorded explicitly. For 4 of 5 targets the parent node
   is NOT embedded; the representation is a niche cosine profile to a bag of
   embedded low-level descendants, not a single "pathway embedding". axes_df
   carries parent_embedded + n_children_embedded + coverage_frac so the writeup
   can state this rather than imply per-pathway vectors.

4. GLOBAL patient sub-split (was: per-cell). The old patient_sub_split ran inside
   the (run,view,pathway) loop on whatever patients survived each join -> the 5
   pathways could be tested on DIFFERENT held-out patients, making "X/5" not a
   coherent within-run statistic. v2 fixes the held-out patient set ONCE,
   globally, seeded, and reuses it for every cell.

5. PERMUTATION P with the (1+sum)/(1+n) correction (was: (null>=obs).mean(),
   which can be 0 and is anti-conservative). Applied to both option A and B.

6. OPTION B is computed but quarantined. With set_size up to 74 and 512-d Z the
   top canonical correlation saturates (~1 for obs AND null); z_B is not
   interpretable. v2 keeps the column for diagnostics but the headline summary
   reports option A only; option B prints under an explicit
   "diagnostic (saturates at high set_size, not a result)" banner.

KEPT (was already correct): the CCA QR+SVD math; the Q_Z-cache invariance
(permutations shuffle Y rows only; Z, Q_Z, R_Z, column means invariant ->
bit-identical, see eval_h3_pathway_cca_perms.py discipline); the circularity
quarantine (scripts/align.py default st_features include gpath2vec_niche, so
R1/R2/R3/R4 z_st/z_mean are partially circular -> view_clean, z_he-only headline).
------------------------------------------------------------------------------

per pathway:
  pathway set = {parent if embedded} U {embedded descendants}  (512-d gpath2vec vec each)
  for each niche, Y row = (niche_gpath2vec . member_j) cosine for j in set
  Z = aligned latent (z_he | z_st | z_mean), 512-d per niche
  multivariate CCA(Z, Y) via QR + SVD; option B = fit-on-all-test,
  option A = global patient sub-split (fit on train, eval on held-out patients),
  perm null shuffles Y rows.

outputs (parallel to AUCell H3 layout):
  provenance.json              embeddings path + sha256 + git-ish run metadata
  pathway_axes.parquet         5 rows: name,id,parent_embedded,n_children_embedded,
                               total_descendants,coverage_frac,set_size,testable
  per_pathway_cca.parquet      120 rows = 8 runs x 3 views x 5 pathways
  perm_nulls.parquet           long-form perm draws for diagnostics
"""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / 'data' / 'embeddings' / 'biological_signals'
REACTOME = ROOT / 'knowledge' / 'reactome'
RUNS = ROOT / 'runs' / 'tnbc-92'
DEFAULT_OUT = RUNS / 'eval' / 'H3' / 'pathway_cca_gpath2vec'

NAMED_TARGETS = {
    'TGF-beta_Signaling':    'R-HSA-170834',
    'Immune_System':         'R-HSA-168256',
    'ECM_Organization':      'R-HSA-1474244',
    'Cell_Cycle':            'R-HSA-1640170',
    'Programmed_Cell_Death': 'R-HSA-5357801',
}

# mutable defaults; main() overrides RUNS / RUN_IDS / GPATH2VEC_ST_INPUT_RUNS
# from --runs-dir / --runs for the v3 retrain grid.
RUN_IDS = ['R1', 'R2', 'R3', 'R4', 'R6', 'B1', 'B2', 'B3']
VIEWS = ['z_he', 'z_st', 'z_mean']
GPATH2VEC_ST_INPUT_RUNS = {'R1', 'R2', 'R3', 'R4'}
SEED = 42
N_PERMS = 500

# a pathway whose member set is smaller than this is reported as underpowered,
# never folded into a sig/null count. TGF-beta R-HSA-170834 (set_size=2) is the
# motivating case: a 2-column CCA target carries no usable signal.
MIN_TESTABLE_SET = 5

# AUCell-comparability lock: member-set sizes for the frozen 395-node set,
# verified 2026-05-17 (proposal_deviations.md "H3 gpath2vec arm"). This still
# guards the Reactome-snapshot / 111-vs-114 drift class. It does NOT (and cannot)
# guard embedding provenance -- see check_provenance() for that.
FROZEN_SET_SIZES = {
    'TGF-beta_Signaling': 2,
    'Immune_System': 74,
    'ECM_Organization': 14,
    'Cell_Cycle': 14,
    'Programmed_Cell_Death': 7,
}


_HELP_LEGACY = ('permit a *_v1_legacy*.pkl embeddings path. off by default: '
                'legacy embeddings are the irreproducible first-FDR artifact '
                'and must not back a paper number.')
_HELP_DRIFT = ('permit member-set sizes to diverge from the arm-A frozen '
               'low-level set ({TGFb:2,Immune:74,ECM:14,CellCycle:14,PCD:7}). '
               'off by default: silent drift invalidates head-to-head vs '
               'AUCell per_node_cca. opt-in for an explicitly different '
               'hierarchy (e.g. level=all rebuild); recorded in provenance.json '
               'and the run is tagged not-comparable to arm-A numbers.')
_HELP_RESTRICT = ('rank-matched mode: restrict member sets to the v1 arm-A '
                  'frozen 395-node hierarchy (dag_metadata INTERSECT '
                  'per_node_cca). isolates representation from coverage '
                  'breadth. recorded in provenance.json.')


# ---------- provenance: the lock the old script was missing -------------------

def sha256_of(path: Path, buf=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(buf), b''):
            h.update(chunk)
    return h.hexdigest()


def check_provenance(emb_path: Path, allow_legacy: bool) -> dict:
    emb_path = emb_path.resolve()
    if not emb_path.is_file():
        raise FileNotFoundError(f'embeddings not found: {emb_path}')
    is_legacy = 'v1_legacy' in emb_path.name or '_legacy' in emb_path.name
    if is_legacy and not allow_legacy:
        raise RuntimeError(
            f'REFUSING to run on a legacy embeddings file:\n  {emb_path}\n'
            'v1_legacy is the irreproducible first-FDR artifact (see '
            'proposal_deviations.md / LEGACY_V1_NOTES.md). a head-to-head H3 '
            'number must come from the reproducible min-FDR arm. pass '
            '--allow-legacy only for an explicitly-labelled diagnostic run.')
    digest = sha256_of(emb_path)
    prov = {
        'embeddings_path': str(emb_path),
        'embeddings_sha256': digest,
        'is_legacy_path': is_legacy,
        'allow_legacy': allow_legacy,
        'seed': SEED,
    }
    print(f'  embeddings : {emb_path}')
    print(f'  sha256     : {digest}')
    print(f'  legacy?    : {is_legacy} (allow_legacy={allow_legacy})')
    return prov


# ---------- phase 1: build {parent + embedded children} per pathway ----------

def load_relations():
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


def build_pathway_sets(node_emb, descendants_of, restrict_to=None):
    """per parent: {parent if embedded} U {embedded descendants}.
    if restrict_to is given (set of R-HSA ids), the member set is further
    intersected with it - this is the rank-matched comparison knob (e.g. clip
    to the arm-A frozen 395-node hierarchy so a different build is scored on
    the same nodes as arm A, isolating representation from coverage breadth).
    returns (axes_df, members: name -> (ids, m x 512 L2-normalized matrix))."""
    embedded = {k for k in node_emb if isinstance(k, str) and k.startswith('R-HSA-')}
    rows, members = [], {}
    for name, pid in NAMED_TARGETS.items():
        all_desc = descendants_of(pid)
        ids = []
        if pid in embedded and (restrict_to is None or pid in restrict_to):
            ids.append(pid)
        descendant_pool = (all_desc & embedded if restrict_to is None
                           else all_desc & embedded & restrict_to)
        ids.extend(sorted(descendant_pool))
        if not ids:
            raise RuntimeError(f'{pid} ({name}): no embedded parent or descendants'
                               + (' (after restrict)' if restrict_to else ''))
        P = np.stack([node_emb[i].astype(np.float64) for i in ids])
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        members[name] = (ids, P)
        n_children = len(ids) - (1 if pid in embedded else 0)
        total_desc = len(all_desc)
        cov = (n_children / total_desc) if total_desc else float('nan')
        set_size = len(ids)
        rows.append({
            'pathway_name': name,
            'pathway_id': pid,
            'parent_embedded': pid in embedded,
            'n_children_embedded': n_children,
            'total_descendants': total_desc,
            'coverage_frac': cov,
            'set_size': set_size,
            'testable': set_size >= MIN_TESTABLE_SET,
        })
        tag = 'TESTABLE' if set_size >= MIN_TESTABLE_SET else 'UNDERPOWERED'
        print(f'  {name:24s} set={set_size:4d}  parent_emb={pid in embedded}  '
              f'cov={cov:5.1%}  -> {tag}')
    return pd.DataFrame(rows), members


# ---------- phase 2: niche-side cosine tables --------------------------------

def niche_key_to_id(k):
    # arm-A keys: cluster_TNBC10_CN5_D2__TNBC10_CN5_D2::2x10 -> TNBC10_CN5_D2::2x10
    # v2 keys (level=all build): cluster_TNBC10_CN5_D2::2x10 -> TNBC10_CN5_D2::2x10
    # `rsplit('__', 1)` handles arm-A; the no-`__` v2 case needs the cluster_ strip.
    s = k.rsplit('__', 1)[-1]
    return s[len('cluster_'):] if s.startswith('cluster_') else s


def build_niche_cosine_tables(node_emb, members):
    niche_keys = [k for k in node_emb if isinstance(k, str) and k.startswith('cluster_')]
    print(f'  niches in gpath2vec embedding: {len(niche_keys):,}')
    N = np.stack([node_emb[k].astype(np.float64) for k in niche_keys])
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    niche_ids = [niche_key_to_id(k) for k in niche_keys]
    out = {}
    for name, (ids, P) in members.items():
        C = N @ P.T
        df = pd.DataFrame(C, index=niche_ids, columns=ids)
        df = df[~df.index.duplicated(keep='first')]
        df.index.name = 'niche_id'
        out[name] = df
    return out


# ---------- phase 3: CCA primitives (unchanged math) -------------------------

def _as_arr(c):
    return np.asarray(c, dtype=np.float32)


def stack_col(s):
    return np.stack([_as_arr(v) for v in s])


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
    ov = np.empty(n)
    ov[order] = adj
    out[valid] = ov
    return out


def perm_p(nulls, obs):
    """(1 + #{null >= obs}) / (1 + n) -- never 0, not anti-conservative."""
    nulls = np.asarray(nulls, dtype=np.float64)
    nulls = nulls[~np.isnan(nulls)]
    if nulls.size == 0 or np.isnan(obs):
        return float('nan')
    return float((1 + np.sum(nulls >= obs)) / (1 + nulls.size))


def perm_p_two(nulls, obs):
    nulls = np.asarray(nulls, dtype=np.float64)
    nulls = nulls[~np.isnan(nulls)]
    if nulls.size == 0 or np.isnan(obs):
        return float('nan')
    return float((1 + np.sum(np.abs(nulls) >= abs(obs))) / (1 + nulls.size))


def top_canonical_corr(Z, Y):
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


def top_canonical_corr_value_only(Z, Y):
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Q_Z, _ = np.linalg.qr(Zc, mode='reduced')
    Q_Y, _ = np.linalg.qr(Yc, mode='reduced')
    s = float(np.linalg.svd(Q_Z.T @ Q_Y, full_matrices=False, compute_uv=False)[0])
    return float(np.clip(s, -1.0, 1.0))


# Q_Z-cached perm variants: permutations shuffle Y rows only; Z (hence Q_Z, R_Z)
# and column means are invariant within a cell, and Y[idx] has Y's column means
# so Yc[idx] == (Y[idx] - Y.mean). bit-identical to the as-written path.

def _val_cached(Q_Z, Yc_perm):
    Q_Y, _ = np.linalg.qr(Yc_perm, mode='reduced')
    s = float(np.linalg.svd(Q_Z.T @ Q_Y, full_matrices=False, compute_uv=False)[0])
    return float(np.clip(s, -1.0, 1.0))


def _dirs_cached(Q_Z, R_Z, Yc_perm):
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


# ---------- global patient sub-split (fixed once) ----------------------------

def global_test_patients(n_test_pats: int) -> set:
    """held-out patient ids, fixed once from the union of all runs' test
    cohorts, seeded. identical for every (run, view, pathway) cell."""
    pats = set()
    for run_id in RUN_IDS:
        p = RUNS / run_id / 'embeddings_test.parquet'
        if p.is_file():
            df = pd.read_parquet(p, columns=['patient_id'])
            pats |= set(df['patient_id'].astype(int).tolist())
    uniq = np.array(sorted(pats))
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    test = set(uniq[:n_test_pats].tolist())
    print(f'  global held-out patients (seed={SEED}, n={n_test_pats}): '
          f'{sorted(test)}  of {len(uniq)} test-cohort patients')
    return test


def run_cca_grid(cosine_per_pathway, axes_df, n_perms, test_pats):
    testable = dict(zip(axes_df['pathway_name'], axes_df['testable']))
    cov_of = dict(zip(axes_df['pathway_name'], axes_df['coverage_frac']))
    obs_rows, perm_rows = [], []

    for run_id in RUN_IDS:
        print(f'\n  {run_id}: load embeddings + join cosine tables...')
        emb = pd.read_parquet(RUNS / run_id / 'embeddings_test.parquet').reset_index()
        emb['niche_id'] = emb['subarray'].astype(str) + '::' + emb['spot_id'].astype(str)

        for view in VIEWS:
            Z_full = get_view(emb, view).astype(np.float64)

            for pname, pid in NAMED_TARGETS.items():
                cos_df = cosine_per_pathway[pname]
                joined = emb[['niche_id', 'patient_id']].merge(
                    cos_df, left_on='niche_id', right_index=True, how='inner')
                if len(joined) == 0:
                    print(f'      {pname}: no overlap, skipping')
                    continue
                idx_in_emb = emb.set_index('niche_id').index
                jpos = idx_in_emb.get_indexer(joined['niche_id'])
                Z = Z_full[jpos]
                Y = joined[cos_df.columns].to_numpy(dtype=np.float64)
                patients = joined['patient_id'].astype(int).to_numpy()

                # GLOBAL split: same held-out patient identity everywhere
                test_mask = np.isin(patients, list(test_pats))
                train_mask = ~test_mask
                n_full = len(joined)
                n_tr = int(train_mask.sum())
                n_te = int(test_mask.sum())
                set_size = Y.shape[1]
                if n_tr < 3 or n_te < 3:
                    print(f'      {pname}: too few train/test niches '
                          f'({n_tr}/{n_te}), skipping option A')

                obs_B = top_canonical_corr_value_only(Z, Y)

                Z_tr, Y_tr = Z[train_mask], Y[train_mask]
                Z_te, Y_te = Z[test_mask], Y[test_mask]
                obs_A = float('nan')
                obs_train = float('nan')
                nulls_A = np.full(n_perms, np.nan)
                if n_tr >= 3 and n_te >= 3:
                    _, a_tr, b_tr = top_canonical_corr(Z_tr, Y_tr)
                    Z_te_c = Z_te - Z_te.mean(axis=0, keepdims=True)
                    Y_te_c = Y_te - Y_te.mean(axis=0, keepdims=True)
                    u_te, v_te = Z_te_c @ a_tr, Y_te_c @ b_tr
                    if u_te.std() > 0 and v_te.std() > 0:
                        obs_A = float(pearsonr(u_te, v_te)[0])
                    Z_tr_c = Z_tr - Z_tr.mean(axis=0, keepdims=True)
                    Y_tr_c = Y_tr - Y_tr.mean(axis=0, keepdims=True)
                    u_tr, v_tr = Z_tr_c @ a_tr, Y_tr_c @ b_tr
                    if u_tr.std() > 0 and v_tr.std() > 0:
                        obs_train = float(pearsonr(u_tr, v_tr)[0])
                    Qz_tr, Rz_tr = np.linalg.qr(Z_tr_c, mode='reduced')
                    rng_A = np.random.default_rng(SEED + 1)
                    for i in range(n_perms):
                        idx = rng_A.permutation(n_tr)
                        a_s, b_s = _dirs_cached(Qz_tr, Rz_tr, Y_tr_c[idx])
                        u_s, v_s = Z_te_c @ a_s, Y_te_c @ b_s
                        if u_s.std() > 0 and v_s.std() > 0:
                            nulls_A[i] = float(pearsonr(u_s, v_s)[0])

                Zc_full = Z - Z.mean(axis=0, keepdims=True)
                Qz_B, _ = np.linalg.qr(Zc_full, mode='reduced')
                Yc_full = Y - Y.mean(axis=0, keepdims=True)
                rng_B = np.random.default_rng(SEED)
                nulls_B = np.empty(n_perms)
                for i in range(n_perms):
                    nulls_B[i] = _val_cached(Qz_B, Yc_full[rng_B.permutation(n_full)])

                view_clean = view == 'z_he' or run_id not in GPATH2VEC_ST_INPUT_RUNS
                nA = nulls_A[~np.isnan(nulls_A)]

                obs_rows.append({
                    'run_id': run_id, 'view': view,
                    'pathway_name': pname, 'pathway_id': pid,
                    'set_size': set_size, 'testable': bool(testable[pname]),
                    'coverage_frac': float(cov_of[pname]),
                    'view_clean': view_clean,
                    'top_corr_full': obs_B,
                    'top_corr_train': obs_train,
                    'top_corr_test': obs_A,
                    'null_B_mean': float(nulls_B.mean()),
                    'null_B_std': float(nulls_B.std()),
                    'null_A_mean': float(nA.mean()) if nA.size else float('nan'),
                    'null_A_std': float(nA.std()) if nA.size else float('nan'),
                    'p_one_B': perm_p(nulls_B, obs_B),
                    'p_two_A': perm_p_two(nA, obs_A),
                    'z_B': (obs_B - float(nulls_B.mean())) / (float(nulls_B.std()) + 1e-12),
                    'z_A': ((obs_A - float(nA.mean())) / (float(nA.std()) + 1e-12)
                            if nA.size and not np.isnan(obs_A) else float('nan')),
                    'n_full': n_full, 'n_train': n_tr, 'n_test': n_te,
                })
                perm_rows.append({
                    'run_id': run_id, 'view': view, 'pathway_name': pname,
                    'null_A': nulls_A.tolist(), 'null_B': nulls_B.tolist(),
                })
                print(f'      {pname:24s} set={set_size:3d} '
                      f'obs_A={obs_A:.3f} z_A={obs_rows[-1]["z_A"]:.2f} '
                      f'testable={testable[pname]} clean={view_clean}')

    df = pd.DataFrame(obs_rows)
    # FDR only over TESTABLE, clean (z_he) option-A cells -- the headline family.
    head = df[(df['testable']) & (df['view'] == 'z_he')].copy()
    df['fdr_A'] = np.nan
    if len(head):
        df.loc[head.index, 'fdr_A'] = bh_fdr(head['p_two_A'].values)
    df['fdr_B'] = bh_fdr(df['p_one_B'].values)  # diagnostic only
    df['sig_A_05'] = df['fdr_A'] < 0.05
    df['sig_B_05'] = df['fdr_B'] < 0.05
    return df, pd.DataFrame(perm_rows)


@click.command()
@click.option('--embeddings-pkl', required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help='REQUIRED. gpath2vec embeddings pickle. no default: '
                   'prevents an accidental v1_legacy run.')
@click.option('--out-dir', type=click.Path(file_okay=False, path_type=Path), default=None,
              help='output dir. defaults to {runs-dir}/eval/H3/pathway_cca_gpath2vec.')
@click.option('--n-perms', type=int, default=N_PERMS, show_default=True)
@click.option('--n-test-pats', type=int, default=3, show_default=True,
              help='held-out patients for the option-A transfer null.')
@click.option('--allow-legacy', is_flag=True, default=False, help=_HELP_LEGACY)
@click.option('--allow-set-size-drift', is_flag=True, default=False, help=_HELP_DRIFT)
@click.option('--restrict-to-arm-a-frozen', is_flag=True, default=False, help=_HELP_RESTRICT)
@click.option('--runs-dir', type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help='parent dir holding {run_id}/embeddings_test.parquet. '
                                 'defaults to runs/tnbc-92/. use runs/tnbc-92_v3/ for v3.')
@click.option('--runs', 'runs_arg', default=None,
              help='comma-separated run ids (e.g. R1_v3,R2_v3,...). defaults to v1 8-run set.')
def main(embeddings_pkl, out_dir, n_perms, n_test_pats, allow_legacy,
         allow_set_size_drift, restrict_to_arm_a_frozen, runs_dir, runs_arg):
    global RUNS, RUN_IDS, GPATH2VEC_ST_INPUT_RUNS
    if runs_dir is not None:
        RUNS = runs_dir.resolve()
    if runs_arg is not None:
        RUN_IDS = [r.strip() for r in runs_arg.split(',') if r.strip()]
        # circular = run whose ST input carries gpath2vec (R1-R4 stems); strip
        # any _v3 suffix so the membership test works on either grid.
        GPATH2VEC_ST_INPUT_RUNS = {
            r for r in RUN_IDS if r.split('_')[0] in {'R1', 'R2', 'R3', 'R4'}}
    args = SimpleNamespace(
        embeddings_pkl=embeddings_pkl,
        out_dir=out_dir if out_dir is not None else RUNS / 'eval' / 'H3' / 'pathway_cca_gpath2vec',
        n_perms=n_perms, n_test_pats=n_test_pats, allow_legacy=allow_legacy,
        allow_set_size_drift=allow_set_size_drift,
        restrict_to_arm_a_frozen=restrict_to_arm_a_frozen,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'runs_dir: {RUNS}')
    print(f'run_ids : {RUN_IDS}')
    print(f'out_dir : {args.out_dir}')
    print(f'n_perms : {args.n_perms}   n_test_pats : {args.n_test_pats}')

    print('\n=== provenance lock ===')
    prov = check_provenance(args.embeddings_pkl, args.allow_legacy)
    prov['n_perms'] = args.n_perms
    prov['n_test_pats'] = args.n_test_pats

    print('\n=== phase 1: embeddings + Reactome descendants ===')
    with open(args.embeddings_pkl, 'rb') as f:
        node_emb = pickle.load(f)
    desc = load_relations()

    print('\n=== phase 2: {parent + embedded children} sets ===')
    restrict_to = None
    if args.restrict_to_arm_a_frozen:
        _DAG = ROOT / 'runs' / 'tnbc-92' / 'eval' / 'H3' / 'pathway_cca' / 'dag_full'
        _dm = pd.read_parquet(_DAG / 'dag_metadata.parquet', columns=['node_id'])
        _pnc = pd.read_parquet(_DAG / 'per_node_cca.parquet', columns=['node_id'])
        restrict_to = set(_dm['node_id']) & set(_pnc['node_id'])
        print(f'  rank-matched mode: restricting members to the arm-A frozen '
              f'{len(restrict_to)}-node hierarchy (dag_metadata INTERSECT '
              f'per_node_cca). isolates representation from coverage breadth.')
        prov['restrict_to_arm_a_frozen'] = True
        prov['arm_a_frozen_node_count'] = len(restrict_to)
    else:
        prov['restrict_to_arm_a_frozen'] = False
    axes_df, members = build_pathway_sets(node_emb, desc, restrict_to=restrict_to)

    actual = {n: int(r) for n, r in zip(axes_df['pathway_name'], axes_df['set_size'])}
    if actual != FROZEN_SET_SIZES:
        if not args.allow_set_size_drift:
            raise RuntimeError(
                'AUCELL-COMPARABILITY LOCK FAILED -- member set sizes diverge from '
                f'the frozen node set.\n  expected: {FROZEN_SET_SIZES}\n  got: {actual}\n'
                'head-to-head vs AUCell per_node_cca is INVALID. re-pin the hierarchy '
                'in proposal_deviations.md before interpreting anything.\n'
                'opt-in with --allow-set-size-drift to run on a different hierarchy '
                '(e.g. level=all gpath2vec rebuild); this is recorded in provenance.json '
                'and disqualifies the run from arm-A head-to-head numbers.')
        print(f'  NOTE: AUCell-comparability lock BYPASSED via '
              f'--allow-set-size-drift; head-to-head with arm-A frozen 395 is '
              f'NOT valid for this run. expected {FROZEN_SET_SIZES}, got {actual}.')
        prov['set_size_drift_allowed'] = True
        prov['frozen_set_sizes_expected'] = FROZEN_SET_SIZES
        prov['frozen_set_sizes_observed'] = actual
    else:
        print(f'  AUCell-comparability lock OK: {FROZEN_SET_SIZES}')
        prov['set_size_drift_allowed'] = False

    n_underpowered = int((~axes_df['testable']).sum())
    if n_underpowered:
        bad = axes_df.loc[~axes_df['testable'], 'pathway_name'].tolist()
        print(f'  NOTE: {n_underpowered} target(s) UNDERPOWERED '
              f'(set_size < {MIN_TESTABLE_SET}): {bad} -- reported separately, '
              'never folded into a sig/null count.')

    prov['min_testable_set'] = MIN_TESTABLE_SET
    prov['coverage_frac'] = {n: float(c) for n, c in
                             zip(axes_df['pathway_name'], axes_df['coverage_frac'])}
    prov['coverage_note'] = ('coverage NOT thresholded; any cut separating '
                             'PCD~17% from Cell_Cycle~11% is arbitrary. '
                             'representativeness reported per-pathway, reader-judged.')
    (args.out_dir / 'provenance.json').write_text(json.dumps(prov, indent=2))
    axes_df.to_parquet(args.out_dir / 'pathway_axes.parquet', index=False)
    print(f'  wrote provenance.json + pathway_axes.parquet')

    print('\n=== phase 3: global patient sub-split ===')
    test_pats = global_test_patients(args.n_test_pats)

    print('\n=== phase 4: niche-vs-set cosine tables ===')
    cos = build_niche_cosine_tables(node_emb, members)

    print('\n=== phase 5: multivariate CCA grid ===')
    df, perm_df = run_cca_grid(cos, axes_df, args.n_perms, test_pats)
    df.to_parquet(args.out_dir / 'per_pathway_cca.parquet', index=False)
    perm_df.to_parquet(args.out_dir / 'perm_nulls.parquet', index=False)
    print(f'\nwrote per_pathway_cca.parquet ({len(df)} rows) + perm_nulls.parquet')

    # ---- per-(run,pathway) report: option A, z_he, NOTHING aggregated -------
    print('\n=== option A (cross-patient transfer), z_he view -- per (run,pathway). '
          'NO K/5 aggregate: aggregates hide input degeneracy ===')
    cov_str = ', '.join(f'{n}={c:.0%}' for n, c in
                        zip(axes_df['pathway_name'], axes_df['coverage_frac']))
    print(f'  subtree coverage (arm A): {cov_str}')
    print('  well-covered: ECM, Immune.  sparse: PCD, Cell_Cycle, TGF-beta.  '
          'state per-pathway in the writeup; never summarize as "K/5 sig".')
    clean = df[df['view'] == 'z_he'].copy()
    hdr = (f'  {"run":4s} {"pathway":22s} {"set":>4s} {"cov":>5s} '
           f'{"z_A":>7s} {"p_two_A":>8s} {"fdr_A":>7s}  status')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for r in RUN_IDS:
        for _, row in clean[clean['run_id'] == r].iterrows():
            if row['set_size'] < MIN_TESTABLE_SET:
                status = f'UNDERPOWERED set<{MIN_TESTABLE_SET}, excl. FDR'
            else:
                status = 'SIG' if row['sig_A_05'] else 'null'
            circ = ' [ST-circular run]' if r in GPATH2VEC_ST_INPUT_RUNS else ''
            fdr = f'{row["fdr_A"]:.3f}' if pd.notna(row['fdr_A']) else '  n/a'
            print(f'  {r:4s} {row["pathway_name"]:22s} {int(row["set_size"]):>4d} '
                  f'{row["coverage_frac"]:>4.0%} {row["z_A"]:>7.2f} '
                  f'{row["p_two_A"]:>8.3f} {fdr:>7s}  {status}{circ}')
    print(f'\n  FDR family = (set_size>={MIN_TESTABLE_SET}) x z_he. coverage is shown, '
          'never thresholded (any PCD~17%/CC~11% cut is arbitrary -- the F1 '
          'anti-pattern); representativeness is the reader\'s call from the cov column.')

    print('\n=== diagnostic ONLY -- option B saturates at high set_size, NOT a result ===')
    for r in RUN_IDS:
        sub = clean[(clean['run_id'] == r) & (clean['set_size'] >= MIN_TESTABLE_SET)]
        if not sub.empty:
            print(f'  {r}: option-B mean z_B={sub["z_B"].mean():.2f} (uninterpretable)')


if __name__ == '__main__':
    main()
