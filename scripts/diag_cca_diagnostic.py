"""
diag_cca_diagnostic.py - understand what CCA is capturing in rank2 baseline.

two questions:
  1. is CCA's R@1=0.318 driven by mc_weights (14d), novae (64d), or tls (1d)?
     run CCA on each component separately
  2. is 0.318 within-subarray retrieval or cross-subarray genuine?
     run CCA cross-subarray (held-out subarray)
"""

import warnings
warnings.filterwarnings('ignore')

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

ROOT = Path(__file__).resolve().parent.parent
V2_DIR = ROOT / 'data' / 'embeddings' / 'virchow2_niche'
NOVAE_DIR = ROOT / 'data' / 'embeddings' / 'novae_all'
BIO_DIR = ROOT / 'data' / 'embeddings' / 'biological_signals'
SEED = 42
N_TEST_SUBARRAYS = 20


def load_bundles(max_subarrays):
    mc_weights_df = pd.read_csv(BIO_DIR / 'mc_weights.tsv', sep='\t')
    tls_df = pd.read_csv(BIO_DIR / 'tls_scores.tsv', sep='\t')
    tls_df['short'] = tls_df['subarray_id'].str.replace(r'^TNBC\d+_', '', regex=True)

    novae_lookup = {}
    for f in NOVAE_DIR.glob('*.npy'):
        parts = f.stem.split('_', 1)
        if len(parts) == 2 and parts[0].startswith('TNBC'):
            novae_lookup[parts[1]] = f

    v2_lookup = {}
    for f in V2_DIR.glob('*.npy'):
        if f.stem.endswith('_meta'):
            continue
        parts = f.stem.split('_', 1)
        if len(parts) == 2 and parts[0].startswith('TNBC'):
            patient = int(parts[0].replace('TNBC', ''))
            v2_lookup[parts[1]] = (f, patient)

    items = list(v2_lookup.items())
    rng = np.random.RandomState(SEED)
    chosen = rng.choice(len(items), min(max_subarrays * 3, len(items)), replace=False)
    items = [items[i] for i in chosen]

    bundles = {}
    for subarray, (v2_path, patient) in items:
        if len(bundles) >= max_subarrays:
            break
        v2 = np.load(v2_path).astype(np.float32)
        slide, pos = subarray.split('_')
        rdata_path = ROOT / 'data' / 'inputs' / 'byArray' / slide / pos / 'selection.RData'
        if not rdata_path.exists():
            continue
        r = subprocess.run(['Rscript', '-e',
                            f'load("{rdata_path}"); cat(rownames(cnts), sep="\\n")'],
                           capture_output=True, text=True)
        spot_order = [s for s in r.stdout.strip().split('\n') if s]
        if len(spot_order) != v2.shape[0]:
            continue

        mw_sub = mc_weights_df[mc_weights_df['subarray'] == subarray].set_index('spot_id')
        tls_sub = tls_df[tls_df['short'] == subarray].set_index('spot_id')
        if len(mw_sub) == 0:
            continue

        novae_emb = None
        if subarray in novae_lookup:
            tmp = np.load(novae_lookup[subarray]).astype(np.float32)
            if tmp.shape[0] == v2.shape[0]:
                novae_emb = tmp

        v2_rows, mw_rows, novae_rows, tls_rows = [], [], [], []
        mc_cols = [f'mc{k}' for k in range(1, 15)]
        for i, sid in enumerate(spot_order):
            if sid not in mw_sub.index:
                continue
            v2_rows.append(v2[i])
            mw_rows.append(mw_sub.loc[sid, mc_cols].values.astype(np.float32))
            tls_val = float(tls_sub.loc[sid, 'tls_score']) if sid in tls_sub.index else 0.0
            tls_rows.append([tls_val])
            novae_rows.append(novae_emb[i] if novae_emb is not None else np.zeros(64, dtype=np.float32))

        if len(v2_rows) < 200:
            continue

        bundles[subarray] = {
            'v2': np.stack(v2_rows),
            'mw': np.stack(mw_rows),
            'novae': np.stack(novae_rows),
            'tls': np.array(tls_rows, dtype=np.float32),
            'patient': patient,
        }
    return bundles


def cca_retrieval(X, Y, n_components=10):
    """fit CCA on (X, Y), compute R@K self-retrieval in projected space."""
    n = X.shape[0]
    k = min(n_components, X.shape[1], Y.shape[1], n - 1)
    cca = CCA(n_components=k, max_iter=200)
    Xc, Yc = cca.fit_transform(X, Y)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8)
    Yn = Yc / (np.linalg.norm(Yc, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Yn.T
    order = (-sim).argsort(axis=1)
    r1 = r5 = r10 = 0
    for i in range(n):
        rank = int(np.where(order[i] == i)[0][0])
        if rank == 0: r1 += 1
        if rank < 5: r5 += 1
        if rank < 10: r10 += 1
    return r1/n, r5/n, r10/n


def cca_retrieval_held_out(X_train, Y_train, X_test, Y_test, n_components=10):
    """fit CCA on train, project test, compute R@K self-retrieval on test."""
    k = min(n_components, X_train.shape[1], Y_train.shape[1])
    cca = CCA(n_components=k, max_iter=200)
    cca.fit(X_train, Y_train)
    Xc = cca.transform(X_test)
    Yc = cca.transform_y(Y_test) if hasattr(cca, 'transform_y') else cca.transform(Y_test)
    # sklearn CCA.transform takes only X by default - use both
    Xc, Yc = cca.transform(X_test), cca.predict(X_test)  # not what we want
    return None  # we'll do it manually


def cca_held_out_retrieval(X_train, Y_train, X_test, Y_test, n_components=10):
    """proper held-out: fit CCA on train, transform both test sides."""
    k = min(n_components, X_train.shape[1], Y_train.shape[1])
    cca = CCA(n_components=k, max_iter=200)
    cca.fit(X_train, Y_train)
    # transform both X_test and Y_test using fitted CCA (returns both)
    Xc, Yc = cca.transform(X_test, Y_test)
    n = X_test.shape[0]
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8)
    Yn = Yc / (np.linalg.norm(Yc, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Yn.T
    order = (-sim).argsort(axis=1)
    r1 = r5 = r10 = 0
    for i in range(n):
        rank = int(np.where(order[i] == i)[0][0])
        if rank == 0: r1 += 1
        if rank < 5: r5 += 1
        if rank < 10: r10 += 1
    return r1/n, r5/n, r10/n


print('=== CCA diagnostic ===\n')
print(f'loading {N_TEST_SUBARRAYS} subarrays...')
bundles = load_bundles(N_TEST_SUBARRAYS)
print(f'loaded {len(bundles)} subarrays\n')

# === QUESTION 1: which ST component drives CCA correlation? ===
print('=== Q1: which ST component drives CCA? ===')
print('per-subarray CCA, average across subarrays\n')

components = {
    'mc_weights only (14d)': lambda b: b['mw'],
    'novae only (64d)': lambda b: b['novae'],
    'tls only (1d)': lambda b: b['tls'],
    'mc + novae (78d)': lambda b: np.concatenate([b['mw'], b['novae']], axis=1),
    'mc + tls (15d)': lambda b: np.concatenate([b['mw'], b['tls']], axis=1),
    'novae + tls (65d)': lambda b: np.concatenate([b['novae'], b['tls']], axis=1),
    'rich ST full (79d)': lambda b: np.concatenate([b['mw'], b['novae'], b['tls']], axis=1),
}

results_q1 = {}
for name, fn in components.items():
    r1s, r5s, r10s = [], [], []
    for sub in bundles:
        b = bundles[sub]
        try:
            r1, r5, r10 = cca_retrieval(b['v2'], fn(b))
            r1s.append(r1); r5s.append(r5); r10s.append(r10)
        except Exception as e:
            pass
    results_q1[name] = (np.mean(r1s), np.mean(r5s), np.mean(r10s))
    print(f'  {name:<28} R@1={np.mean(r1s):.4f}  R@5={np.mean(r5s):.4f}  R@10={np.mean(r10s):.4f}')

# === QUESTION 2: within-subarray vs cross-subarray ===
print('\n=== Q2: within-subarray vs cross-subarray CCA ===')
print('hold one subarray out, fit CCA on rest, evaluate on held-out\n')

# pool train (all but last) and test (last subarray)
sub_keys = sorted(bundles.keys())
test_sub = sub_keys[-1]
train_subs = sub_keys[:-1]

X_train = np.concatenate([bundles[s]['v2'] for s in train_subs])
Y_train_mw = np.concatenate([bundles[s]['mw'] for s in train_subs])
Y_train_full = np.concatenate([
    np.concatenate([bundles[s]['mw'], bundles[s]['novae'], bundles[s]['tls']], axis=1)
    for s in train_subs
])

X_test = bundles[test_sub]['v2']
Y_test_mw = bundles[test_sub]['mw']
Y_test_full = np.concatenate([bundles[test_sub]['mw'], bundles[test_sub]['novae'], bundles[test_sub]['tls']], axis=1)

print(f'train: {len(train_subs)} subarrays, {X_train.shape[0]} spots')
print(f'test:  1 subarray ({test_sub}), {X_test.shape[0]} spots')

# within-subarray reference: train and test on same held-out subarray
print(f'\nwithin-subarray CCA on {test_sub}:')
for name, Y in [('mc_weights only', Y_test_mw), ('full rich ST', Y_test_full)]:
    r1, r5, r10 = cca_retrieval(X_test, Y)
    print(f'  {name:<20} R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}')

# cross-subarray: fit on train, eval on test
print(f'\ncross-subarray CCA (fit on {len(train_subs)} train, eval on held-out):')
for name, (Yt, Yh) in [
    ('mc_weights only', (Y_train_mw, Y_test_mw)),
    ('full rich ST', (Y_train_full, Y_test_full)),
]:
    r1, r5, r10 = cca_held_out_retrieval(X_train, Yt, X_test, Yh)
    print(f'  {name:<20} R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}')

print('\n=== interpretation ===')
print('if cross-subarray R@1 collapses (-> 0.001):')
print('  CCA was exploiting per-subarray structure (shortcut)')
print('  the MLP needs to learn cross-subarray-stable features')
print('  the 0.318 number is an upper bound from leakage')
print('if cross-subarray R@1 stays high (> 0.1):')
print('  CCA found genuine cross-modal biology')
print('  this is a strong baseline the MLP needs to beat')