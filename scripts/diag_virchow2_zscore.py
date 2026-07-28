"""
diag_virchow2_zscore.py - z-score per-subarray test for Virchow2 niche.

does z-scoring within each subarray fix Virchow2's cross-subarray geometry?
if yes: mixed-subarray batches viable
if no: per-subarray batches required
"""

import warnings
warnings.filterwarnings('ignore')

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
V2_DIR = ROOT / 'data' / 'embeddings' / 'virchow2_niche'
BIO_DIR = ROOT / 'data' / 'embeddings' / 'biological_signals'
SEED = 42


def load_aligned():
    mc = pd.read_csv(BIO_DIR / 'mc_labels.tsv', sep='\t')
    morph = pd.read_csv(BIO_DIR / 'morphology_labels.tsv', sep='\t')
    morph['short'] = morph['subarray_id'].str.replace(r'^TNBC\d+_', '', regex=True)

    files = sorted(V2_DIR.glob('*.npy'))
    files = [f for f in files if not f.stem.endswith('_meta')]
    lookup = {}
    for f in files:
        parts = f.stem.split('_', 1)
        if len(parts) == 2 and parts[0].startswith('TNBC'):
            short_id = parts[1]
            patient = int(parts[0].replace('TNBC', ''))
            lookup[short_id] = (f, patient)
    print(f'  v2 niche subarrays: {len(lookup)}')

    raw_rows, zscore_rows = [], []
    mc_labels, morph_labels = [], []
    patients, subarrays = [], []

    n_loaded = 0
    for subarray, (path, patient) in lookup.items():
        emb = np.load(path)
        slide, pos = subarray.split('_')
        rdata_path = ROOT / 'data' / 'inputs' / 'byArray' / slide / pos / 'selection.RData'
        if not rdata_path.exists():
            continue
        r = subprocess.run(['Rscript', '-e',
                            f'load("{rdata_path}"); cat(rownames(cnts), sep="\\n")'],
                           capture_output=True, text=True)
        spot_ids = [s for s in r.stdout.strip().split('\n') if s]
        if len(spot_ids) != emb.shape[0]:
            continue

        emb_centered = emb - emb.mean(axis=0)
        emb_std = emb_centered.std(axis=0) + 1e-8
        emb_z = emb_centered / emb_std

        mc_sub = mc[mc['subarray'] == subarray].set_index('spot_id')
        morph_sub = morph[morph['short'] == subarray].set_index('spot_id')

        for i, sid in enumerate(spot_ids):
            mc_label = mc_sub.loc[sid, 'megacluster'] if sid in mc_sub.index else None
            m_label = morph_sub.loc[sid, 'dominant_5class'] if sid in morph_sub.index else None
            if mc_label is None and m_label is None:
                continue
            raw_rows.append(emb[i])
            zscore_rows.append(emb_z[i])
            mc_labels.append(mc_label)
            morph_labels.append(m_label)
            patients.append(patient)
            subarrays.append(subarray)
        n_loaded += 1
        if n_loaded % 50 == 0:
            print(f'    loaded {n_loaded}/{len(lookup)}')

    return (np.array(raw_rows), np.array(zscore_rows),
            mc_labels, morph_labels, np.array(patients), np.array(subarrays))


def silhouette_subsampled(X, y, max_n=10000):
    if any(v is None for v in y):
        keep = [i for i, v in enumerate(y) if v is not None]
        X = X[keep]
        y = [y[i] for i in keep]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    rng = np.random.RandomState(SEED)
    if len(X) > max_n:
        idx = rng.choice(len(X), max_n, replace=False)
        X = X[idx]
        y_enc = y_enc[idx]
    return silhouette_score(X, y_enc, metric='cosine')


def kmeans_ari(X, y):
    if any(v is None for v in y):
        keep = [i for i, v in enumerate(y) if v is not None]
        X = X[keep]
        y = [y[i] for i in keep]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    km = MiniBatchKMeans(n_clusters=n_classes, random_state=SEED, n_init=3, batch_size=1024)
    pred = km.fit_predict(X)
    return adjusted_rand_score(y_enc, pred)


def linear_probe(X, y, groups):
    if any(v is None for v in y):
        keep = [i for i, v in enumerate(y) if v is not None]
        X = X[keep]
        y = [y[i] for i in keep]
        groups = groups[keep]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    rng = np.random.RandomState(SEED)
    keep = []
    for c in range(n_classes):
        idx_c = np.where(y_enc == c)[0]
        if len(idx_c) > 500:
            idx_c = rng.choice(idx_c, 500, replace=False)
        keep.extend(idx_c)
    keep = np.array(keep)
    X_sub = X[keep]
    y_sub = y_enc[keep]
    g_sub = groups[keep]
    gkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    accs = []
    for train_idx, test_idx in gkf.split(X_sub, y_sub, g_sub):
        scaler = StandardScaler().fit(X_sub[train_idx])
        clf = SGDClassifier(loss='log_loss', max_iter=500, random_state=SEED)
        clf.fit(scaler.transform(X_sub[train_idx]), y_sub[train_idx])
        accs.append(clf.score(scaler.transform(X_sub[test_idx]), y_sub[test_idx]))
    return np.mean(accs), 1.0 / n_classes


print('=== Virchow2 niche z-score test ===')
print('loading...')
X_raw, X_z, mc_labels, morph_labels, patients, subarrays = load_aligned()
print(f'\ntotal spots: {len(X_raw)}')

print('\n--- silhouette (cosine, vs MC) ---')
sil_r_mc = silhouette_subsampled(X_raw, mc_labels)
sil_z_mc = silhouette_subsampled(X_z, mc_labels)
print(f'  raw:        {sil_r_mc:+.4f}')
print(f'  z-scored:   {sil_z_mc:+.4f}')
print(f'  delta:      {sil_z_mc - sil_r_mc:+.4f}')

print('\n--- silhouette (cosine, vs morph 5-class) ---')
sil_r_m = silhouette_subsampled(X_raw, morph_labels)
sil_z_m = silhouette_subsampled(X_z, morph_labels)
print(f'  raw:        {sil_r_m:+.4f}')
print(f'  z-scored:   {sil_z_m:+.4f}')
print(f'  delta:      {sil_z_m - sil_r_m:+.4f}')

print('\n--- ARI (kmeans K=14 vs MC) ---')
ari_r = kmeans_ari(X_raw, mc_labels)
ari_z = kmeans_ari(X_z, mc_labels)
print(f'  raw:        {ari_r:.4f}')
print(f'  z-scored:   {ari_z:.4f}')

print('\n--- linear probe MC ---')
acc_r, ch = linear_probe(X_raw, mc_labels, patients)
acc_z, _ = linear_probe(X_z, mc_labels, patients)
print(f'  raw:        {acc_r*100:.1f}% ({acc_r/ch:.1f}x chance)')
print(f'  z-scored:   {acc_z*100:.1f}% ({acc_z/ch:.1f}x chance)')

print('\n--- linear probe morph ---')
acc_r_m, ch_m = linear_probe(X_raw, morph_labels, patients)
acc_z_m, _ = linear_probe(X_z, morph_labels, patients)
print(f'  raw:        {acc_r_m*100:.1f}% ({acc_r_m/ch_m:.1f}x chance)')
print(f'  z-scored:   {acc_z_m*100:.1f}% ({acc_z_m/ch_m:.1f}x chance)')

print('\n=== DECISION ===')
if sil_z_mc > 0:
    print('Z-SCORE WORKS: cross-subarray MC silhouette positive')
    print('  -> mixed-subarray batches viable')
elif sil_z_mc > sil_r_mc + 0.05:
    print('Z-SCORE PARTIAL: improvement but not fully positive')
    print('  -> per-subarray batches recommended (cleaner)')
else:
    print('Z-SCORE DOES NOT HELP MUCH:')
    print('  -> per-subarray batches required')