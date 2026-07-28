"""
diag_novae_zscore.py - test if z-score per-subarray fixes Novae cross-subarray geometry.

unfiltered Novae cross-subarray silhouette: -0.246 (negative, anti-structure)
hypothesis: this is just a per-subarray mean shift, not biology destruction
test: z-score each subarray independently, recompute cross-subarray silhouette

if z-scored sil > 0: mixed batches with z-score Novae work, keep it
if z-scored sil still <= 0: per-subarray batches, within-subarray evaluation
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
NOVAE_DIR = ROOT / 'data' / 'embeddings' / 'novae_all'
BIO_DIR = ROOT / 'data' / 'embeddings' / 'biological_signals'
SEED = 42


def load_aligned():
    """load all novae embeddings + MC labels + morphology labels with z-score variants."""
    mc = pd.read_csv(BIO_DIR / 'mc_labels.tsv', sep='\t')
    morph = pd.read_csv(BIO_DIR / 'morphology_labels.tsv', sep='\t')
    morph['short'] = morph['subarray_id'].str.replace(r'^TNBC\d+_', '', regex=True)

    novae_files = sorted(NOVAE_DIR.glob('*.npy'))
    novae_lookup = {}
    for f in novae_files:
        parts = f.stem.split('_', 1)
        if len(parts) == 2 and parts[0].startswith('TNBC'):
            short_id = parts[1]
            patient_num = int(parts[0].replace('TNBC', ''))
            novae_lookup[short_id] = (f, patient_num)

    raw_rows = []
    zscore_rows = []
    mc_labels = []
    morph_labels = []
    patients = []
    subarrays = []

    n_loaded = 0
    for subarray, (novae_path, patient_num) in novae_lookup.items():
        emb = np.load(novae_path)

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

        # z-score within this subarray
        emb_centered = emb - emb.mean(axis=0)
        emb_std = emb_centered.std(axis=0) + 1e-8
        emb_zscored = emb_centered / emb_std

        mc_sub = mc[mc['subarray'] == subarray].set_index('spot_id')
        morph_sub = morph[morph['short'] == subarray].set_index('spot_id')

        for i, sid in enumerate(spot_ids):
            mc_label = mc_sub.loc[sid, 'megacluster'] if sid in mc_sub.index else None
            m_label = morph_sub.loc[sid, 'dominant_5class'] if sid in morph_sub.index else None
            if mc_label is None and m_label is None:
                continue
            raw_rows.append(emb[i])
            zscore_rows.append(emb_zscored[i])
            mc_labels.append(mc_label)
            morph_labels.append(m_label)
            patients.append(patient_num)
            subarrays.append(subarray)

        n_loaded += 1
        if n_loaded % 50 == 0:
            print(f'  loaded {n_loaded}/{len(novae_lookup)} subarrays')

    return (np.array(raw_rows), np.array(zscore_rows),
            mc_labels, morph_labels, patients, subarrays)


def silhouette_subsampled(X, y, label, max_n=10000):
    """compute silhouette on subsample for speed."""
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

    sil = silhouette_score(X, y_enc, metric='cosine')
    return sil, len(le.classes_)


def kmeans_ari(X, y, n_clusters):
    """k-means + ARI."""
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


def linear_probe(X, y, groups, label):
    """3-fold patient-stratified probe."""
    if any(v is None for v in y):
        keep = [i for i, v in enumerate(y) if v is not None]
        X = X[keep]
        y = [y[i] for i in keep]
        groups = [groups[i] for i in keep]
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
    groups_sub = np.array([groups[i] for i in keep])

    gkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    accs = []
    for train_idx, test_idx in gkf.split(X_sub, y_sub, groups_sub):
        scaler = StandardScaler().fit(X_sub[train_idx])
        clf = SGDClassifier(loss='log_loss', max_iter=500, random_state=SEED)
        clf.fit(scaler.transform(X_sub[train_idx]), y_sub[train_idx])
        accs.append(clf.score(scaler.transform(X_sub[test_idx]), y_sub[test_idx]))
    return np.mean(accs), 1.0 / n_classes


print('=== Novae z-score test ===')
print('loading aligned embeddings + labels...')
X_raw, X_zscore, mc_labels, morph_labels, patients, subarrays = load_aligned()
print(f'\ntotal spots: {len(X_raw)}')

# silhouette + ARI
print('\n--- silhouette (cosine, vs MC labels) ---')
sil_raw_mc, n_mc = silhouette_subsampled(X_raw, mc_labels, 'MC')
sil_z_mc, _ = silhouette_subsampled(X_zscore, mc_labels, 'MC')
print(f'  raw novae:    {sil_raw_mc:+.4f}')
print(f'  z-scored:     {sil_z_mc:+.4f}')
print(f'  delta:        {sil_z_mc - sil_raw_mc:+.4f}')

print('\n--- silhouette (cosine, vs morphology 5-class) ---')
sil_raw_m, n_m = silhouette_subsampled(X_raw, morph_labels, 'morph')
sil_z_m, _ = silhouette_subsampled(X_zscore, morph_labels, 'morph')
print(f'  raw novae:    {sil_raw_m:+.4f}')
print(f'  z-scored:     {sil_z_m:+.4f}')
print(f'  delta:        {sil_z_m - sil_raw_m:+.4f}')

print('\n--- ARI (k-means K=14 vs MC) ---')
ari_raw_mc = kmeans_ari(X_raw, mc_labels, 14)
ari_z_mc = kmeans_ari(X_zscore, mc_labels, 14)
print(f'  raw novae:    {ari_raw_mc:.4f}')
print(f'  z-scored:     {ari_z_mc:.4f}')

print('\n--- linear probe MC 14-class ---')
acc_raw_mc, ch_mc = linear_probe(X_raw, mc_labels, patients, 'MC')
acc_z_mc, _ = linear_probe(X_zscore, mc_labels, patients, 'MC')
print(f'  raw novae:    {acc_raw_mc*100:.1f}% ({acc_raw_mc/ch_mc:.1f}x chance)')
print(f'  z-scored:     {acc_z_mc*100:.1f}% ({acc_z_mc/ch_mc:.1f}x chance)')

print('\n--- linear probe morphology 5-class ---')
acc_raw_m, ch_m = linear_probe(X_raw, morph_labels, patients, 'morph')
acc_z_m, _ = linear_probe(X_zscore, morph_labels, patients, 'morph')
print(f'  raw novae:    {acc_raw_m*100:.1f}% ({acc_raw_m/ch_m:.1f}x chance)')
print(f'  z-scored:     {acc_z_m*100:.1f}% ({acc_z_m/ch_m:.1f}x chance)')

print('\n=== DECISION ===')
if sil_z_mc > 0 and sil_z_m > 0:
    print('Z-SCORE WORKS: cross-subarray silhouette positive on both labels')
    print('  → use mixed-subarray batches with z-score Novae')
    print('  → keep Novae in rich ST representation')
elif sil_z_mc > sil_raw_mc + 0.05 or sil_z_m > sil_raw_m + 0.05:
    print('Z-SCORE PARTIAL: improvement but not fully positive')
    print('  → option 1: per-subarray batches, within-subarray eval')
    print('  → option 2: try Harmony as second pass')
else:
    print('Z-SCORE DOES NOT HELP:')
    print('  → use per-subarray batches')
    print('  → within-subarray evaluation against baselines')
    print('  → drop Novae from cross-subarray representation, keep for within-subarray')