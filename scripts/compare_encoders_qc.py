"""
compare_encoders_qc.py - QC comparison of UNI2 raw, Virchow2 niche, Novae.

runs same metrics on all three encoders for direct comparison:
  - within-subarray silhouette (vs MC, vs morphology)
  - cross-subarray silhouette (vs MC, vs morphology)
  - linear probe MC (14-class) and morph (5-class), patient-stratified CV
  - batch effect ratio (inter-patient / intra-patient cosine distance)

output: comparison table + decision gates for rank 1a encoder choice
"""

import warnings
warnings.filterwarnings('ignore')

import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / 'data' / 'embeddings'
BIO_DIR = EMB_DIR / 'biological_signals'
SEED = 42

ENCODERS = {
    'uni2_raw': 'uni2_raw',
    'virchow2_niche': 'virchow2_niche',
    'novae': 'novae_all',
}


def load_encoder_with_labels(encoder_dir_name):
    """load all embeddings + MC labels + morph labels for one encoder."""
    enc_dir = EMB_DIR / encoder_dir_name
    if not enc_dir.exists():
        return None

    mc = pd.read_csv(BIO_DIR / 'mc_labels.tsv', sep='\t')
    morph = pd.read_csv(BIO_DIR / 'morphology_labels.tsv', sep='\t')
    morph['short'] = morph['subarray_id'].str.replace(r'^TNBC\d+_', '', regex=True)

    # build short_id -> file lookup (handles both TNBC{N}_CN{X}_{Y} and CN{X}_{Y} naming)
    files = sorted(enc_dir.glob('*.npy'))
    files = [f for f in files if not f.stem.endswith('_meta')]
    lookup = {}
    for f in files:
        parts = f.stem.split('_', 1)
        if len(parts) == 2 and parts[0].startswith('TNBC'):
            short_id = parts[1]
            patient = int(parts[0].replace('TNBC', ''))
            lookup[short_id] = (f, patient)

    rows_emb = []
    mc_labels = []
    morph_labels = []
    patients = []
    subarrays = []

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

        mc_sub = mc[mc['subarray'] == subarray].set_index('spot_id')
        morph_sub = morph[morph['short'] == subarray].set_index('spot_id')

        for i, sid in enumerate(spot_ids):
            mc_label = mc_sub.loc[sid, 'megacluster'] if sid in mc_sub.index else None
            m_label = morph_sub.loc[sid, 'dominant_5class'] if sid in morph_sub.index else None
            if mc_label is None and m_label is None:
                continue
            rows_emb.append(emb[i])
            mc_labels.append(mc_label)
            morph_labels.append(m_label)
            patients.append(patient)
            subarrays.append(subarray)
        n_loaded += 1
        if n_loaded % 50 == 0:
            print(f'    {n_loaded}/{len(lookup)}')

    return (np.array(rows_emb), mc_labels, morph_labels,
            np.array(patients), np.array(subarrays))


def cross_subarray_silhouette(X, labels, max_n=10000):
    if any(v is None for v in labels):
        keep = [i for i, v in enumerate(labels) if v is not None]
        X = X[keep]
        labels = [labels[i] for i in keep]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    rng = np.random.RandomState(SEED)
    if len(X) > max_n:
        idx = rng.choice(len(X), max_n, replace=False)
        X = X[idx]
        y = y[idx]
    return silhouette_score(X, y, metric='cosine')


def within_subarray_silhouette(X, labels, subarrays, max_subarrays=30):
    """compute silhouette within each subarray, average."""
    if any(v is None for v in labels):
        keep = [i for i, v in enumerate(labels) if v is not None]
        X = X[keep]
        labels = [labels[i] for i in keep]
        subarrays = subarrays[keep]
    labels = np.array(labels)

    unique_subs = np.unique(subarrays)
    rng = np.random.RandomState(SEED)
    if len(unique_subs) > max_subarrays:
        unique_subs = rng.choice(unique_subs, max_subarrays, replace=False)

    sils = []
    for sub in unique_subs:
        mask = subarrays == sub
        X_sub = X[mask]
        y_sub = labels[mask]
        if len(set(y_sub)) < 2 or len(y_sub) < 10:
            continue
        try:
            sil = silhouette_score(X_sub, y_sub, metric='cosine')
            sils.append(sil)
        except Exception:
            continue
    return np.mean(sils) if sils else float('nan')


def linear_probe_3fold(X, labels, patients):
    if any(v is None for v in labels):
        keep = [i for i, v in enumerate(labels) if v is not None]
        X = X[keep]
        labels = [labels[i] for i in keep]
        patients = patients[keep]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    rng = np.random.RandomState(SEED)
    keep = []
    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        if len(idx_c) > 500:
            idx_c = rng.choice(idx_c, 500, replace=False)
        keep.extend(idx_c)
    keep = np.array(keep)
    X_sub = X[keep]
    y_sub = y[keep]
    g_sub = patients[keep]

    gkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    accs = []
    for train_idx, test_idx in gkf.split(X_sub, y_sub, g_sub):
        scaler = StandardScaler().fit(X_sub[train_idx])
        clf = SGDClassifier(loss='log_loss', max_iter=500, random_state=SEED)
        clf.fit(scaler.transform(X_sub[train_idx]), y_sub[train_idx])
        accs.append(clf.score(scaler.transform(X_sub[test_idx]), y_sub[test_idx]))
    return np.mean(accs), 1.0 / n_classes


def batch_effect_ratio(X, patients):
    """inter/intra patient cosine distance ratio."""
    centroids = {}
    for pid in np.unique(patients):
        mask = patients == pid
        centroids[pid] = X[mask].mean(axis=0)
    pids = sorted(centroids.keys())
    cmat = np.stack([centroids[p] for p in pids])
    inter = pdist(cmat, 'cosine').mean()

    intras = []
    for pid in np.unique(patients):
        mask = patients == pid
        Xp = X[mask]
        c = Xp.mean(axis=0)
        norms = np.linalg.norm(Xp, axis=1) * np.linalg.norm(c) + 1e-12
        cos = 1 - (Xp @ c) / norms
        intras.append(cos.mean())
    return inter / np.mean(intras)


# main loop
results = {}
for name, dirname in ENCODERS.items():
    print(f'\n=== {name} ===')
    print('  loading...')
    out = load_encoder_with_labels(dirname)
    if out is None:
        print('  SKIP - directory not found')
        continue
    X, mc_labels, morph_labels, patients, subarrays = out
    print(f'  total spots: {len(X)}, dim: {X.shape[1]}, patients: {len(set(patients))}')

    print('  cross-subarray silhouette...')
    cs_mc = cross_subarray_silhouette(X, mc_labels)
    cs_m = cross_subarray_silhouette(X, morph_labels)

    print('  within-subarray silhouette...')
    ws_mc = within_subarray_silhouette(X, mc_labels, subarrays)
    ws_m = within_subarray_silhouette(X, morph_labels, subarrays)

    print('  linear probe MC...')
    lp_mc, ch_mc = linear_probe_3fold(X, mc_labels, patients)
    print('  linear probe morph...')
    lp_m, ch_m = linear_probe_3fold(X, morph_labels, patients)

    print('  batch effect ratio...')
    ber = batch_effect_ratio(X, patients)

    results[name] = {
        'dim': X.shape[1],
        'n_spots': len(X),
        'within_sil_mc': ws_mc,
        'within_sil_morph': ws_m,
        'cross_sil_mc': cs_mc,
        'cross_sil_morph': cs_m,
        'lp_mc': lp_mc,
        'lp_mc_chance': ch_mc,
        'lp_morph': lp_m,
        'lp_morph_chance': ch_m,
        'batch_ratio': ber,
    }

# print comparison table
print('\n\n=== comparison table ===\n')
metrics = [
    ('dim', 'dim', '{:d}'),
    ('n_spots', 'n spots', '{:,}'),
    ('within_sil_mc', 'within-sub sil (MC)', '{:+.4f}'),
    ('cross_sil_mc', 'cross-sub sil (MC)', '{:+.4f}'),
    ('within_sil_morph', 'within-sub sil (morph)', '{:+.4f}'),
    ('cross_sil_morph', 'cross-sub sil (morph)', '{:+.4f}'),
    ('lp_mc', 'lin probe MC', '{:.1%}'),
    ('lp_morph', 'lin probe morph', '{:.1%}'),
    ('batch_ratio', 'batch ratio', '{:.3f}'),
]

names = list(results.keys())
header = f'{"metric":<25}' + ''.join(f'{n:>18}' for n in names)
print(header)
print('-' * len(header))
for key, label, fmt in metrics:
    row = f'{label:<25}'
    for n in names:
        v = results[n].get(key)
        if v is None:
            row += f'{"N/A":>18}'
        elif key.startswith('lp_'):
            row += f'{fmt.format(v):>18}'
        else:
            row += f'{fmt.format(v):>18}'
    print(row)

# decision gates
print('\n=== decision gates ===')
v2 = results.get('virchow2_niche', {})
u = results.get('uni2_raw', {})
n = results.get('novae', {})

if v2 and u:
    if v2.get('lp_mc', 0) > u.get('lp_mc', 0):
        print('Virchow2 niche > UNI2 raw on MC linear probe -> primary H&E encoder')
    else:
        print('Virchow2 niche <= UNI2 raw -> investigate')

    if v2.get('cross_sil_mc', -1) > 0:
        print('Virchow2 cross-subarray sil > 0 -> mixed-subarray batches OK')
    else:
        print(f'Virchow2 cross-subarray sil = {v2.get("cross_sil_mc"):.4f} -> use per-subarray batches')

# save
import json
with open(BIO_DIR / 'encoder_qc_comparison.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\nsaved to {BIO_DIR}/encoder_qc_comparison.json')