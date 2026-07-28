"""
validate_novae_qc.py - QC validation for Novae embeddings against biological labels.

asks: do Novae 64-d embeddings encode meaningful biology on this platform?
or are they noise from training-data mismatch (Novae trained on Xenium, applied to 100um spots)?

four tests:
  1. linear probe Novae -> morphology 5-class (cross-patient CV)
  2. linear probe Novae -> MC 14-class (cross-patient CV)
  3. silhouette score on MC labels
  4. ARI between Novae K-means clusters and MC labels

decision gate:
  if linear probe accuracy > 2x chance for both tasks -> Novae passes QC, use as ST encoder
  if accuracy near chance -> Novae embeddings are noise, replace with mc_weights as ST representation

usage:
    python scripts/validate_novae_qc.py
"""

import warnings
warnings.filterwarnings('ignore')

import subprocess
import io
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, silhouette_score, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parent.parent
NOVAE_DIR = ROOT / 'data' / 'embeddings' / 'novae_all'
BIO_DIR = ROOT / 'data' / 'embeddings' / 'biological_signals'
SEED = 42


def load_novae_with_labels():
    """build master table: subarray, spot_id, patient_id, novae_emb (64d), mc_label, morph_label."""
    # load labels
    mc = pd.read_csv(BIO_DIR / 'mc_labels.tsv', sep='\t')
    morph = pd.read_csv(BIO_DIR / 'morphology_labels.tsv', sep='\t')
    morph = morph.rename(columns={'subarray_id': 'full_subarray_id'})
    morph['subarray'] = morph['full_subarray_id'].str.replace(r'^TNBC\d+_', '', regex=True)

    print(f'mc_labels: {len(mc)} spots, {mc["subarray"].nunique()} subarrays')
    print(f'morphology: {len(morph)} spots, {morph["subarray"].nunique()} subarrays')

    # find novae files matching mc_labels subarrays
    # novae uses TNBC{N}_CN{X}_{Y} format
    novae_files = sorted(NOVAE_DIR.glob('*.npy'))
    novae_lookup = {}
    for f in novae_files:
        # strip TNBC{N}_ prefix to get CN{X}_{Y}
        parts = f.stem.split('_', 1)
        if len(parts) == 2 and parts[0].startswith('TNBC'):
            short_id = parts[1]
            patient_num = int(parts[0].replace('TNBC', ''))
            novae_lookup[short_id] = (f, patient_num)
    print(f'novae files matched to short id: {len(novae_lookup)}')

    # build master alignment
    rows_emb = []
    rows_mc = []
    rows_morph = []
    rows_meta = []

    n_loaded = 0
    for subarray, (novae_path, patient_num) in novae_lookup.items():
        emb = np.load(novae_path)  # (n_spots, 64)

        # load spot ids in same order via selection.RData
        slide = subarray.split('_')[0]  # CN5
        pos = subarray.split('_')[1]    # D2
        rdata_path = ROOT / 'data' / 'inputs' / 'byArray' / slide / pos / 'selection.RData'
        if not rdata_path.exists():
            continue
        r = subprocess.run(['Rscript', '-e',
                            f'load("{rdata_path}"); cat(rownames(cnts), sep="\\n")'],
                           capture_output=True, text=True)
        spot_ids = [s for s in r.stdout.strip().split('\n') if s]
        if len(spot_ids) != emb.shape[0]:
            continue

        # build per-spot alignment
        mc_sub = mc[mc['subarray'] == subarray].set_index('spot_id')
        morph_sub = morph[morph['subarray'] == subarray].set_index('spot_id')

        for i, sid in enumerate(spot_ids):
            mc_label = mc_sub.loc[sid, 'megacluster'] if sid in mc_sub.index else None
            morph_label = morph_sub.loc[sid, 'dominant_5class'] if sid in morph_sub.index else None

            if mc_label is not None or morph_label is not None:
                rows_emb.append(emb[i])
                rows_mc.append(mc_label)
                rows_morph.append(morph_label)
                rows_meta.append((subarray, sid, patient_num))

        n_loaded += 1
        if n_loaded % 25 == 0:
            print(f'  loaded {n_loaded}/{len(novae_lookup)} subarrays')

    X = np.array(rows_emb)
    print(f'\ntotal aligned spots: {len(X)}')
    return X, rows_mc, rows_morph, rows_meta


def linear_probe(X, y, groups, label_name, n_classes):
    """patient-stratified group K-fold linear probe."""
    if y is None or any(v is None for v in y):
        keep = [i for i, v in enumerate(y) if v is not None]
        X = X[keep]
        y = [y[i] for i in keep]
        groups = [groups[i] for i in keep]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes_actual = len(le.classes_)
    chance = 1.0 / n_classes_actual

    print(f'\n--- {label_name} probe ---')
    print(f'  n_samples: {len(X)}, n_classes: {n_classes_actual}, chance: {chance*100:.1f}%')
    print(f'  class counts: {dict(zip(le.classes_, np.bincount(y_enc)))}')

    # subsample to 200 per class for speed
    rng = np.random.RandomState(SEED)
    keep = []
    for c in range(n_classes_actual):
        idx_c = np.where(y_enc == c)[0]
        if len(idx_c) > 500:
            idx_c = rng.choice(idx_c, 500, replace=False)
        keep.extend(idx_c)
    keep = np.array(keep)
    X_sub = X[keep]
    y_sub = y_enc[keep]
    groups_sub = np.array([groups[i] for i in keep])

    # group K-fold by patient
    gkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    accs = []
    f1s = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_sub, y_sub, groups_sub)):
        scaler = StandardScaler().fit(X_sub[train_idx])
        clf = SGDClassifier(loss='log_loss', max_iter=500, random_state=SEED)
        clf.fit(scaler.transform(X_sub[train_idx]), y_sub[train_idx])
        pred = clf.predict(scaler.transform(X_sub[test_idx]))
        acc = (pred == y_sub[test_idx]).mean()
        f1 = f1_score(y_sub[test_idx], pred, average='macro')
        accs.append(acc)
        f1s.append(f1)
        n_train_pat = len(set(groups_sub[train_idx]))
        n_test_pat = len(set(groups_sub[test_idx]))
        print(f'  fold {fold+1}: acc={acc*100:.1f}%, macro_f1={f1:.3f} '
              f'({n_train_pat} train pat / {n_test_pat} test pat)')

    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1s)
    print(f'  mean: {mean_acc*100:.1f}% accuracy, {mean_f1:.3f} macro-F1, '
          f'{mean_acc/chance:.1f}x chance')
    return mean_acc, mean_f1, chance


def cluster_metrics(X, y, label_name):
    """K-means clustering vs ground truth: silhouette + ARI."""
    if any(v is None for v in y):
        keep = [i for i, v in enumerate(y) if v is not None]
        X = X[keep]
        y = [y[i] for i in keep]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    # subsample for silhouette speed
    rng = np.random.RandomState(SEED)
    if len(X) > 10000:
        idx = rng.choice(len(X), 10000, replace=False)
        X_sub, y_sub = X[idx], y_enc[idx]
    else:
        X_sub, y_sub = X, y_enc

    # silhouette on Novae embeddings vs ground truth labels
    sil = silhouette_score(X_sub, y_sub, metric='cosine')

    # K-means on Novae with same K, ARI vs ground truth
    km = MiniBatchKMeans(n_clusters=n_classes, random_state=SEED, n_init=3, batch_size=1024)
    pred = km.fit_predict(X)
    ari = adjusted_rand_score(y_enc, pred)

    print(f'\n--- {label_name} cluster metrics ---')
    print(f'  silhouette (cosine, vs {n_classes} ground truth classes): {sil:.4f}')
    print(f'    interpretation: > 0.1 = some structure, > 0.25 = clear, > 0.5 = strong')
    print(f'  ARI (Novae K={n_classes} vs ground truth): {ari:.4f}')
    print(f'    interpretation: 0 = random, > 0.1 = weak signal, > 0.3 = clear')

    return sil, ari


if __name__ == '__main__':
    print('=== Novae QC validation ===')
    X, mc_labels, morph_labels, meta = load_novae_with_labels()
    patients = [m[2] for m in meta]

    # 1. linear probe MC 14-class
    mc_acc, mc_f1, mc_chance = linear_probe(X, mc_labels, patients, 'MC 14-class', 14)

    # 2. linear probe morphology 5-class
    morph_acc, morph_f1, morph_chance = linear_probe(X, morph_labels, patients, 'morphology 5-class', 5)

    # 3. cluster metrics
    mc_sil, mc_ari = cluster_metrics(X, mc_labels, 'MC 14-class')
    morph_sil, morph_ari = cluster_metrics(X, morph_labels, 'morphology 5-class')

    # decision
    print('\n=== DECISION GATE ===')
    print(f'MC linear probe: {mc_acc*100:.1f}% ({mc_acc/mc_chance:.1f}x chance)')
    print(f'morphology linear probe: {morph_acc*100:.1f}% ({morph_acc/morph_chance:.1f}x chance)')
    print(f'MC silhouette: {mc_sil:.4f}, ARI: {mc_ari:.4f}')
    print(f'morphology silhouette: {morph_sil:.4f}, ARI: {morph_ari:.4f}')
    print()

    mc_pass = mc_acc > 2 * mc_chance
    morph_pass = morph_acc > 2 * morph_chance

    if mc_pass and morph_pass:
        print('VERDICT: Novae PASSES - encodes meaningful biology, use as ST encoder')
    elif mc_pass or morph_pass:
        print('VERDICT: Novae MARGINAL - some signal, consider supplementing with mc_weights')
    else:
        print('VERDICT: Novae FAILS - embeddings are noise on this platform')
        print('         REPLACE with mc_weights (14-d soft NMF) as primary ST representation')