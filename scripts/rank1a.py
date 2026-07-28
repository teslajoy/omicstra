"""
rank1a.py - validation gate before alignment training.

linear probe: Virchow2 niche embeddings (1280-d) -> MC label (14-class)
split: patient-stratified 5-fold cross validation
metric: macro-F1, accuracy, per-class F1, confusion matrix

decision gate:
  macro-F1 > 0.3 -> proceed to alignment training
  macro-F1 < 0.15 -> H&E encoder doesn't capture MC structure, investigate

answers: can H&E morphology predict cell program identity?
         is the biological signal alignment will learn already in Virchow2?

usage:
    python scripts/rank1a.py
"""

import warnings
warnings.filterwarnings('ignore')

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
V2_DIR = ROOT / 'data' / 'embeddings' / 'virchow2_niche'
BIO_DIR = ROOT / 'data' / 'embeddings' / 'biological_signals'
RUNS_DIR = ROOT / 'runs' / 'tnbc-92' / 'rank1a'
SEED = 42
N_FOLDS = 5
MAX_PER_CLASS = 2000  # cap per-class for tractable training time

print('=== rank1a: Virchow2 niche -> MC 14-class linear probe ===\n')

# load aligned data
print('loading Virchow2 niche + MC labels...')
mc = pd.read_csv(BIO_DIR / 'mc_labels.tsv', sep='\t')
files = sorted(V2_DIR.glob('*.npy'))
files = [f for f in files if not f.stem.endswith('_meta')]

lookup = {}
for f in files:
    parts = f.stem.split('_', 1)
    if len(parts) == 2 and parts[0].startswith('TNBC'):
        short_id = parts[1]
        patient = int(parts[0].replace('TNBC', ''))
        lookup[short_id] = (f, patient)
print(f'  Virchow2 subarrays: {len(lookup)}')

X_rows = []
y_rows = []
patient_rows = []
subarray_rows = []

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
    for i, sid in enumerate(spot_ids):
        if sid in mc_sub.index:
            X_rows.append(emb[i])
            y_rows.append(int(mc_sub.loc[sid, 'megacluster']))
            patient_rows.append(patient)
            subarray_rows.append(subarray)
    n_loaded += 1
    if n_loaded % 50 == 0:
        print(f'  loaded {n_loaded}/{len(lookup)}')

X = np.array(X_rows, dtype=np.float32)
y = np.array(y_rows)
patients = np.array(patient_rows)
subarrays = np.array(subarray_rows)
print(f'\ntotal: {len(X)} spots, {X.shape[1]}d, {len(np.unique(patients))} patients')
print(f'class distribution:')
for c in sorted(np.unique(y)):
    print(f'  MC{c:2d}: {(y == c).sum():6d} ({(y == c).mean()*100:.1f}%)')

# stratified subsample to MAX_PER_CLASS for tractable training
rng = np.random.RandomState(SEED)
keep = []
for c in np.unique(y):
    idx_c = np.where(y == c)[0]
    if len(idx_c) > MAX_PER_CLASS:
        idx_c = rng.choice(idx_c, MAX_PER_CLASS, replace=False)
    keep.extend(idx_c)
keep = np.array(keep)
X_use = X[keep]
y_use = y[keep]
patients_use = patients[keep]
subarrays_use = subarrays[keep]
print(f'\nstratified subsample (max {MAX_PER_CLASS}/class): {len(X_use)} spots')

n_classes = len(np.unique(y_use))
chance = 1.0 / n_classes
print(f'\n--- {N_FOLDS}-fold patient-stratified CV ---')
print(f'chance: {chance*100:.1f}% (1/{n_classes})')
print(f'target: macro-F1 > 0.30 (proceed) | < 0.15 (investigate)\n')

gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results = []
all_pred = np.zeros_like(y_use)
all_true = y_use.copy()

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_use, y_use, patients_use)):
    n_train_pat = len(set(patients_use[train_idx]))
    n_test_pat = len(set(patients_use[test_idx]))

    scaler = StandardScaler().fit(X_use[train_idx])
    X_tr = scaler.transform(X_use[train_idx])
    X_te = scaler.transform(X_use[test_idx])

    clf = LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs', random_state=SEED
    )
    clf.fit(X_tr, y_use[train_idx])
    pred = clf.predict(X_te)
    all_pred[test_idx] = pred

    acc = accuracy_score(y_use[test_idx], pred)
    macro_f1 = f1_score(y_use[test_idx], pred, average='macro')
    weighted_f1 = f1_score(y_use[test_idx], pred, average='weighted')

    fold_results.append({
        'fold': fold + 1,
        'n_train_pat': n_train_pat,
        'n_test_pat': n_test_pat,
        'n_train_spots': len(train_idx),
        'n_test_spots': len(test_idx),
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
    })
    print(f'fold {fold+1}: acc={acc*100:.1f}%, macro-F1={macro_f1:.3f}, '
          f'weighted-F1={weighted_f1:.3f}  ({n_train_pat} train pat / {n_test_pat} test pat)')

# aggregate
mean_acc = np.mean([f['accuracy'] for f in fold_results])
mean_macro_f1 = np.mean([f['macro_f1'] for f in fold_results])
std_macro_f1 = np.std([f['macro_f1'] for f in fold_results])
mean_weighted_f1 = np.mean([f['weighted_f1'] for f in fold_results])

print(f'\n=== aggregate ({N_FOLDS}-fold mean) ===')
print(f'accuracy:    {mean_acc*100:.1f}% ({mean_acc/chance:.1f}x chance)')
print(f'macro F1:    {mean_macro_f1:.3f} +/- {std_macro_f1:.3f}')
print(f'weighted F1: {mean_weighted_f1:.3f}')

# per-class report (across all folds via concatenated predictions)
print(f'\n--- per-class report (all folds combined) ---')
report = classification_report(all_true, all_pred,
                                labels=sorted(np.unique(y_use)),
                                target_names=[f'MC{c}' for c in sorted(np.unique(y_use))],
                                digits=3, zero_division=0)
print(report)

cm = confusion_matrix(all_true, all_pred, labels=sorted(np.unique(y_use)))
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

# decision gate
print('\n=== DECISION GATE ===')
if mean_macro_f1 > 0.30:
    verdict = 'PASS'
    decision = 'proceed to alignment training'
elif mean_macro_f1 > 0.15:
    verdict = 'MARGINAL'
    decision = 'signal exists but weak - alignment may struggle, proceed with caution'
else:
    verdict = 'FAIL'
    decision = 'H&E encoder does not capture MC structure - investigate before alignment'

print(f'verdict: {verdict}')
print(f'macro-F1: {mean_macro_f1:.3f} (target > 0.30)')
print(f'decision: {decision}')

# save results
RUNS_DIR.mkdir(parents=True, exist_ok=True)
results = {
    'experiment': 'rank1a',
    'task': 'linear probe Virchow2 niche -> MC 14-class',
    'encoder': 'virchow2_niche',
    'encoder_dim': int(X.shape[1]),
    'target': 'megacluster',
    'n_classes': int(n_classes),
    'chance': float(chance),
    'n_spots_total': int(len(X)),
    'n_spots_used': int(len(X_use)),
    'n_patients': int(len(np.unique(patients_use))),
    'split': f'{N_FOLDS}-fold patient-stratified',
    'preprocessing': 'StandardScaler per fold',
    'classifier': 'LogisticRegression(C=1.0, multinomial)',
    'fold_results': fold_results,
    'aggregate': {
        'accuracy': float(mean_acc),
        'macro_f1': float(mean_macro_f1),
        'macro_f1_std': float(std_macro_f1),
        'weighted_f1': float(mean_weighted_f1),
        'amplification_over_chance': float(mean_acc / chance),
    },
    'verdict': verdict,
    'decision': decision,
    'class_distribution': {f'MC{int(c)}': int((y == c).sum()) for c in sorted(np.unique(y))},
    'confusion_matrix_normalized': cm_norm.tolist(),
}
with open(RUNS_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)
np.save(RUNS_DIR / 'confusion_matrix.npy', cm)
np.save(RUNS_DIR / 'predictions.npy', all_pred)
np.save(RUNS_DIR / 'true_labels.npy', all_true)

print(f'\nsaved to runs/tnbc-92/rank1a/')
print(f'  results.json')
print(f'  confusion_matrix.npy')
print(f'  predictions.npy')
print(f'  true_labels.npy')