"""
run_alignment_90p.py - 90-patient cross-modal alignment experiment.

uses ComBat-corrected UNI2 + Novae embeddings, patient-held-out split,
late interaction MLP -> shared 512d -> InfoNCE.

answers H1: do aligned embeddings retrieve better cross-modal matches
than baselines, with cross-patient generalization?

usage:
    python scripts/run_alignment_90p.py
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# --- config ---

EXP_ID = 'exp02_90p'
SEED = 42

CONFIG = {
    # encoders
    'he_encoder': 'uni2_reinhard_combat_all',
    'st_encoder': 'novae_all',
    'he_dim': 1536,
    'st_dim': 64,

    # patient split
    'n_train_patients': 70,
    'n_val_patients': 10,
    'n_test_patients': 10,  # actually 9 since 89 total

    # architecture
    'shared_dim': 512,
    'dropout': 0.1,

    # training
    'batch_size': 1024,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 100,
    'temperature': 0.07,
    'patience': 15,
    'warmup_epochs': 5,
    'early_stopping_metric': 'val_loss',
    'early_stopping_mode': 'min',
    'seed': SEED,
}

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / 'data' / 'embeddings'
RUNS_DIR = ROOT / 'runs' / 'tnbc-92'

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(f'experiment: {EXP_ID}')
print(f'device: {device}')
for k, v in CONFIG.items():
    print(f'  {k}: {v}')

# --- data loading ---

he_dir = EMB_DIR / CONFIG['he_encoder']
st_dir = EMB_DIR / CONFIG['st_encoder']

# find matched subarrays
matched_keys = sorted([f.stem for f in he_dir.glob('*.npy')
                       if (st_dir / f.name).exists()])
print(f'\nmatched subarrays: {len(matched_keys)}')

# build patient -> subarrays mapping
patient_subarrays = defaultdict(list)
for key in matched_keys:
    patient = key.split('_')[0]
    patient_subarrays[patient].append(key)

patients = sorted(patient_subarrays.keys())
n_patients = len(patients)
print(f'patients: {n_patients}')

# stratified split by subarray count per patient
sa_counts = np.array([len(patient_subarrays[p]) for p in patients])
# sort patients by subarray count for stratified splitting
sorted_idx = np.argsort(sa_counts)
# shuffle within count groups
rng = np.random.RandomState(SEED)
count_groups = defaultdict(list)
for i in sorted_idx:
    count_groups[sa_counts[i]].append(i)
shuffled_idx = []
for count in sorted(count_groups.keys()):
    group = count_groups[count]
    rng.shuffle(group)
    shuffled_idx.extend(group)
shuffled_idx = np.array(shuffled_idx)

# interleave assignment: train/val/test spread across count strata
n_train_p = CONFIG['n_train_patients']
n_val_p = CONFIG['n_val_patients']
n_test_p = min(CONFIG['n_test_patients'], n_patients - n_train_p - n_val_p)

# simple stratified: cycle through sorted patients
assignments = []
for i, pi in enumerate(shuffled_idx):
    mod = i % (n_train_p + n_val_p + n_test_p)
    if mod < n_test_p:
        assignments.append(('test', pi))
    elif mod < n_test_p + n_val_p:
        assignments.append(('val', pi))
    else:
        assignments.append(('train', pi))

train_patients = [patients[pi] for split, pi in assignments if split == 'train']
val_patients = [patients[pi] for split, pi in assignments if split == 'val']
test_patients = [patients[pi] for split, pi in assignments if split == 'test']

print(f'\nsplit: {len(train_patients)} train / {len(val_patients)} val / {len(test_patients)} test patients')

# load embeddings per split
def load_split(patient_list):
    he_list, st_list, keys_list, pids_list = [], [], [], []
    for pid in patient_list:
        for key in patient_subarrays[pid]:
            he = np.load(he_dir / f'{key}.npy')
            st = np.load(st_dir / f'{key}.npy')
            assert he.shape[0] == st.shape[0], f'{key}: {he.shape[0]} vs {st.shape[0]}'
            he_list.append(he)
            st_list.append(st)
            keys_list.extend([key] * he.shape[0])
            pids_list.extend([pid] * he.shape[0])
    return (np.vstack(he_list), np.vstack(st_list),
            np.array(keys_list), np.array(pids_list))

he_train, st_train, keys_train, pids_train = load_split(train_patients)
he_val, st_val, keys_val, pids_val = load_split(val_patients)
he_test, st_test, keys_test, pids_test = load_split(test_patients)

print(f'train: {he_train.shape[0]} spots from {len(train_patients)} patients')
print(f'val:   {he_val.shape[0]} spots from {len(val_patients)} patients')
print(f'test:  {he_test.shape[0]} spots from {len(test_patients)} patients')

CONFIG['n_train'] = he_train.shape[0]
CONFIG['n_val'] = he_val.shape[0]
CONFIG['n_test'] = he_test.shape[0]
CONFIG['train_patients'] = train_patients
CONFIG['val_patients'] = val_patients
CONFIG['test_patients'] = test_patients

# --- model ---

class AlignmentMLP(nn.Module):
    """independent MLP projections for cross-modal alignment (late interaction)."""

    def __init__(self, he_dim, st_dim, shared_dim, dropout=0.0):
        super().__init__()
        self.he_proj = nn.Sequential(
            nn.Linear(he_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
        )
        self.st_proj = nn.Sequential(
            nn.LayerNorm(st_dim),
            nn.Linear(st_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
        )

    def forward(self, he, st):
        he_emb = F.normalize(self.he_proj(he), dim=-1)
        st_emb = F.normalize(self.st_proj(st), dim=-1)
        return he_emb, st_emb


def info_nce_loss(he_emb, st_emb, temperature):
    """symmetric InfoNCE loss."""
    logits = he_emb @ st_emb.T / temperature
    labels = torch.arange(len(he_emb), device=he_emb.device)
    loss_he2st = F.cross_entropy(logits, labels)
    loss_st2he = F.cross_entropy(logits.T, labels)
    return (loss_he2st + loss_st2he) / 2


model = AlignmentMLP(
    he_dim=CONFIG['he_dim'], st_dim=CONFIG['st_dim'],
    shared_dim=CONFIG['shared_dim'], dropout=CONFIG['dropout']
).to(device)

n_params = sum(p.numel() for p in model.parameters())
CONFIG['n_params'] = n_params
print(f'\nalignment MLP: {n_params:,} params')
print(f'  param-to-sample ratio: 1:{CONFIG["n_train"] // n_params}')

# --- dataset + loaders ---

class PairedEmbeddingDataset(Dataset):
    def __init__(self, he_emb, st_emb):
        self.he = torch.tensor(he_emb, dtype=torch.float32)
        self.st = torch.tensor(st_emb, dtype=torch.float32)

    def __len__(self):
        return len(self.he)

    def __getitem__(self, idx):
        return self.he[idx], self.st[idx]


train_ds = PairedEmbeddingDataset(he_train, st_train)
val_ds = PairedEmbeddingDataset(he_val, st_val)

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                          drop_last=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                        num_workers=0)

print(f'train batches: {len(train_loader)} (batch_size={CONFIG["batch_size"]})')

# --- training ---

@torch.no_grad()
def compute_aligned_embeddings(model, he_np, st_np, batch_size=512):
    model.eval()
    he_aligned, st_aligned = [], []
    for i in range(0, len(he_np), batch_size):
        he_batch = torch.tensor(he_np[i:i+batch_size], dtype=torch.float32, device=device)
        st_batch = torch.tensor(st_np[i:i+batch_size], dtype=torch.float32, device=device)
        he_emb, st_emb = model(he_batch, st_batch)
        he_aligned.append(he_emb.cpu().numpy())
        st_aligned.append(st_emb.cpu().numpy())
    return np.vstack(he_aligned), np.vstack(st_aligned)


optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                              weight_decay=CONFIG['weight_decay'])

warmup = CONFIG['warmup_epochs']
def lr_lambda(epoch):
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(CONFIG['epochs'] - warmup, 1)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

history = {'train_loss': [], 'val_loss': [], 'lr': []}
best_metric = float('inf')
best_state = None
patience_counter = 0

print('\n--- training ---')
t0 = time.time()
for epoch in range(CONFIG['epochs']):
    model.train()
    epoch_loss = 0
    for he_batch, st_batch in train_loader:
        he_batch, st_batch = he_batch.to(device), st_batch.to(device)
        he_emb, st_emb = model(he_batch, st_batch)
        loss = info_nce_loss(he_emb, st_emb, CONFIG['temperature'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for he_batch, st_batch in val_loader:
            he_batch, st_batch = he_batch.to(device), st_batch.to(device)
            he_emb, st_emb = model(he_batch, st_batch)
            loss = info_nce_loss(he_emb, st_emb, CONFIG['temperature'])
            val_loss += loss.item()
    val_loss /= len(val_loader)

    scheduler.step()
    lr = scheduler.get_last_lr()[0]

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['lr'].append(lr)

    improved = val_loss < best_metric
    if improved:
        best_metric = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        marker = ' *'
    else:
        patience_counter += 1
        marker = ''

    if (epoch + 1) % 5 == 0 or epoch == 0 or patience_counter == 0:
        elapsed = time.time() - t0
        print(f'epoch {epoch+1:3d}/{CONFIG["epochs"]}  train={train_loss:.4f}  '
              f'val={val_loss:.4f}  lr={lr:.6f}  [{elapsed:.0f}s]{marker}')

    if patience_counter >= CONFIG['patience']:
        print(f'\nearly stopping at epoch {epoch+1} (patience={CONFIG["patience"]})')
        break

model.load_state_dict(best_state)
model = model.to(device)
print(f'\nbest val_loss: {best_metric:.4f}')
print(f'total time: {time.time() - t0:.0f}s')

# --- evaluation ---

print('\n--- evaluation on held-out test patients ---')


def retrieval_metrics(query_emb, gallery_emb, ks=[1, 5, 10]):
    sim = cosine_similarity(query_emb, gallery_emb)
    ranks = (-sim).argsort(axis=1)
    n = len(query_emb)
    correct_ranks = np.array([np.where(ranks[i] == i)[0][0] for i in range(n)])
    results = {}
    for k in ks:
        results[f'recall@{k}'] = float((correct_ranks < k).mean())
    results['mrr'] = float((1.0 / (correct_ranks + 1)).mean())
    results['median_rank'] = float(np.median(correct_ranks))
    return results, correct_ranks


he_test_aligned, st_test_aligned = compute_aligned_embeddings(model, he_test, st_test)
n_test = len(he_test)

he2st, he2st_ranks = retrieval_metrics(he_test_aligned, st_test_aligned)
st2he, st2he_ranks = retrieval_metrics(st_test_aligned, he_test_aligned)

random_baseline = {
    'recall@1': 1.0 / n_test, 'recall@5': 5.0 / n_test,
    'recall@10': 10.0 / n_test,
}

aligned_r1 = (he2st['recall@1'] + st2he['recall@1']) / 2

print(f'\n{"=" * 70}')
print(f'{EXP_ID}: ComBat UNI2 + Novae | patient-held-out | n_test={n_test}')
print(f'{"=" * 70}')
print(f'{"method":<25} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"MRR":>8}')
print('-' * 70)
print(f'{"random":<25} {random_baseline["recall@1"]:>8.4f} '
      f'{random_baseline["recall@5"]:>8.4f} {random_baseline["recall@10"]:>8.4f}')
print(f'{"aligned H&E->ST":<25} {he2st["recall@1"]:>8.4f} '
      f'{he2st["recall@5"]:>8.4f} {he2st["recall@10"]:>8.4f} {he2st["mrr"]:>8.4f}')
print(f'{"aligned ST->H&E":<25} {st2he["recall@1"]:>8.4f} '
      f'{st2he["recall@5"]:>8.4f} {st2he["recall@10"]:>8.4f} {st2he["mrr"]:>8.4f}')
print('-' * 70)
print(f'mean R@1: {aligned_r1:.4f} (random: {random_baseline["recall@1"]:.4f}, '
      f'{aligned_r1 / random_baseline["recall@1"]:.1f}x)')

# --- patient separability in shared space ---

print('\n--- patient separability in shared 512-d space ---')

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# project all spots through alignment model
he_all = np.vstack([he_train, he_val, he_test])
st_all = np.vstack([st_train, st_val, st_test])
pids_all = np.concatenate([pids_train, pids_val, pids_test])

he_aligned_all, st_aligned_all = compute_aligned_embeddings(model, he_all, st_all)

# patient separability on H&E side of shared space
le = LabelEncoder()
y = le.fit_transform(pids_all)
n_classes = len(le.classes_)

rng = np.random.RandomState(SEED)
keep = []
for pid_idx in range(n_classes):
    idx_p = np.where(y == pid_idx)[0]
    if len(idx_p) > 200:
        idx_p = rng.choice(idx_p, 200, replace=False)
    keep.extend(idx_p)
keep = np.array(keep)
X_sub, y_sub = he_aligned_all[keep], y[keep]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
accs = []
for train_idx, test_idx in skf.split(X_sub, y_sub):
    scaler = StandardScaler().fit(X_sub[train_idx])
    clf = SGDClassifier(loss='hinge', max_iter=1000, random_state=SEED)
    clf.fit(scaler.transform(X_sub[train_idx]), y_sub[train_idx])
    accs.append(clf.score(scaler.transform(X_sub[test_idx]), y_sub[test_idx]))

shared_sep = np.mean(accs)
chance = 1.0 / n_classes
print(f'patient separability (shared H&E space): {shared_sep*100:.1f}% '
      f'(chance: {chance*100:.1f}%, {shared_sep/chance:.1f}x)')
print(f'vs raw UNI2: 62.6% (56x)')
print(f'vs ComBat UNI2: 4.1% (3.7x)')
if shared_sep < 0.041:
    print('-> MLP preserved ComBat correction (good)')
elif shared_sep < 0.10:
    print('-> MLP slightly re-encoded patient identity (acceptable)')
else:
    print('-> WARNING: MLP re-encoded patient identity (shortcut learning)')

# --- spatial-NN baseline ---

print('\n--- spatial-NN baseline ---')

# for each test spot, find nearest spatial neighbor and check if
# retrieval rank correlates with spatial distance
# (requires spot coordinates - skip if not available)
print('(spatial-NN baseline requires spot coordinates - deferred)')

# --- save results ---

run_dir = RUNS_DIR / EXP_ID
run_dir.mkdir(parents=True, exist_ok=True)

results = {
    'config': CONFIG,
    'metrics': {
        'he2st': he2st,
        'st2he': st2he,
        'recall_at_1': aligned_r1,
        'amplification_over_random': aligned_r1 / random_baseline['recall@1'],
    },
    'baselines': {
        'random_r1': random_baseline['recall@1'],
    },
    'patient_separability': {
        'shared_space': float(shared_sep),
        'shared_space_amplification': float(shared_sep / chance),
        'raw_uni2': 0.626,
        'combat_uni2': 0.041,
    },
    'training': {
        'best_val_loss': float(best_metric),
        'epochs_actual': len(history['train_loss']),
    },
}

with open(run_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
with open(run_dir / 'history.json', 'w') as f:
    json.dump(history, f)
torch.save(best_state, run_dir / 'model.pt')

print(f'\nsaved to {run_dir.relative_to(ROOT)}/')
for f in sorted(run_dir.iterdir()):
    print(f'  {f.name} ({f.stat().st_size / 1024:.0f} KB)')

print('\n--- summary ---')
print(f'cross-patient R@1: {aligned_r1:.4f} '
      f'({aligned_r1/random_baseline["recall@1"]:.1f}x over random)')
if aligned_r1 > 2 * random_baseline['recall@1']:
    print('H1 SUPPORTED: alignment learned patient-invariant cross-modal structure')
else:
    print('H1 NOT SUPPORTED at this resolution')