"""
rank2_alignment.py - InfoNCE alignment of Virchow2 niche (H&E) and rich ST.

architecture:
  H&E side:  Virchow2 niche (1280-d, raw, no z-score)
             -> MLP -> 512-d shared space

  ST side:   rich ST = mc_weights(14) + novae_zscored(64) + tls(1) = 79-d
             -> MLP -> 512-d shared space

  loss: InfoNCE with biology-driven positive pairs
        positive: cosine(mc_weights[i], mc_weights[j]) > 0.8 (within subarray)
        negative: all other spots in batch (in-batch negatives)

training: per-subarray batches
  batch = ALL spots from one subarray (~1000-1800)
  ensures positive pairs are biologically grounded
  mc_weight cosine determines who's a "match"

evaluation:
  H1 retrieval: R@1, R@5, R@10, MRR (within-subarray)
  baselines: random, unaligned (raw cosine), late fusion (raw concat + cosine)
  H2: ARI of aligned-space clusters vs MC labels (per-subarray)

usage:
    python scripts/rank2_alignment.py
"""

import warnings
warnings.filterwarnings('ignore')

import json
import subprocess
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
V2_DIR = ROOT / 'data' / 'embeddings' / 'virchow2_niche'
NOVAE_DIR = ROOT / 'data' / 'embeddings' / 'novae_all'
BIO_DIR = ROOT / 'data' / 'embeddings' / 'biological_signals'
RUNS_DIR = ROOT / 'runs' / 'tnbc-92' / 'rank2_alignment'
SEED = 42

CONFIG = {
    'h_e_dim': 1280,
    'st_dim': 79,
    'shared_dim': 512,
    'dropout': 0.1,
    'temperature': 0.1,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 50,
    'patience': 10,
    'mc_pos_threshold': 0.8,   # cosine sim threshold for positive pairs
    'mc_neg_threshold': 0.3,   # cosine sim threshold for hard negatives
    'min_spots_per_subarray': 200,  # skip very small subarrays
    'val_subarray_fraction': 0.15,
    'test_subarray_fraction': 0.15,
}

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ----------------- data loading -----------------

def load_aligned_per_subarray(max_subarrays=None):
    """build per-subarray bundles: virchow2_emb, rich_st, mc_weights, mc_labels.

    rich ST construction (no z-score, all components raw):
      - mc_weights: 14d, sums to 1.0 (NMF property)
      - novae: 64d, raw
      - tls: 1d, raw
      - = 79d
    LayerNorm(79) in ST projection puts components on comparable scale at MLP input.
    """
    print('loading data per subarray...')

    mc_labels_df = pd.read_csv(BIO_DIR / 'mc_labels.tsv', sep='\t')
    mc_weights_df = pd.read_csv(BIO_DIR / 'mc_weights.tsv', sep='\t')
    tls_df = pd.read_csv(BIO_DIR / 'tls_scores.tsv', sep='\t')
    tls_df['short'] = tls_df['subarray_id'].str.replace(r'^TNBC\d+_', '', regex=True)

    # build novae lookup
    novae_lookup = {}
    for f in NOVAE_DIR.glob('*.npy'):
        parts = f.stem.split('_', 1)
        if len(parts) == 2 and parts[0].startswith('TNBC'):
            novae_lookup[parts[1]] = f

    # build virchow2 lookup
    v2_lookup = {}
    for f in V2_DIR.glob('*.npy'):
        if f.stem.endswith('_meta'):
            continue
        parts = f.stem.split('_', 1)
        if len(parts) == 2 and parts[0].startswith('TNBC'):
            patient = int(parts[0].replace('TNBC', ''))
            v2_lookup[parts[1]] = (f, patient)

    bundles = {}
    n_loaded = 0
    items = list(v2_lookup.items())
    if max_subarrays is not None:
        rng = np.random.RandomState(SEED)
        # take more candidates than needed since some will fail loading
        n_take = min(max_subarrays * 3, len(items))
        chosen = rng.choice(len(items), n_take, replace=False)
        items = [items[i] for i in chosen]
    for subarray, (v2_path, patient) in items:
        if max_subarrays is not None and len(bundles) >= max_subarrays:
            break
        v2_emb = np.load(v2_path).astype(np.float32)

        slide, pos = subarray.split('_')
        rdata_path = ROOT / 'data' / 'inputs' / 'byArray' / slide / pos / 'selection.RData'
        if not rdata_path.exists():
            continue
        r = subprocess.run(['Rscript', '-e',
                            f'load("{rdata_path}"); cat(rownames(cnts), sep="\\n")'],
                           capture_output=True, text=True)
        spot_order = [s for s in r.stdout.strip().split('\n') if s]
        if len(spot_order) != v2_emb.shape[0]:
            continue

        mw_sub = mc_weights_df[mc_weights_df['subarray'] == subarray].set_index('spot_id')
        ml_sub = mc_labels_df[mc_labels_df['subarray'] == subarray].set_index('spot_id')
        tls_sub = tls_df[tls_df['short'] == subarray].set_index('spot_id')

        if len(mw_sub) == 0:
            continue

        # novae (raw, no z-score)
        novae_emb = None
        if subarray in novae_lookup:
            novae_emb = np.load(novae_lookup[subarray]).astype(np.float32)
            if novae_emb.shape[0] != v2_emb.shape[0]:
                novae_emb = None

        # build per-spot data
        v2_rows = []
        st_rows = []
        mc_w_rows = []
        mc_l_rows = []

        mc_cols = [f'mc{k}' for k in range(1, 15)]
        for i, sid in enumerate(spot_order):
            if sid not in mw_sub.index:
                continue
            mc_w = mw_sub.loc[sid, mc_cols].values.astype(np.float32)
            tls = float(tls_sub.loc[sid, 'tls_score']) if sid in tls_sub.index else 0.0
            novae_vec = novae_emb[i] if novae_emb is not None else np.zeros(64, dtype=np.float32)
            mc_label = int(ml_sub.loc[sid, 'megacluster']) if sid in ml_sub.index else -1

            rich_st = np.concatenate([mc_w, novae_vec, [tls]]).astype(np.float32)
            v2_rows.append(v2_emb[i])
            st_rows.append(rich_st)
            mc_w_rows.append(mc_w)
            mc_l_rows.append(mc_label)

        if len(v2_rows) < CONFIG['min_spots_per_subarray']:
            continue

        bundles[subarray] = {
            'v2': np.stack(v2_rows),
            'st': np.stack(st_rows),
            'mc_w': np.stack(mc_w_rows),
            'mc_l': np.array(mc_l_rows),
            'patient': patient,
        }
        n_loaded += 1
        if n_loaded % 50 == 0:
            print(f'  loaded {n_loaded} subarrays')

    print(f'\ntotal usable subarrays: {len(bundles)}')
    return bundles


def split_subarrays(bundles, val_frac, test_frac):
    """patient-grouped split: keep all subarrays of a patient in same split."""
    patient_to_subarrays = {}
    for sub, b in bundles.items():
        patient_to_subarrays.setdefault(b['patient'], []).append(sub)

    patients = sorted(patient_to_subarrays.keys())
    rng = np.random.RandomState(SEED)
    rng.shuffle(patients)

    n = len(patients)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_pat = set(patients[:n_test])
    val_pat = set(patients[n_test:n_test + n_val])
    train_pat = set(patients[n_test + n_val:])

    train, val, test = [], [], []
    for p, subs in patient_to_subarrays.items():
        if p in train_pat:
            train.extend(subs)
        elif p in val_pat:
            val.extend(subs)
        else:
            test.extend(subs)
    return train, val, test


# ----------------- model -----------------

class AlignmentMLP(nn.Module):
    def __init__(self, h_e_dim, st_dim, shared_dim, dropout):
        super().__init__()
        self.h_e_proj = nn.Sequential(
            nn.Linear(h_e_dim, shared_dim),
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

    def forward(self, h_e, st):
        h_e_emb = F.normalize(self.h_e_proj(h_e), dim=-1)
        st_emb = F.normalize(self.st_proj(st), dim=-1)
        return h_e_emb, st_emb


def info_nce_loss(h_e_emb, st_emb, temperature):
    """standard symmetric InfoNCE (CLIP loss).
    diagonal = positive pair, all others = negative.
    """
    logits = h_e_emb @ st_emb.T / temperature
    labels = torch.arange(len(h_e_emb), device=h_e_emb.device)
    loss_he2st = F.cross_entropy(logits, labels)
    loss_st2he = F.cross_entropy(logits.T, labels)
    return (loss_he2st + loss_st2he) / 2


# ----------------- training -----------------

def train_one_epoch(model, train_subarrays, bundles, optimizer):
    model.train()
    losses = []
    rng = np.random.RandomState()
    order = list(train_subarrays)
    rng.shuffle(order)

    for sub in order:
        b = bundles[sub]
        v2 = torch.from_numpy(b['v2']).to(device)
        st = torch.from_numpy(b['st']).to(device)
        mc_w = torch.from_numpy(b['mc_w']).to(device)

        h_e_emb, st_emb = model(v2, st)
        loss = info_nce_loss(h_e_emb, st_emb, CONFIG['temperature'])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


@torch.no_grad()
def eval_epoch(model, subarrays, bundles):
    model.eval()
    losses = []
    for sub in subarrays:
        b = bundles[sub]
        v2 = torch.from_numpy(b['v2']).to(device)
        st = torch.from_numpy(b['st']).to(device)
        mc_w = torch.from_numpy(b['mc_w']).to(device)
        h_e_emb, st_emb = model(v2, st)
        loss = info_nce_loss(h_e_emb, st_emb, CONFIG['temperature'])
        losses.append(loss.item())
    return np.mean(losses)


# ----------------- evaluation -----------------

@torch.no_grad()
def per_subarray_retrieval(model, subarrays, bundles):
    """compute R@1, R@5, R@10, MRR per subarray (within-subarray exact retrieval).
    standard cross-modal retrieval: for each H&E patch i, find the matching ST spot i.
    """
    model.eval()
    all_metrics = []
    for sub in subarrays:
        b = bundles[sub]
        v2 = torch.from_numpy(b['v2']).to(device)
        st = torch.from_numpy(b['st']).to(device)
        h_e_emb, st_emb = model(v2, st)
        sim = (h_e_emb @ st_emb.T).cpu().numpy()
        n = sim.shape[0]
        r1, r5, r10, mrr = _retrieval_from_sim(sim)
        all_metrics.append({'subarray': sub, 'n_spots': n,
                            'R@1': r1, 'R@5': r5, 'R@10': r10, 'MRR': mrr})
    return pd.DataFrame(all_metrics)


def _retrieval_from_sim(sim):
    """compute R@1, R@5, R@10, MRR using exact-self retrieval (diagonal as gold).
    this is the proposal's R@K - retrieve the matched ST spot for an H&E query."""
    n = sim.shape[0]
    order = (-sim).argsort(axis=1)
    r1 = r5 = r10 = 0
    mrr = 0.0
    for i in range(n):
        rank = int(np.where(order[i] == i)[0][0])
        if rank == 0: r1 += 1
        if rank < 5:  r5 += 1
        if rank < 10: r10 += 1
        mrr += 1.0 / (rank + 1)
    return r1/n, r5/n, r10/n, mrr/n


def baseline_random(subarrays, bundles):
    """baseline: random retrieval order. expected R@1 = 1/n_spots."""
    rng = np.random.RandomState(SEED)
    all_metrics = []
    for sub in subarrays:
        b = bundles[sub]
        n = b['v2'].shape[0]
        sim = rng.randn(n, n).astype(np.float32)
        r1, r5, r10, mrr = _retrieval_from_sim(sim)
        all_metrics.append({'subarray': sub, 'n_spots': n,
                            'R@1': r1, 'R@5': r5, 'R@10': r10, 'MRR': mrr})
    return pd.DataFrame(all_metrics)


def baseline_cca(subarrays, bundles, n_components=64):
    """baseline: CCA fit per subarray, retrieve via cosine in CCA space."""
    from sklearn.cross_decomposition import CCA
    all_metrics = []
    for sub in subarrays:
        b = bundles[sub]
        v2 = b['v2']
        st = b['st']
        n = v2.shape[0]
        k = min(n_components, v2.shape[1], st.shape[1], n - 1)
        try:
            cca = CCA(n_components=k, max_iter=200)
            v2_c, st_c = cca.fit_transform(v2, st)
        except Exception:
            v2_c, st_c = v2[:, :k], st[:, :k]
        v2_n = v2_c / (np.linalg.norm(v2_c, axis=1, keepdims=True) + 1e-8)
        st_n = st_c / (np.linalg.norm(st_c, axis=1, keepdims=True) + 1e-8)
        sim = v2_n @ st_n.T
        r1, r5, r10, mrr = _retrieval_from_sim(sim)
        all_metrics.append({'subarray': sub, 'n_spots': n,
                            'R@1': r1, 'R@5': r5, 'R@10': r10, 'MRR': mrr})
    return pd.DataFrame(all_metrics)


def baseline_late_fusion(subarrays, bundles):
    """baseline: late fusion - L2-norm both modalities, concat to (1280+79)d.
    nearest neighbor in concatenated space (effectively just paired distance).
    """
    all_metrics = []
    for sub in subarrays:
        b = bundles[sub]
        v2 = b['v2']
        st = b['st']
        n = v2.shape[0]
        v2_n = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8)
        st_n = st / (np.linalg.norm(st, axis=1, keepdims=True) + 1e-8)
        # for cross-modal retrieval in concat space, query is concat(v2_i, 0)
        # gallery is concat(0, st_j) - this gives sim = v2_i . 0 + 0 . st_j = 0
        # so the standard "late fusion" baseline for cross-modal is: project both to same dim with random matrix
        # simpler: pad shorter modality and compute cosine
        d = max(v2_n.shape[1], st_n.shape[1])
        rng = np.random.RandomState(SEED)
        # random projection both to common d
        if v2_n.shape[1] != d:
            P = rng.randn(v2_n.shape[1], d).astype(np.float32) / np.sqrt(v2_n.shape[1])
            v2_proj = v2_n @ P
        else:
            v2_proj = v2_n
        if st_n.shape[1] != d:
            P = rng.randn(st_n.shape[1], d).astype(np.float32) / np.sqrt(st_n.shape[1])
            st_proj = st_n @ P
        else:
            st_proj = st_n
        v2_proj = v2_proj / (np.linalg.norm(v2_proj, axis=1, keepdims=True) + 1e-8)
        st_proj = st_proj / (np.linalg.norm(st_proj, axis=1, keepdims=True) + 1e-8)
        sim = v2_proj @ st_proj.T
        r1, r5, r10, mrr = _retrieval_from_sim(sim)
        all_metrics.append({'subarray': sub, 'n_spots': n,
                            'R@1': r1, 'R@5': r5, 'R@10': r10, 'MRR': mrr})
    return pd.DataFrame(all_metrics)


# ----------------- main -----------------

@click.command()
@click.option('--subset', type=int, default=None,
              help='use N subarrays for quick test (default: all)')
@click.option('--epochs', type=int, default=None,
              help='override CONFIG epochs')
@click.option('--run-name', default='rank2_alignment',
              help='subdir name under runs/tnbc-92/')
def main(subset, epochs, run_name):
    global RUNS_DIR
    RUNS_DIR = ROOT / 'runs' / 'tnbc-92' / run_name

    if epochs is not None:
        CONFIG['epochs'] = epochs
        CONFIG['patience'] = max(3, epochs // 2)

    print('=== rank2: alignment training ===\n')
    print(f'device: {device}')
    if subset is not None:
        print(f'SUBSET MODE: {subset} subarrays')
    for k, v in CONFIG.items():
        print(f'  {k}: {v}')

    bundles = load_aligned_per_subarray(max_subarrays=subset)
    train_subs, val_subs, test_subs = split_subarrays(
        bundles, CONFIG['val_subarray_fraction'], CONFIG['test_subarray_fraction'])
    train_pat = len(set(bundles[s]['patient'] for s in train_subs))
    val_pat = len(set(bundles[s]['patient'] for s in val_subs))
    test_pat = len(set(bundles[s]['patient'] for s in test_subs))
    print(f'\nsplit: {len(train_subs)} train ({train_pat} pat) / '
          f'{len(val_subs)} val ({val_pat} pat) / '
          f'{len(test_subs)} test ({test_pat} pat) subarrays')

    # check that the rich ST is right size
    sample_st = bundles[train_subs[0]]['st']
    print(f'rich ST shape sample: {sample_st.shape} (expected dim={CONFIG["st_dim"]})')
    assert sample_st.shape[1] == CONFIG['st_dim']

    model = AlignmentMLP(
        CONFIG['h_e_dim'], CONFIG['st_dim'], CONFIG['shared_dim'], CONFIG['dropout']
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nmodel: {n_params:,} params')

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                   weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'])

    # baselines on test (exact self-retrieval - standard cross-modal R@K)
    print('\n--- baselines on test (exact self retrieval) ---')
    base_random = baseline_random(test_subs, bundles)
    base_cca = baseline_cca(test_subs, bundles)
    base_lf = baseline_late_fusion(test_subs, bundles)
    for name, df in [('random', base_random), ('CCA', base_cca), ('late fusion', base_lf)]:
        print(f'{name:<12} R@1={df["R@1"].mean():.4f}  '
              f'R@5={df["R@5"].mean():.4f}  '
              f'R@10={df["R@10"].mean():.4f}  '
              f'MRR={df["MRR"].mean():.4f}')

    print('\n--- training ---')
    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    best_state = None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(CONFIG['epochs']):
        train_loss = train_one_epoch(model, train_subs, bundles, optimizer)
        val_loss = eval_epoch(model, val_subs, bundles)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            mark = ' *'
        else:
            patience_counter += 1
            mark = ''

        elapsed = time.time() - t0
        print(f'epoch {epoch+1:3d}/{CONFIG["epochs"]}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  [{elapsed:.0f}s]{mark}')

        if patience_counter >= CONFIG['patience']:
            print(f'\nearly stopping at epoch {epoch+1}')
            break

    model.load_state_dict(best_state)
    model = model.to(device)
    print(f'\nbest val_loss: {best_val:.4f}')

    # final evaluation on test set
    print('\n--- aligned retrieval on test ---')
    aligned_test = per_subarray_retrieval(model, test_subs, bundles)
    print(f'aligned      R@1={aligned_test["R@1"].mean():.4f}  '
          f'R@5={aligned_test["R@5"].mean():.4f}  '
          f'R@10={aligned_test["R@10"].mean():.4f}  '
          f'MRR={aligned_test["MRR"].mean():.4f}')

    # comparison table
    print('\n=== comparison (test set, exact self-retrieval) ===')
    header = f'{"method":<12} {"R@1":>10} {"R@5":>10} {"R@10":>10} {"MRR":>10}'
    print(header)
    print('-' * len(header))
    for name, df in [('random', base_random), ('CCA', base_cca),
                      ('late fusion', base_lf), ('aligned', aligned_test)]:
        print(f'{name:<12} {df["R@1"].mean():>10.4f} {df["R@5"].mean():>10.4f} '
              f'{df["R@10"].mean():>10.4f} {df["MRR"].mean():>10.4f}')

    # save
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'config': CONFIG,
        'split': {'train': train_subs, 'val': val_subs, 'test': test_subs},
        'baseline_random': {col: float(base_random[col].mean()) for col in ['R@1', 'R@5', 'R@10', 'MRR']},
        'baseline_cca': {col: float(base_cca[col].mean()) for col in ['R@1', 'R@5', 'R@10', 'MRR']},
        'baseline_late_fusion': {col: float(base_lf[col].mean()) for col in ['R@1', 'R@5', 'R@10', 'MRR']},
        'aligned': {col: float(aligned_test[col].mean()) for col in ['R@1', 'R@5', 'R@10', 'MRR']},
        'best_val_loss': float(best_val),
        'epochs_actual': len(history['train_loss']),
    }
    with open(RUNS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(RUNS_DIR / 'history.json', 'w') as f:
        json.dump(history, f)
    aligned_test.to_csv(RUNS_DIR / 'aligned_per_subarray.tsv', sep='\t', index=False)
    base_random.to_csv(RUNS_DIR / 'baseline_random_per_subarray.tsv', sep='\t', index=False)
    base_cca.to_csv(RUNS_DIR / 'baseline_cca_per_subarray.tsv', sep='\t', index=False)
    base_lf.to_csv(RUNS_DIR / 'baseline_late_fusion_per_subarray.tsv', sep='\t', index=False)
    torch.save(best_state, RUNS_DIR / 'model.pt')
    print(f'\nsaved to {RUNS_DIR.relative_to(ROOT)}/')


if __name__ == '__main__':
    main()
