#!/usr/bin/env python
"""align.py - contrastive alignment training for tnbc-92 niche grid.

trains one config, saves all artifacts needed downstream for H1 / H2 / H3 eval
and synthesis plots. classical baselines (CCA, Procrustes, unaligned concat)
are not trained here - they live in scripts/align_classical.py.

usage:
    python scripts/align.py --config projects/tnbc-92/alignment/config/R1.json
    python scripts/align.py --config ... --subset 20 --epochs 5  # smoke test overrides

outputs per run at runs/tnbc-92/{run_id}/:
    run_config.json           # full config + resolved defaults
    split.json                # train/val/test patient lists + seed
    training_log.csv          # per-epoch train_loss, val_loss, val_cos_sim, val_r1, lr
    checkpoint.pt             # best model state_dict
    embeddings_test.parquet   # per-niche z_he, z_st, + attention_weights (cross_attn only)
    metrics_h1_raw.json       # R@K, MRR, median_rank, alignment_gap, AUC, CKA

scope: matched 260 subarrays per data/embeddings/niches/manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NICHES_DIR = ROOT / 'data' / 'embeddings' / 'niches'
RUNS_ROOT = ROOT / 'runs' / 'tnbc-92'


def _git_commit_sha() -> str:
    """current git HEAD sha (12 chars), or 'unknown' if not in a git repo."""
    import subprocess
    try:
        r = subprocess.run(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip()[:12] if r.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'

# ------------------------------------------------------------------------- #
# 1. config
# ------------------------------------------------------------------------- #

@dataclass
class Config:
    run_id: str
    loss: Literal['infonce', 'supcon', 'barlow', 'aninfonce'] = 'infonce'
    fusion: Literal['late', 'cross_attn'] = 'late'   # 'early' TBD
    projection: Literal['mlp'] = 'mlp'                # only trained option
    supervision: Literal['none', 'mc_weights_soft'] = 'none'
    st_features: tuple[str, ...] = ('novae_niche', 'gpath2vec_niche')  # 64 + 512 = 576
    anisotropic: bool = False           # R5: per-dim learnable scaling for AnInfoNCE

    shared_dim: int = 512
    tau: float = 0.07
    tau_target: float = 0.1            # supcon soft-target softmax temperature
    barlow_lambda: float = 5e-3        # off-diagonal weight in barlow loss
    dropout: float = 0.3

    lr: float = 5e-4
    weight_decay: float = 1e-3
    epochs: int = 50
    patience: int = 10
    seed: int = 42

    val_frac: float = 0.15
    test_frac: float = 0.15
    subset_n_subarrays: int | None = None   # None = use all matched
    device: str = 'cpu'                     # 'cuda' if available
    niches_dir: str | None = None           # None -> DEFAULT_NICHES_DIR (v1 back-compat)

    # provenance fields (filled at runtime by main(), not by user in config)
    git_commit_sha: str | None = None
    gpath2vec_parquet_sha256: str | None = None
    niches_manifest_path: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> 'Config':
        data = json.loads(Path(path).read_text())
        data['st_features'] = tuple(data.get('st_features', ['novae_niche', 'gpath2vec_niche']))
        # hard ban: mc_weights_niche is supervision-only per program.md constraint #3
        assert 'mc_weights_niche' not in data['st_features'], (
            f'CIRCULAR DEPENDENCY BAN: mc_weights_niche must not appear in '
            f'st_features (supervision-only per program.md hard constraint #3). '
            f'got st_features={data["st_features"]}')
        return cls(**data)

    def to_json(self) -> str:
        d = asdict(self)
        d['st_features'] = list(d['st_features'])
        return json.dumps(d, indent=2)


# ------------------------------------------------------------------------- #
# 2. data
# ------------------------------------------------------------------------- #

def _as_array(cell, dtype=np.float32) -> np.ndarray:
    """parquet list columns come back as lists; normalize to np array."""
    return np.asarray(cell, dtype=dtype)


def load_subarray(path: Path, st_features: tuple[str, ...]) -> dict:
    """load one subarray's niche parquet, materialize feature tensors.

    returns dict with numpy arrays:
        spot_id        (N,) str
        patient_id     int
        archetype      int
        compartment    (N,) str / None
        x_he_niche     (N, 1280)  virchow2 niche (late/cross_attn fallback)
        x_he_tokens    (N, 7, 1280)  virchow2 cell tokens (cross_attn)
        x_st           (N, D_st)  concatenated ST features
        mc_w           (N, 14)  for supcon supervision
    """
    df = pd.read_parquet(path)
    N = len(df)
    he_niche = np.stack([_as_array(v) for v in df['virchow2_niche']])
    he_tokens = np.stack([_as_array(v).reshape(7, 1280) for v in df['virchow2_cell_tokens']])

    st_parts = []
    for feat in st_features:
        arr = np.stack([_as_array(v) for v in df[feat]])
        st_parts.append(arr)
    x_st = np.concatenate(st_parts, axis=1)

    mc_w = np.stack([_as_array(v) for v in df['mc_weights_niche']])
    # supcon supervision: impute NaN mc_w rows (boundary pooling) with uniform
    # prior so their soft targets are uninformative but don't break the loss.
    nan_mcw = np.isnan(mc_w).any(axis=1)
    if nan_mcw.any():
        mc_w[nan_mcw] = 1.0 / mc_w.shape[1]

    # drop niches with NaN in always-used features (he + st). a single NaN row
    # poisons the per-subarray batch through the softmax.
    valid = (
        ~np.isnan(he_niche).any(axis=1)
        & ~np.isnan(he_tokens.reshape(len(he_tokens), -1)).any(axis=1)
        & ~np.isnan(x_st).any(axis=1)
    )

    pid = df['patient_id'].iloc[0]
    arch = df['archetype'].iloc[0]
    return {
        'spot_id': df.index.values[valid],
        'patient_id': int(pid) if pd.notna(pid) else None,
        'archetype': int(arch) if pd.notna(arch) else None,
        'compartment': df['compartment'].values[valid],
        'subarray': str(df['subarray'].iloc[0]),
        'x_he_niche': he_niche[valid].astype(np.float32),
        'x_he_tokens': he_tokens[valid].astype(np.float32),
        'x_st': x_st[valid].astype(np.float32),
        'mc_w': mc_w[valid].astype(np.float32),
        'n_dropped_nan': int((~valid).sum()),
    }


def discover_subarrays(niches_dir: Path = DEFAULT_NICHES_DIR) -> list[Path]:
    return sorted(niches_dir.glob('*.parquet'))


def _try_stratify(arches: np.ndarray) -> np.ndarray | None:
    """return stratify array if every class has >= 2 members, else None."""
    unique, counts = np.unique(arches, return_counts=True)
    if counts.min() >= 2:
        return arches
    return None


def split_by_patient(
    subarrays: list[dict],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """three-way patient split, stratified by archetype when class counts allow.

    falls back to unstratified split for smoke-test subsets with rare classes.
    """
    by_patient = {}
    for sub in subarrays:
        by_patient[sub['patient_id']] = sub['archetype']
    patients = np.array(sorted(by_patient.keys()))
    arches = np.array([by_patient[p] for p in patients])

    stratify1 = _try_stratify(arches)
    train_val_p, test_p = train_test_split(
        patients, test_size=test_frac, stratify=stratify1, random_state=seed,
    )
    train_val_arch = np.array([by_patient[p] for p in train_val_p])
    val_size = val_frac / (1.0 - test_frac)
    stratify2 = _try_stratify(train_val_arch)
    train_p, val_p = train_test_split(
        train_val_p, test_size=val_size, stratify=stratify2, random_state=seed,
    )
    return sorted(train_p.tolist()), sorted(val_p.tolist()), sorted(test_p.tolist())


# ------------------------------------------------------------------------- #
# 3. models (fusion)
# ------------------------------------------------------------------------- #

class MLPBlock(nn.Module):
    """LN -> Linear -> ReLU -> BN -> Dropout -> Linear -> L2-norm."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.ln(x)
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.drop(x)
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


class LateFusion(nn.Module):
    """independent two-stream MLPs -> shared 512d (proposal's default).

    `anisotropic=True` registers a per-dim learnable log-scale parameter
    `aniso_log_scale` (shape (shared,), init 0). when the loss is AnInfoNCE,
    `compute_loss` reads it and applies `diag(exp(2*log_scale))` to the
    bilinear inner product - equivalent to a learned diagonal Mahalanobis
    metric on the shared latent. init 0 -> uniform scale = standard InfoNCE
    at step 0 (clean warm-start).
    """
    def __init__(self, he_dim: int, st_dim: int, shared: int, dropout: float,
                 anisotropic: bool = False):
        super().__init__()
        self.he_mlp = MLPBlock(he_dim, shared, dropout)
        self.st_mlp = MLPBlock(st_dim, shared, dropout)
        if anisotropic:
            self.aniso_log_scale = nn.Parameter(torch.zeros(shared))
        else:
            self.aniso_log_scale = None

    def forward(self, x_he_niche, x_he_tokens, x_st):
        z_he = self.he_mlp(x_he_niche)
        z_st = self.st_mlp(x_st)
        return z_he, z_st, None


class CrossAttnFusion(nn.Module):
    """ST query attends to 7 H&E tile tokens; symmetric InfoNCE between z_he_fused and z_st."""
    def __init__(self, he_tile_dim: int, st_dim: int, shared: int, dropout: float):
        super().__init__()
        self.tile_ln = nn.LayerNorm(he_tile_dim)
        self.st_ln = nn.LayerNorm(st_dim)
        self.st_proj = nn.Linear(st_dim, shared)
        self.k_proj = nn.Linear(he_tile_dim, shared)
        self.v_proj = nn.Linear(he_tile_dim, shared)
        self.out_ln = nn.LayerNorm(shared)
        self.he_out = nn.Linear(shared, shared)
        self.st_out = nn.Linear(shared, shared)
        self.shared = shared

    def forward(self, x_he_niche, x_he_tokens, x_st):
        # x_he_tokens: (N, 7, 1280)
        # x_st:        (N, D_st)
        # z_he MUST be derived purely from H&E tiles (no ST residual leak).
        # ST only shapes the ATTENTION WEIGHTS over tiles, not the content.
        tile_tokens = self.tile_ln(x_he_tokens)              # (N, 7, 1280)
        st_ln = self.st_ln(x_st)
        st_query = self.st_proj(st_ln).unsqueeze(1)          # (N, 1, shared)
        K = self.k_proj(tile_tokens)                         # (N, 7, shared)
        V = self.v_proj(tile_tokens)                         # (N, 7, shared)
        scores = torch.matmul(st_query, K.transpose(-1, -2)) / (self.shared ** 0.5)
        attn = F.softmax(scores, dim=-1)                     # (N, 1, 7)
        attn_out = torch.matmul(attn, V).squeeze(1)          # (N, shared) - pure H&E content
        z_he = F.normalize(self.he_out(self.out_ln(attn_out)), dim=-1)  # no ST residual
        z_st = F.normalize(self.st_out(st_query.squeeze(1)), dim=-1)
        return z_he, z_st, attn.squeeze(1)


def build_model(cfg: Config, he_dim: int, he_tile_dim: int, st_dim: int) -> nn.Module:
    if cfg.fusion == 'late':
        return LateFusion(he_dim, st_dim, cfg.shared_dim, cfg.dropout,
                          anisotropic=cfg.anisotropic)
    if cfg.fusion == 'cross_attn':
        # cross_attn + anisotropic is a separate experiment; raise if mis-combined
        assert not cfg.anisotropic, (
            'anisotropic=True is currently only wired for late fusion (R5). '
            'cross_attn + aninfonce is a separate experiment - extend '
            'CrossAttnFusion with aniso_log_scale before enabling.')
        return CrossAttnFusion(he_tile_dim, st_dim, cfg.shared_dim, cfg.dropout)
    raise ValueError(f'unsupported fusion: {cfg.fusion}')


# ------------------------------------------------------------------------- #
# 4. losses
# ------------------------------------------------------------------------- #

def infonce_loss(z_he, z_st, tau=0.07):
    """standard symmetric InfoNCE (CLIP-style). matched pairs on diagonal = positives."""
    logits = z_he @ z_st.T / tau
    N = logits.size(0)
    targets = torch.arange(N, device=logits.device)
    loss_he = F.cross_entropy(logits, targets)
    loss_st = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_he + loss_st)


def aninfonce_loss(z_he, z_st, aniso_log_scale, tau=0.07):
    """anisotropic InfoNCE (R5). per-dim learnable scaling on the bilinear inner
    product -> learned diagonal Mahalanobis metric on the shared latent.

    `aniso_log_scale` (D,) is a learnable model parameter (from LateFusion when
    cfg.anisotropic=True). bilinear form: `z_he @ diag(exp(2*log_scale)) @ z_st.T`.
    log_scale = 0 -> uniform scale -> identical to standard InfoNCE.

    motivation: standard InfoNCE treats all directions of the shared latent
    equivalently; anisotropic temperature lets the loss weight axes that carry
    matched-pair signal more heavily, which may reduce direction-aliasing on
    correlated pathway-anchored embeddings (the R4 specificity finding).
    """
    assert aniso_log_scale is not None, (
        'aninfonce requires anisotropic=True on the fusion model so '
        '`aniso_log_scale` is registered as a learnable parameter.')
    scale_sq = (2.0 * aniso_log_scale).exp()                         # (D,) positive per-dim
    logits = (z_he * scale_sq.unsqueeze(0)) @ z_st.T / tau           # (N, N) - Mahalanobis bilinear
    N = logits.size(0)
    targets = torch.arange(N, device=logits.device)
    loss_he = F.cross_entropy(logits, targets)
    loss_st = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_he + loss_st)


def supcon_loss(z_he, z_st, mc_w, tau=0.07, tau_target=0.1):
    """soft-supervised contrastive: targets come from mc_weights cosine similarity."""
    # normalize mc_w and build soft target distribution
    mc_norm = F.normalize(torch.from_numpy(mc_w).to(z_he.device) if not torch.is_tensor(mc_w) else F.normalize(mc_w, dim=-1), dim=-1)
    sim_target = mc_norm @ mc_norm.T                  # (N, N), cosine in mc space
    target_dist = F.softmax(sim_target / tau_target, dim=-1)
    logits = z_he @ z_st.T / tau
    log_pred_he = F.log_softmax(logits, dim=-1)
    log_pred_st = F.log_softmax(logits.T, dim=-1)
    loss_he = -(target_dist * log_pred_he).sum(dim=-1).mean()
    loss_st = -(target_dist * log_pred_st).sum(dim=-1).mean()
    return 0.5 * (loss_he + loss_st)


def barlow_loss(z_he, z_st, lam=5e-3):
    """barlow twins: cross-correlation matrix -> identity."""
    # center (BN handles this partially; explicit centering here for robustness)
    zc_he = z_he - z_he.mean(0, keepdim=True)
    zc_st = z_st - z_st.mean(0, keepdim=True)
    # normalize per-dim std
    zc_he = zc_he / (zc_he.std(0, keepdim=True) + 1e-8)
    zc_st = zc_st / (zc_st.std(0, keepdim=True) + 1e-8)
    N, D = z_he.shape
    C = (zc_he.T @ zc_st) / N                         # (D, D)
    on_diag = (torch.diagonal(C) - 1.0).pow(2).sum()
    off_diag = (C - torch.diag(torch.diagonal(C))).pow(2).sum()
    return on_diag + lam * off_diag


def compute_loss(cfg: Config, z_he, z_st, mc_w=None, model=None):
    if cfg.loss == 'infonce':
        return infonce_loss(z_he, z_st, cfg.tau)
    if cfg.loss == 'aninfonce':
        assert model is not None, 'aninfonce needs `model` to read aniso_log_scale'
        return aninfonce_loss(z_he, z_st, model.aniso_log_scale, cfg.tau)
    if cfg.loss == 'supcon':
        assert mc_w is not None, 'supcon requires mc_weights_niche supervision'
        return supcon_loss(z_he, z_st, mc_w, cfg.tau, cfg.tau_target)
    if cfg.loss == 'barlow':
        return barlow_loss(z_he, z_st, cfg.barlow_lambda)
    raise ValueError(f'unknown loss: {cfg.loss}')


# ------------------------------------------------------------------------- #
# 5. eval metrics
# ------------------------------------------------------------------------- #

def retrieval_metrics(z_he: torch.Tensor, z_st: torch.Tensor) -> dict:
    """R@1/5/10, MRR, median rank, alignment gap, AUC over one unified pool."""
    sim = (z_he @ z_st.T).cpu().numpy()
    N = sim.shape[0]
    ranks = (-sim).argsort(axis=1).argsort(axis=1)[np.arange(N), np.arange(N)]  # 0-indexed rank of self
    r1 = float((ranks < 1).mean())
    r5 = float((ranks < 5).mean())
    r10 = float((ranks < 10).mean())
    mrr = float((1.0 / (ranks + 1)).mean())
    med_rank = float(np.median(ranks + 1))
    matched = np.diag(sim)
    mask = ~np.eye(N, dtype=bool)
    mismatched = sim[mask]
    gap = float(matched.mean() - mismatched.mean())
    # AUC: matched vs sampled mismatched (equal counts for speed)
    neg_sample = np.random.default_rng(0).choice(mismatched, size=min(len(matched) * 10, len(mismatched)), replace=False)
    labels = np.concatenate([np.ones(len(matched)), np.zeros(len(neg_sample))])
    scores = np.concatenate([matched, neg_sample])
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = float('nan')
    return {'r1': r1, 'r5': r5, 'r10': r10, 'mrr': mrr, 'median_rank': med_rank,
            'alignment_gap': gap, 'auc': auc, 'n_queries': N}


def cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """centered kernel alignment (linear kernel)."""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = (Xc.T @ Yc).reshape(-1) @ (Xc.T @ Yc).reshape(-1)
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(num / (den + 1e-12))


# ------------------------------------------------------------------------- #
# 6. training
# ------------------------------------------------------------------------- #

def set_seed(seed: int):
    """seed every randomness source we touch. publishability gate: rerunning
    with the same seed must produce bit-identical metrics (see step 5 acceptance
    check in v3_phase2_plan.md)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # use_deterministic_algorithms forces non-determinism errors to raise. on
    # CPU this is effectively a guarantee with the seeded ops above; off by
    # default to avoid breaking ops that are deterministic-but-not-flagged.
    # turn on for the final determinism re-run check.


def epoch_pass(
    model, subs: list[dict], cfg: Config, train: bool,
    optim=None, device='cpu',
):
    model.train() if train else model.eval()
    losses, cos_sims, r1s = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for sub in subs:
            x_he_niche = torch.from_numpy(sub['x_he_niche']).to(device)
            x_he_tokens = torch.from_numpy(sub['x_he_tokens']).to(device)
            x_st = torch.from_numpy(sub['x_st']).to(device)
            mc_w = torch.from_numpy(sub['mc_w']).to(device) if cfg.loss == 'supcon' else None
            z_he, z_st, _ = model(x_he_niche, x_he_tokens, x_st)
            loss = compute_loss(cfg, z_he, z_st, mc_w=mc_w, model=model)
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            losses.append(loss.item())
            # val diagnostics (cheap)
            with torch.no_grad():
                cos_sims.append(float((z_he * z_st).sum(-1).mean()))
                sim = z_he @ z_st.T
                N = sim.size(0)
                ranks = (-sim).argsort(dim=1).argsort(dim=1)[torch.arange(N), torch.arange(N)]
                r1s.append(float((ranks < 1).float().mean()))
    return float(np.mean(losses)), float(np.mean(cos_sims)), float(np.mean(r1s))


def train_model(cfg: Config, subs_by_split: dict, run_dir: Path, device='cpu'):
    train_subs, val_subs = subs_by_split['train'], subs_by_split['val']
    he_dim = train_subs[0]['x_he_niche'].shape[1]
    he_tile_dim = train_subs[0]['x_he_tokens'].shape[2]
    st_dim = train_subs[0]['x_st'].shape[1]
    model = build_model(cfg, he_dim, he_tile_dim, st_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    log_path = run_dir / 'training_log.csv'
    best_val = -float('inf')
    bad_epochs = 0
    with log_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_cos_sim', 'val_r1', 'lr'])
        for ep in range(cfg.epochs):
            random.shuffle(train_subs)
            tr_loss, _, _ = epoch_pass(model, train_subs, cfg, train=True, optim=optim, device=device)
            va_loss, va_cos, va_r1 = epoch_pass(model, val_subs, cfg, train=False, device=device)
            lr = optim.param_groups[0]['lr']
            writer.writerow([ep, tr_loss, va_loss, va_cos, va_r1, lr])
            f.flush()
            print(f'  epoch {ep:3d} | train {tr_loss:.4f} | val_loss {va_loss:.4f} | '
                  f'val_cos {va_cos:.4f} | val_r1 {va_r1:.4f}')
            if va_cos > best_val:
                best_val = va_cos
                bad_epochs = 0
                torch.save(model.state_dict(), run_dir / 'checkpoint.pt')
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    print(f'  early stop at epoch {ep} (val_cos plateau)')
                    break
    model.load_state_dict(torch.load(run_dir / 'checkpoint.pt'))
    return model


# ------------------------------------------------------------------------- #
# 7. test + save
# ------------------------------------------------------------------------- #

def save_test_artifacts(model, test_subs: list[dict], cfg: Config, run_dir: Path, device='cpu'):
    """run model on test subarrays, save embeddings + attention + H1 metrics."""
    model.eval()
    rows = []
    all_z_he, all_z_st = [], []
    all_he_niche, all_st = [], []  # for CKA before projection
    with torch.no_grad():
        for sub in test_subs:
            x_he_niche = torch.from_numpy(sub['x_he_niche']).to(device)
            x_he_tokens = torch.from_numpy(sub['x_he_tokens']).to(device)
            x_st = torch.from_numpy(sub['x_st']).to(device)
            z_he, z_st, attn = model(x_he_niche, x_he_tokens, x_st)
            z_he_np = z_he.cpu().numpy().astype(np.float32)
            z_st_np = z_st.cpu().numpy().astype(np.float32)
            attn_np = attn.cpu().numpy().astype(np.float32) if attn is not None else None
            for i, sid in enumerate(sub['spot_id']):
                row = {
                    'spot_id': sid,
                    'subarray': sub['subarray'],
                    'patient_id': sub['patient_id'],
                    'archetype': sub['archetype'],
                    'compartment': sub['compartment'][i],
                    'z_he': z_he_np[i].tolist(),
                    'z_st': z_st_np[i].tolist(),
                }
                if attn_np is not None:
                    row['attention_weights'] = attn_np[i].tolist()
                rows.append(row)
            all_z_he.append(z_he_np)
            all_z_st.append(z_st_np)
            all_he_niche.append(sub['x_he_niche'])
            all_st.append(sub['x_st'])

    emb_df = pd.DataFrame(rows).set_index('spot_id')
    emb_df.to_parquet(run_dir / 'embeddings_test.parquet')

    Z_he = np.concatenate(all_z_he)
    Z_st = np.concatenate(all_z_st)
    X_he = np.concatenate(all_he_niche)
    X_st = np.concatenate(all_st)

    # H1 metrics on full test pool (cross-subarray retrieval by construction)
    metrics = retrieval_metrics(torch.from_numpy(Z_he), torch.from_numpy(Z_st))
    metrics['cka_before'] = cka_linear(X_he, X_st)
    metrics['cka_after'] = cka_linear(Z_he, Z_st)
    metrics['test_subarrays'] = len(test_subs)
    metrics['test_niches'] = len(emb_df)
    (run_dir / 'metrics_h1_raw.json').write_text(json.dumps(metrics, indent=2))
    print(f'\nH1 metrics: R@1={metrics["r1"]:.3f} R@5={metrics["r5"]:.3f} '
          f'R@10={metrics["r10"]:.3f} MRR={metrics["mrr"]:.3f} '
          f'gap={metrics["alignment_gap"]:.3f} CKA {metrics["cka_before"]:.3f}->{metrics["cka_after"]:.3f}')


# ------------------------------------------------------------------------- #
# 8. main
# ------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--subset', type=int, default=None, help='override: first N subarrays for smoke test')
    ap.add_argument('--epochs', type=int, default=None, help='override: epochs')
    ap.add_argument('--niches-dir', type=Path, default=None,
                    help='override: path to the niche-join parquet dir '
                         '(defaults to config.niches_dir or v1 location). '
                         'use data/embeddings/niches_v3 for the v3 retrain.')
    ap.add_argument('--runs-root', type=Path, default=None,
                    help='override: parent dir for the run output. defaults '
                         'to runs/tnbc-92/. use runs/tnbc-92_v3/ for retrains.')
    args = ap.parse_args()

    cfg = Config.from_file(Path(args.config))
    if args.subset is not None:
        cfg.subset_n_subarrays = args.subset
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.niches_dir is not None:
        cfg.niches_dir = str(args.niches_dir)
    niches_dir = (Path(cfg.niches_dir) if cfg.niches_dir else DEFAULT_NICHES_DIR)
    assert niches_dir.is_dir(), f'niches_dir not found: {niches_dir}'

    runs_root = args.runs_root if args.runs_root is not None else RUNS_ROOT

    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but unavailable, falling back to CPU')
        device = 'cpu'
    set_seed(cfg.seed)

    # provenance: lock the niche-join build the run was trained on
    cfg.git_commit_sha = _git_commit_sha()
    cfg.niches_manifest_path = str((niches_dir / 'manifest.json').resolve())
    if (niches_dir / 'manifest.json').is_file():
        _man = json.loads((niches_dir / 'manifest.json').read_text())
        cfg.gpath2vec_parquet_sha256 = _man.get('sources', {}).get('gpath2vec_sha256')

    run_dir = runs_root / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'run_config.json').write_text(cfg.to_json())

    print(f'run_id: {cfg.run_id}')
    print(f'loss={cfg.loss} fusion={cfg.fusion} projection={cfg.projection} '
          f'supervision={cfg.supervision} st_features={cfg.st_features}')
    print(f'niches_dir: {niches_dir.relative_to(ROOT) if niches_dir.is_relative_to(ROOT) else niches_dir}')
    print(f'output: {run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}')
    print(f'git: {cfg.git_commit_sha}, gpath_sha: {(cfg.gpath2vec_parquet_sha256 or "n/a")[:16]}...')

    # load all matched subarrays
    paths = discover_subarrays(niches_dir)
    if cfg.subset_n_subarrays is not None:
        paths = paths[:cfg.subset_n_subarrays]
    print(f'loading {len(paths)} subarrays...')
    subs_raw = [load_subarray(p, cfg.st_features) for p in paths]
    # drop subarrays missing patient_id or archetype (affects split + eval)
    subs = [s for s in subs_raw if s['patient_id'] is not None and s['archetype'] is not None]
    if len(subs) < len(subs_raw):
        dropped = [s['subarray'] for s in subs_raw if s['patient_id'] is None or s['archetype'] is None]
        print(f'  dropped {len(dropped)} subarray(s) with missing patient_id/archetype: {dropped}')

    train_p, val_p, test_p = split_by_patient(subs, cfg.val_frac, cfg.test_frac, cfg.seed)
    split_info = {
        'seed': cfg.seed,
        'train_patients': train_p,
        'val_patients': val_p,
        'test_patients': test_p,
        'n_train': len(train_p), 'n_val': len(val_p), 'n_test': len(test_p),
    }
    (run_dir / 'split.json').write_text(json.dumps(split_info, indent=2))
    train_subs = [s for s in subs if s['patient_id'] in set(train_p)]
    val_subs = [s for s in subs if s['patient_id'] in set(val_p)]
    test_subs = [s for s in subs if s['patient_id'] in set(test_p)]
    print(f'split: train={len(train_subs)} subs / val={len(val_subs)} / test={len(test_subs)}')
    print(f'       patients: train={len(train_p)} / val={len(val_p)} / test={len(test_p)}')

    model = train_model(
        cfg, {'train': train_subs, 'val': val_subs},
        run_dir=run_dir, device=device,
    )
    save_test_artifacts(model, test_subs, cfg, run_dir=run_dir, device=device)
    _rd = run_dir.resolve()
    print(f'\ndone. artifacts in {_rd.relative_to(ROOT) if _rd.is_relative_to(ROOT) else _rd}')


if __name__ == '__main__':
    main()
