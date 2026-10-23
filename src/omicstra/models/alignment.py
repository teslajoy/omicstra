"""neural alignment models and loss functions. requires torch (encode extra).

extracted from scripts/align.py. the arithmetic is unchanged.

two fusion architectures
------------------------
    LateFusion      independent MLP streams per modality -> shared space
    CrossAttnFusion ST-conditioned aggregation of H&E values; no direct
                    ST residual

four contrastive losses, verified against configs/v3/
-----------------------------------------------------
    InfoNCE         symmetric CLIP-style (R1, R4, R6)
    SupCon          soft-supervised via mc_weights cosine (R2)
    Barlow Twins    cross-correlation -> identity (R3)
    AnInfoNCE       per-dim learnable Mahalanobis bilinear (R5)

the caller is responsible for data loading, splitting, training
orchestration, seeding, and metric computation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """LN -> Linear -> ReLU -> BN -> Dropout -> Linear -> L2-norm.

    source: scripts/align.py:224-240
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.drop(x)
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


class LateFusion(nn.Module):
    """independent two-stream MLPs -> shared space.

    anisotropic=True registers a per-dim learnable log-scale parameter
    for AnInfoNCE. init 0 -> uniform scale = standard InfoNCE at step 0.

    forward(x_he_niche, x_he_tokens, x_st) -> (z_he, z_st, None).
    x_he_tokens is accepted but unused (late fusion aggregates at the
    niche level). the signature matches CrossAttnFusion for dispatch.

    source: scripts/align.py:243-266
    """
    def __init__(self, he_dim: int, st_dim: int, shared_dim: int,
                 dropout: float, anisotropic: bool = False):
        super().__init__()
        self.he_mlp = MLPBlock(he_dim, shared_dim, dropout)
        self.st_mlp = MLPBlock(st_dim, shared_dim, dropout)
        if anisotropic:
            self.aniso_log_scale = nn.Parameter(torch.zeros(shared_dim))
        else:
            self.aniso_log_scale = None

    def forward(self, x_he_niche: torch.Tensor, x_he_tokens: torch.Tensor,
                x_st: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        z_he = self.he_mlp(x_he_niche)
        z_st = self.st_mlp(x_st)
        return z_he, z_st, None


class CrossAttnFusion(nn.Module):
    """ST-conditioned aggregation of H&E values; no direct ST residual.

    ST query shapes attention weights over H&E tile tokens. z_he is
    derived from the attended H&E values only.

    forward(x_he_niche, x_he_tokens, x_st) -> (z_he, z_st, attn).

    source: scripts/align.py:269-298
    """
    def __init__(self, he_tile_dim: int, st_dim: int, shared_dim: int,
                 dropout: float):
        super().__init__()
        self.tile_ln = nn.LayerNorm(he_tile_dim)
        self.st_ln = nn.LayerNorm(st_dim)
        self.st_proj = nn.Linear(st_dim, shared_dim)
        self.k_proj = nn.Linear(he_tile_dim, shared_dim)
        self.v_proj = nn.Linear(he_tile_dim, shared_dim)
        self.out_ln = nn.LayerNorm(shared_dim)
        self.he_out = nn.Linear(shared_dim, shared_dim)
        self.st_out = nn.Linear(shared_dim, shared_dim)
        self.shared = shared_dim

    def forward(self, x_he_niche: torch.Tensor, x_he_tokens: torch.Tensor,
                x_st: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tile_tokens = self.tile_ln(x_he_tokens)
        st_ln = self.st_ln(x_st)
        st_query = self.st_proj(st_ln).unsqueeze(1)
        K = self.k_proj(tile_tokens)
        V = self.v_proj(tile_tokens)
        scores = torch.matmul(st_query, K.transpose(-1, -2)) / (self.shared ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, V).squeeze(1)
        z_he = F.normalize(self.he_out(self.out_ln(attn_out)), dim=-1)
        z_st = F.normalize(self.st_out(st_query.squeeze(1)), dim=-1)
        return z_he, z_st, attn.squeeze(1)


def build_model(fusion: str, *, he_dim: int, st_dim: int, shared_dim: int,
                dropout: float, he_tile_dim: int | None = None,
                anisotropic: bool = False) -> nn.Module:
    """factory dispatch on fusion string. cross_attn requires he_tile_dim;
    anisotropic only for late fusion.

    source: scripts/align.py:301-312
    """
    if fusion == "late":
        return LateFusion(he_dim, st_dim, shared_dim, dropout,
                          anisotropic=anisotropic)
    if fusion == "cross_attn":
        if anisotropic:
            raise ValueError(
                "anisotropic=True is only wired for late fusion (R5). "
                "cross_attn + aninfonce is a separate experiment.")
        if he_tile_dim is None:
            raise ValueError("cross_attn fusion requires he_tile_dim")
        return CrossAttnFusion(he_tile_dim, st_dim, shared_dim, dropout)
    raise ValueError(f"unsupported fusion: {fusion!r}")


# --- loss functions -----------------------------------------------------------

def infonce_loss(z_he: torch.Tensor, z_st: torch.Tensor, *, tau: float
                 ) -> torch.Tensor:
    """symmetric InfoNCE (CLIP-style). matched pairs on diagonal.

    source: scripts/align.py:319-326
    """
    logits = z_he @ z_st.T / tau
    N = logits.size(0)
    targets = torch.arange(N, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets)
                  + F.cross_entropy(logits.T, targets))


def aninfonce_loss(z_he: torch.Tensor, z_st: torch.Tensor,
                   aniso_log_scale: torch.Tensor, *, tau: float
                   ) -> torch.Tensor:
    """anisotropic InfoNCE. per-dim learnable scaling on the bilinear form.

    bilinear: z_he @ diag(exp(2 * log_scale)) @ z_st.T.
    log_scale = 0 -> uniform scale -> identical to standard InfoNCE.

    source: scripts/align.py:329-351
    """
    if aniso_log_scale is None:
        raise ValueError(
            "aninfonce requires anisotropic=True on the fusion model")
    scale_sq = (2.0 * aniso_log_scale).exp()
    logits = (z_he * scale_sq.unsqueeze(0)) @ z_st.T / tau
    N = logits.size(0)
    targets = torch.arange(N, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets)
                  + F.cross_entropy(logits.T, targets))


def supcon_loss(z_he: torch.Tensor, z_st: torch.Tensor,
                mc_w: torch.Tensor, *, tau: float, tau_target: float
                ) -> torch.Tensor:
    """soft-supervised contrastive via mc_weights cosine similarity.

    source: scripts/align.py:354-365
    """
    mc_norm = F.normalize(mc_w.float(), dim=-1)
    sim_target = mc_norm @ mc_norm.T
    target_dist = F.softmax(sim_target / tau_target, dim=-1)
    logits = z_he @ z_st.T / tau
    log_pred_he = F.log_softmax(logits, dim=-1)
    log_pred_st = F.log_softmax(logits.T, dim=-1)
    loss_he = -(target_dist * log_pred_he).sum(dim=-1).mean()
    loss_st = -(target_dist * log_pred_st).sum(dim=-1).mean()
    return 0.5 * (loss_he + loss_st)


def barlow_loss(z_he: torch.Tensor, z_st: torch.Tensor, *, lam: float
                ) -> torch.Tensor:
    """barlow twins: cross-correlation matrix -> identity.

    source: scripts/align.py:368-380
    """
    zc_he = z_he - z_he.mean(0, keepdim=True)
    zc_st = z_st - z_st.mean(0, keepdim=True)
    zc_he = zc_he / (zc_he.std(0, keepdim=True) + 1e-8)
    zc_st = zc_st / (zc_st.std(0, keepdim=True) + 1e-8)
    N, D = z_he.shape
    C = (zc_he.T @ zc_st) / N
    on_diag = (torch.diagonal(C) - 1.0).pow(2).sum()
    off_diag = (C - torch.diag(torch.diagonal(C))).pow(2).sum()
    return on_diag + lam * off_diag


def compute_loss(loss: str, z_he: torch.Tensor, z_st: torch.Tensor, *,
                 mc_w: torch.Tensor | None = None,
                 model: nn.Module | None = None,
                 tau: float = 0.07, tau_target: float = 0.1,
                 lam: float = 5e-3) -> torch.Tensor:
    """dispatch on loss name. defaults match the published grid.

    source: scripts/align.py:383-394
    """
    if loss == "infonce":
        return infonce_loss(z_he, z_st, tau=tau)
    if loss == "aninfonce":
        if model is None:
            raise ValueError("aninfonce needs model to read aniso_log_scale")
        return aninfonce_loss(z_he, z_st, model.aniso_log_scale, tau=tau)
    if loss == "supcon":
        if mc_w is None:
            raise ValueError("supcon needs mc_w")
        return supcon_loss(z_he, z_st, mc_w, tau=tau, tau_target=tau_target)
    if loss == "barlow":
        return barlow_loss(z_he, z_st, lam=lam)
    raise ValueError(f"unsupported loss: {loss!r}")


__all__ = [
    "MLPBlock", "LateFusion", "CrossAttnFusion", "build_model",
    "infonce_loss", "aninfonce_loss", "supcon_loss", "barlow_loss",
    "compute_loss",
]
