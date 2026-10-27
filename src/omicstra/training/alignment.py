"""contrastive alignment training and test-artifact generation.

extracted from scripts/align.py:457-577. uses models.alignment for
architectures and losses, measures.evaluation for metrics. callable from
notebooks without importing protocols or graphs.

seed handling: train_model calls set_seed at the top. callers that run
in a subprocess or resume from a checkpoint call set_seed themselves
before invoking these functions - global RNG state is not inherited
across process boundaries.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from omicstra.measures.evaluation import linear_cka, retrieval
from omicstra.models.alignment import build_model, compute_loss
from omicstra.seed import set_seed


def epoch_pass(
    model: torch.nn.Module,
    subs: list[dict],
    loss_name: str,
    *,
    train: bool,
    optim: torch.optim.Optimizer | None = None,
    device: str = "cpu",
    tau: float = 0.07,
    tau_target: float = 0.1,
    lam: float = 5e-3,
) -> tuple[float, float, float]:
    """one pass over subarrays. returns (mean_loss, mean_cos_sim, mean_r1).

    source: scripts/align.py:457-484
    """
    model.train() if train else model.eval()
    losses, cos_sims, r1s = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for sub in subs:
            x_he_niche = torch.from_numpy(sub["x_he_niche"]).to(device)
            x_he_tokens = torch.from_numpy(sub["x_he_tokens"]).to(device)
            x_st = torch.from_numpy(sub["x_st"]).to(device)
            mc_w = torch.from_numpy(sub["mc_w"]).to(device) if loss_name == "supcon" else None
            z_he, z_st, _ = model(x_he_niche, x_he_tokens, x_st)
            loss = compute_loss(
                loss_name, z_he, z_st,
                mc_w=mc_w, model=model,
                tau=tau, tau_target=tau_target, lam=lam,
            )
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            losses.append(loss.item())
            with torch.no_grad():
                cos_sims.append(float((z_he * z_st).sum(-1).mean()))
                sim = z_he @ z_st.T
                n = sim.size(0)
                ranks = (-sim).argsort(dim=1).argsort(dim=1)[
                    torch.arange(n), torch.arange(n)
                ]
                r1s.append(float((ranks < 1).float().mean()))
    return float(np.mean(losses)), float(np.mean(cos_sims)), float(np.mean(r1s))


def train_model(
    *,
    he_dim: int,
    st_dim: int,
    shared_dim: int = 512,
    dropout: float = 0.3,
    fusion: str = "late",
    anisotropic: bool = False,
    he_tile_dim: int = 1280,
    loss_name: str = "infonce",
    lr: float = 5e-4,
    weight_decay: float = 1e-3,
    epochs: int = 50,
    patience: int = 10,
    seed: int = 42,
    device: str = "cpu",
    train_subs: list[dict],
    val_subs: list[dict],
    run_dir: Path,
    tau: float = 0.07,
    tau_target: float = 0.1,
    lam: float = 5e-3,
) -> torch.nn.Module:
    """train one alignment run. writes checkpoint.pt and training_log.csv.

    calls set_seed(seed) before model init - does not assume inherited
    global RNG state.

    source: scripts/align.py:487-520
    """
    set_seed(seed)
    model = build_model(
        fusion, he_dim=he_dim, st_dim=st_dim, shared_dim=shared_dim,
        dropout=dropout, he_tile_dim=he_tile_dim if fusion == "cross_attn" else None,
        anisotropic=anisotropic,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_kw = {"tau": tau, "tau_target": tau_target, "lam": lam}
    log_path = run_dir / "training_log.csv"
    best_val = -float("inf")
    bad_epochs = 0
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_cos_sim", "val_r1", "lr"])
        for ep in range(epochs):
            random.shuffle(train_subs)
            tr_loss, _, _ = epoch_pass(
                model, train_subs, loss_name, train=True, optim=optim,
                device=device, **loss_kw,
            )
            va_loss, va_cos, va_r1 = epoch_pass(
                model, val_subs, loss_name, train=False, device=device,
                **loss_kw,
            )
            lr_now = optim.param_groups[0]["lr"]
            writer.writerow([ep, tr_loss, va_loss, va_cos, va_r1, lr_now])
            f.flush()
            if va_cos > best_val:
                best_val = va_cos
                bad_epochs = 0
                torch.save(model.state_dict(), run_dir / "checkpoint.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", weights_only=True))
    return model


def embed_test(
    model: torch.nn.Module,
    test_subs: list[dict],
    *,
    device: str = "cpu",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """run model on test subarrays, return embeddings + pre-projection arrays.

    returns (emb_df, Z_he, Z_st, X_he_raw, X_st_raw). no file I/O.

    source: scripts/align.py:527-558
    """
    model.eval()
    rows = []
    all_z_he, all_z_st = [], []
    all_he_niche, all_st = [], []
    with torch.no_grad():
        for sub in test_subs:
            x_he_niche = torch.from_numpy(sub["x_he_niche"]).to(device)
            x_he_tokens = torch.from_numpy(sub["x_he_tokens"]).to(device)
            x_st = torch.from_numpy(sub["x_st"]).to(device)
            z_he, z_st, attn = model(x_he_niche, x_he_tokens, x_st)
            z_he_np = z_he.cpu().numpy().astype(np.float32)
            z_st_np = z_st.cpu().numpy().astype(np.float32)
            attn_np = attn.cpu().numpy().astype(np.float32) if attn is not None else None
            for i, sid in enumerate(sub["spot_id"]):
                row = {
                    "spot_id": sid,
                    "subarray": sub["subarray"],
                    "patient_id": sub["patient_id"],
                    "archetype": sub["archetype"],
                    "compartment": sub["compartment"][i],
                    "z_he": z_he_np[i].tolist(),
                    "z_st": z_st_np[i].tolist(),
                }
                if attn_np is not None:
                    row["attention_weights"] = attn_np[i].tolist()
                rows.append(row)
            all_z_he.append(z_he_np)
            all_z_st.append(z_st_np)
            all_he_niche.append(sub["x_he_niche"])
            all_st.append(sub["x_st"])

    emb_df = pd.DataFrame(rows).set_index("spot_id")
    Z_he = np.concatenate(all_z_he)
    Z_st = np.concatenate(all_z_st)
    X_he = np.concatenate(all_he_niche)
    X_st = np.concatenate(all_st)
    return emb_df, Z_he, Z_st, X_he, X_st


from omicstra.adapters.alignment_data import save_test_artifacts as save_test_artifacts