"""load and split niche-join parquets for alignment training.

extracted from scripts/align.py:119-218. the arithmetic is unchanged.
callable from notebooks and tests without importing protocols or graphs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _as_array(cell, dtype=np.float32) -> np.ndarray:
    return np.asarray(cell, dtype=dtype)


def load_subarray(
    path: Path,
    st_features: tuple[str, ...],
) -> dict:
    """load one niche-join parquet, materialise feature tensors.

    source: scripts/align.py:124-176
    """
    df = pd.read_parquet(path)
    he_niche = np.stack([_as_array(v) for v in df["virchow2_niche"]])
    he_tokens = np.stack(
        [_as_array(v).reshape(7, 1280) for v in df["virchow2_cell_tokens"]]
    )

    st_parts = []
    for feat in st_features:
        arr = np.stack([_as_array(v) for v in df[feat]])
        st_parts.append(arr)
    x_st = np.concatenate(st_parts, axis=1)

    mc_w = np.stack([_as_array(v) for v in df["mc_weights_niche"]])
    nan_mcw = np.isnan(mc_w).any(axis=1)
    if nan_mcw.any():
        mc_w[nan_mcw] = 1.0 / mc_w.shape[1]

    valid = (
        ~np.isnan(he_niche).any(axis=1)
        & ~np.isnan(he_tokens.reshape(len(he_tokens), -1)).any(axis=1)
        & ~np.isnan(x_st).any(axis=1)
    )

    pid = df["patient_id"].iloc[0]
    arch = df["archetype"].iloc[0]
    return {
        "spot_id": df.index.values[valid],
        "patient_id": int(pid) if pd.notna(pid) else None,
        "archetype": int(arch) if pd.notna(arch) else None,
        "compartment": df["compartment"].values[valid],
        "subarray": str(df["subarray"].iloc[0]),
        "x_he_niche": he_niche[valid].astype(np.float32),
        "x_he_tokens": he_tokens[valid].astype(np.float32),
        "x_st": x_st[valid].astype(np.float32),
        "mc_w": mc_w[valid].astype(np.float32),
        "n_dropped_nan": int((~valid).sum()),
    }


def discover_subarrays(niches_dir: Path) -> list[Path]:
    """source: scripts/align.py:179-180"""
    return sorted(niches_dir.glob("*.parquet"))


def _try_stratify(arches: np.ndarray) -> np.ndarray | None:
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

    seed is passed explicitly to sklearn random_state - no reliance on
    global RNG state.

    source: scripts/align.py:191-217
    """
    by_patient: dict[int, int] = {}
    for sub in subarrays:
        by_patient[sub["patient_id"]] = sub["archetype"]
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