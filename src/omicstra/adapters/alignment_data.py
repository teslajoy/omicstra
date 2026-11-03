"""load, split and save niche-join parquets for alignment training.

extracted from scripts/align.py:119-218. the arithmetic is unchanged.
callable from notebooks and tests without importing protocols or graphs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _as_array(cell, dtype=np.float32) -> np.ndarray:
    return np.asarray(cell, dtype=dtype)


class AlignmentDeclarationMissing(RuntimeError):
    """a field the alignment adapter requires is absent or invalid."""


@dataclass(frozen=True)
class AlignmentContract:
    """resolved declarations for alignment data loading.

    every field that was a hardcoded string in the previous version of
    load_subarray is now a field here, resolved from the cohort's own
    declarations before any data is read.
    """
    he_niche_column: str
    he_tokens_column: str
    he_dim: int
    k_plus_1: int
    supervision_column: str
    subject_column: str
    stratification_column: str | None
    unit_label_columns: tuple[str, ...]
    sample_column: str = "subarray"

    @classmethod
    def from_project_dir(cls, project_dir: str | Path) -> AlignmentContract:
        """resolve alignment declarations from a cohort's project files."""
        d = Path(project_dir)
        if not d.is_dir():
            raise AlignmentDeclarationMissing(f"no directory at {d}")

        raw: dict = {}
        for name in ("cohort.json", "project.json"):
            p = d / name
            if p.is_file():
                try:
                    data = json.loads(p.read_text())
                    if isinstance(data, dict):
                        raw.update(data)
                except (OSError, ValueError):
                    pass
        if not raw:
            raise AlignmentDeclarationMissing(
                f"no cohort.json or project.json in {d}")

        problems: list[str] = []

        encoders = raw.get("encoders", [])
        if not encoders:
            problems.append("no encoders declared")

        he_enc = None
        for enc in encoders:
            if enc.get("role") in ("primary", "morphology"):
                he_enc = enc
                break
        if he_enc is None and encoders:
            problems.append(
                f"no encoder with role 'primary' or 'morphology'. "
                f"declared roles: {[e.get('role') for e in encoders]}")

        he_dim = he_enc.get("dim") if he_enc else None
        if he_enc and he_dim is None:
            problems.append(
                f"morphology encoder {he_enc.get('name')!r} declares no dim")

        k = raw.get("k_neighbors")
        if k is None:
            problems.append("no k_neighbors declared")

        supervision = raw.get("supervision")
        if not supervision:
            problems.append("no supervision declared")

        subject = raw.get("subject_id_column")
        if not subject:
            problems.append(
                "no subject_id_column declared (expected in cohort.json)")

        if problems:
            raise AlignmentDeclarationMissing(
                f"{len(problems)} declaration(s) missing or invalid:\n  "
                + "\n  ".join(problems))

        he_name = he_enc["name"]
        he_niche_col = f"{he_name}_niche"
        he_tokens_col = f"{he_name}_cell_tokens"

        strat = None
        join_inputs = raw.get("join_inputs", {})
        subj_spec = join_inputs.get("subject", {})
        subj_labels = subj_spec.get("labels", [])
        if subj_labels:
            strat = subj_labels[0]

        unit_labels: list[str] = []
        for label_name, label_spec in (join_inputs.get("labels") or {}).items():
            if label_spec.get("role") == "centre":
                if label_name != supervision:
                    unit_labels.append(label_name)

        return cls(
            he_niche_column=he_niche_col,
            he_tokens_column=he_tokens_col,
            he_dim=he_dim,
            k_plus_1=k + 1,
            supervision_column=supervision,
            subject_column=subject,
            stratification_column=strat,
            unit_label_columns=tuple(unit_labels),
        )

    @classmethod
    def for_fixture(
        cls,
        *,
        he_niche_column: str,
        he_tokens_column: str,
        he_dim: int,
        k_plus_1: int,
        supervision_column: str,
        subject_column: str,
        stratification_column: str | None = None,
        unit_label_columns: tuple[str, ...] = (),
    ) -> AlignmentContract:
        """build a contract directly for tests without project files."""
        return cls(
            he_niche_column=he_niche_column,
            he_tokens_column=he_tokens_column,
            he_dim=he_dim,
            k_plus_1=k_plus_1,
            supervision_column=supervision_column,
            subject_column=subject_column,
            stratification_column=stratification_column,
            unit_label_columns=unit_label_columns,
        )


def _coerce_id(val):
    if val is None or (hasattr(val, '__class__') and pd.isna(val)):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return val


def load_subarray(
    path: Path,
    st_features: tuple[str, ...],
    contract: AlignmentContract | None = None,
    *,
    project_dir: str | Path | None = None,
) -> dict:
    """load one niche-join parquet, materialise feature tensors.

    source: scripts/align.py:124-176. arithmetic unchanged.

    resolution order for the contract:
    1. explicit contract argument (tests, notebooks)
    2. resolved from project_dir declarations
    3. refused - no silent defaults
    """
    if contract is None:
        if project_dir is None:
            raise AlignmentDeclarationMissing(
                "load_subarray requires either a contract or a project_dir. "
                "the adapter does not default to any cohort's column names")
        contract = AlignmentContract.from_project_dir(project_dir)

    df = pd.read_parquet(path)

    _require_columns(df, path, [
        contract.he_niche_column,
        contract.he_tokens_column,
        contract.supervision_column,
        contract.subject_column,
        contract.sample_column,
    ])

    he_niche = np.stack([_as_array(v) for v in df[contract.he_niche_column]])

    actual_niche_dim = he_niche.shape[1]
    if actual_niche_dim != contract.he_dim:
        raise AlignmentDeclarationMissing(
            f"{path.name}: declared he_dim={contract.he_dim} but "
            f"{contract.he_niche_column} has width {actual_niche_dim}")

    first_tok = _as_array(df[contract.he_tokens_column].iloc[0])
    expected_tok_len = contract.k_plus_1 * contract.he_dim
    if len(first_tok) != expected_tok_len:
        raise AlignmentDeclarationMissing(
            f"{path.name}: declared k_plus_1={contract.k_plus_1} x "
            f"he_dim={contract.he_dim} = {expected_tok_len} but "
            f"{contract.he_tokens_column} has length {len(first_tok)}")

    he_tokens = np.stack(
        [_as_array(v).reshape(contract.k_plus_1, contract.he_dim)
         for v in df[contract.he_tokens_column]]
    )

    st_parts = []
    for feat in st_features:
        if feat not in df.columns:
            raise AlignmentDeclarationMissing(
                f"{path.name}: st_feature {feat!r} not in columns. "
                f"available: {sorted(df.columns)}")
        arr = np.stack([_as_array(v) for v in df[feat]])
        st_parts.append(arr)
    x_st = np.concatenate(st_parts, axis=1)

    mc_w = np.stack([_as_array(v) for v in df[contract.supervision_column]])
    nan_mcw = np.isnan(mc_w).any(axis=1)
    if nan_mcw.any():
        mc_w[nan_mcw] = 1.0 / mc_w.shape[1]

    valid = (
        ~np.isnan(he_niche).any(axis=1)
        & ~np.isnan(he_tokens.reshape(len(he_tokens), -1)).any(axis=1)
        & ~np.isnan(x_st).any(axis=1)
    )

    pid = df[contract.subject_column].iloc[0]
    strat_val = None
    if contract.stratification_column and contract.stratification_column in df.columns:
        strat_val = df[contract.stratification_column].iloc[0]

    result = {
        "spot_id": df.index.values[valid],
        "patient_id": _coerce_id(pid),
        "archetype": _coerce_id(strat_val),
        "subarray": str(df[contract.sample_column].iloc[0]),
        "x_he_niche": he_niche[valid].astype(np.float32),
        "x_he_tokens": he_tokens[valid].astype(np.float32),
        "x_st": x_st[valid].astype(np.float32),
        "mc_w": mc_w[valid].astype(np.float32),
        "n_dropped_nan": int((~valid).sum()),
    }

    for label_col in contract.unit_label_columns:
        if label_col in df.columns:
            result[label_col] = df[label_col].values[valid]

    if "compartment" not in result and "compartment" in df.columns:
        result["compartment"] = df["compartment"].values[valid]

    return result


def _require_columns(df: pd.DataFrame, path: Path, columns: list[str]):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise AlignmentDeclarationMissing(
            f"{path.name}: missing columns {missing}. "
            f"available: {sorted(df.columns)}")


def build_matrices(
    subs: list[dict],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """stack H&E niche and ST features across subarrays, with per-niche metadata.

    source: scripts/align_classical.py:55-70
    """
    X_he = np.concatenate([s["x_he_niche"] for s in subs])
    X_st = np.concatenate([s["x_st"] for s in subs])
    meta_rows = []
    for s in subs:
        n = len(s["spot_id"])
        compartment = s.get("compartment", np.full(n, None))
        for sid, comp in zip(s["spot_id"], compartment):
            meta_rows.append({
                "spot_id": sid,
                "subarray": s["subarray"],
                "patient_id": s["patient_id"],
                "archetype": s["archetype"],
                "compartment": comp,
            })
    meta = pd.DataFrame(meta_rows)
    assert len(meta) == X_he.shape[0] == X_st.shape[0], (
        f"row mismatch: meta={len(meta)}, X_he={X_he.shape[0]}, X_st={X_st.shape[0]}")
    return X_he, X_st, meta


def save_test_artifacts(
    emb_df: pd.DataFrame,
    Z_he: np.ndarray,
    Z_st: np.ndarray,
    X_he: np.ndarray,
    X_st: np.ndarray,
    *,
    run_dir: Path,
    ks: tuple[int, ...] = (1, 5, 10),
    negative_ratio: int = 10,
    negative_seed: int = 0,
) -> dict:
    """write embeddings_test.parquet and metrics_h1_raw.json.

    source: scripts/align.py:560-577
    """
    from omicstra.measures.evaluation import linear_cka, retrieval

    emb_df.to_parquet(run_dir / "embeddings_test.parquet")

    sim = (Z_he @ Z_st.T).astype(np.float64)
    metrics = retrieval(sim, ks=ks, negative_ratio=negative_ratio,
                        negative_seed=negative_seed)
    metrics["cka_before"] = linear_cka(X_he, X_st, centre=True)
    metrics["cka_after"] = linear_cka(Z_he, Z_st, centre=True)
    metrics["test_subarrays"] = len(set(emb_df["subarray"])) if "subarray" in emb_df.columns else 0
    metrics["test_niches"] = len(emb_df)
    (run_dir / "metrics_h1_raw.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def discover_subarrays(niches_dir: Path) -> list[Path]:
    """source: scripts/align.py:179-180"""
    return sorted(niches_dir.glob("*.parquet"))


def _try_stratify(arches: np.ndarray) -> np.ndarray | None:
    if arches.dtype == object and all(v is None for v in arches):
        return None
    try:
        unique, counts = np.unique(arches, return_counts=True)
    except TypeError:
        return None
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
