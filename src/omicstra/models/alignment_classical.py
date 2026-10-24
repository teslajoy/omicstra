"""classical linear-alignment baselines. numpy and scikit-learn; no torch dependency.

package implementation extracted from the preserved reference script
(scripts/align_classical.py, 260 lines). the arithmetic is unchanged;
what changes is that each method is callable from notebooks and tests
without running a script.

three baselines
---------------
    B1  CCA         ridge-regularised covariance-whitening CCA via SVD
    B2  Procrustes  PCA each modality, then orthogonal Procrustes rotation
    B3  Unaligned   PCA each modality independently, L2-norm, no alignment

all three fit on training arrays and apply to test arrays. the caller is
responsible for enforcing a shared patient split across methods and for
ensuring paired row ordering between modalities.

returns test embeddings only. a fitted-model return that allows
transforming new samples without refitting is a follow-up.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def l2_normalize(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row. rows with norm below eps are left unchanged
    (not made unit length) to preserve the original script behavior."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return (x / n).astype(np.float32)


def _validate_array(X: np.ndarray, name: str) -> None:
    """require a 2-d numeric array with at least one feature."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name}: expected ndarray, got {type(X).__name__}")
    if X.ndim != 2:
        raise ValueError(f"{name}: expected 2-d array, got {X.ndim}-d")
    if X.shape[1] == 0:
        raise ValueError(f"{name}: zero features")
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError(f"{name}: expected numeric dtype, got {X.dtype}")


def _validate_paired(X_he: np.ndarray, X_st: np.ndarray, label: str) -> None:
    if X_he.shape[0] != X_st.shape[0]:
        raise ValueError(
            f"{label}: H&E has {X_he.shape[0]} rows, ST has {X_st.shape[0]}. "
            "the two modalities must have the same number of paired rows.")
    if X_he.shape[0] == 0:
        raise ValueError(f"{label}: zero rows")


def _validate_finite(X: np.ndarray, name: str) -> None:
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).any(axis=1).sum())
        raise ValueError(
            f"{name}: {bad} row(s) with non-finite values out of {X.shape[0]}")


def _validate_dims(X_train: np.ndarray, X_test: np.ndarray, name: str) -> None:
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"{name}: train has {X_train.shape[1]} features, "
            f"test has {X_test.shape[1]}")


def _validate_shared_dim(shared_dim: int, method: str) -> None:
    if not isinstance(shared_dim, int) or shared_dim < 1:
        raise ValueError(
            f"{method}: shared_dim must be a positive integer, got {shared_dim}")


def _validate_reg(reg: float) -> None:
    if not (np.isfinite(reg) and reg > 0):
        raise ValueError(f"reg must be finite and positive, got {reg}")


def run_cca(
    X_he_train: np.ndarray,
    X_st_train: np.ndarray,
    X_he_test: np.ndarray,
    X_st_test: np.ndarray,
    shared_dim: int,
    *,
    reg: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, int]:
    """ridge-regularised covariance-whitening CCA via SVD.

    projects both modalities into a shared space where their canonical
    correlations are maximised. reg stabilises covariance inversion.

    the returned k_used is min(shared_dim, D_he, D_st, N_train - 1),
    which may be less than shared_dim when the data rank is limiting.

    returns (z_he_test, z_st_test, k_used).

    source: scripts/align_classical.py:79-114
    """
    _validate_reg(reg)
    _validate_shared_dim(shared_dim, "CCA")
    for a, n in ((X_he_train, "CCA X_he_train"), (X_st_train, "CCA X_st_train"),
                 (X_he_test, "CCA X_he_test"), (X_st_test, "CCA X_st_test")):
        _validate_array(a, n)
        _validate_finite(a, n)
    _validate_paired(X_he_train, X_st_train, "CCA train")
    _validate_paired(X_he_test, X_st_test, "CCA test")
    _validate_dims(X_he_train, X_he_test, "CCA H&E")
    _validate_dims(X_st_train, X_st_test, "CCA ST")

    k = min(shared_dim, X_he_train.shape[1], X_st_train.shape[1],
            X_he_train.shape[0] - 1)

    Xc = X_he_train - X_he_train.mean(0, keepdims=True)
    Yc = X_st_train - X_st_train.mean(0, keepdims=True)
    N = Xc.shape[0]
    Cxx = (Xc.T @ Xc) / N + reg * np.eye(Xc.shape[1], dtype=np.float64)
    Cyy = (Yc.T @ Yc) / N + reg * np.eye(Yc.shape[1], dtype=np.float64)
    Cxy = (Xc.T @ Yc) / N

    def whiten(C: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, reg)
        return vecs @ np.diag(vals ** -0.5) @ vecs.T

    Wx = whiten(Cxx.astype(np.float64))
    Wy = whiten(Cyy.astype(np.float64))
    M = Wx @ Cxy.astype(np.float64) @ Wy
    U, _S, Vt = np.linalg.svd(M, full_matrices=False)
    A = Wx @ U[:, :k]
    B = Wy @ Vt[:k, :].T

    mu_he = X_he_train.mean(0, keepdims=True)
    mu_st = X_st_train.mean(0, keepdims=True)
    z_he = (X_he_test - mu_he) @ A
    z_st = (X_st_test - mu_st) @ B
    return l2_normalize(z_he), l2_normalize(z_st), k


def run_procrustes(
    X_he_train: np.ndarray,
    X_st_train: np.ndarray,
    X_he_test: np.ndarray,
    X_st_test: np.ndarray,
    shared_dim: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """PCA each modality to shared_dim, then orthogonal Procrustes.

    fits X_he_pca @ R ~ X_st_pca. test embeddings in the ST basis.

    source: scripts/align_classical.py:117-133
    """
    _validate_shared_dim(shared_dim, "Procrustes")
    for a, n in ((X_he_train, "Procrustes X_he_train"), (X_st_train, "Procrustes X_st_train"),
                 (X_he_test, "Procrustes X_he_test"), (X_st_test, "Procrustes X_st_test")):
        _validate_array(a, n)
        _validate_finite(a, n)
    _validate_paired(X_he_train, X_st_train, "Procrustes train")
    _validate_paired(X_he_test, X_st_test, "Procrustes test")
    _validate_dims(X_he_train, X_he_test, "Procrustes H&E")
    _validate_dims(X_st_train, X_st_test, "Procrustes ST")

    pca_he = PCA(n_components=shared_dim, random_state=42)
    pca_st = PCA(n_components=shared_dim, random_state=42)
    H_tr = pca_he.fit_transform(X_he_train)
    S_tr = pca_st.fit_transform(X_st_train)
    U, _, Vt = np.linalg.svd(H_tr.T @ S_tr, full_matrices=False)
    R = U @ Vt
    H_te = pca_he.transform(X_he_test) @ R
    S_te = pca_st.transform(X_st_test)
    return l2_normalize(H_te), l2_normalize(S_te), shared_dim


def run_unaligned(
    X_he_train: np.ndarray,
    X_st_train: np.ndarray,
    X_he_test: np.ndarray,
    X_st_test: np.ndarray,
    shared_dim: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """PCA each modality independently to shared_dim, L2-norm, no alignment.

    unaligned baseline: each modality is reduced to the same dimensionality
    but the PCA bases are fitted independently with no cross-modal signal.
    cross-modal cosine similarity in this space depends on the arbitrary
    correspondence between independently chosen PCA axes.

    source: scripts/align_classical.py:136-150
    """
    _validate_shared_dim(shared_dim, "Unaligned")
    for a, n in ((X_he_train, "Unaligned X_he_train"), (X_st_train, "Unaligned X_st_train"),
                 (X_he_test, "Unaligned X_he_test"), (X_st_test, "Unaligned X_st_test")):
        _validate_array(a, n)
        _validate_finite(a, n)
    _validate_paired(X_he_train, X_st_train, "Unaligned train")
    _validate_paired(X_he_test, X_st_test, "Unaligned test")
    _validate_dims(X_he_train, X_he_test, "Unaligned H&E")
    _validate_dims(X_st_train, X_st_test, "Unaligned ST")

    pca_he = PCA(n_components=shared_dim, random_state=42).fit(X_he_train)
    pca_st = PCA(n_components=shared_dim, random_state=42).fit(X_st_train)
    return (
        l2_normalize(pca_he.transform(X_he_test)),
        l2_normalize(pca_st.transform(X_st_test)),
        shared_dim,
    )


BASELINES = {"cca": run_cca, "procrustes": run_procrustes, "unaligned": run_unaligned}

__all__ = [
    "BASELINES", "l2_normalize",
    "run_cca", "run_procrustes", "run_unaligned",
]
