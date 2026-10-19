"""the statistics the routing matrix cites. pure functions over arrays.

the maths, not the order and not the authority. nothing here reads a file,
knows a run id, or names a cohort, and every function is callable from a
notebook with two numpy arrays - which is the point: a statistic a notebook
cannot check is a statistic nobody checked.

ported from the code that produced the published numbers, against
`script_map/evaluation_maths.md`. the arithmetic is unchanged. what changes is
that each statistic now exists ONCE, and that the parameters the scripts left
implicit are arguments a caller must supply.

three machines, and everything else reduces to them
---------------------------------------------------
    permutation_test    z and p from a null distribution. the caller NAMES the
                        tail, because the one- and two-sided tails of one test
                        were once reported as two findings
    bh_fdr              the family is an argument. "corrected across the family
                        tested" is only true if someone said what the family was
    bootstrap_ci        resampled by a DECLARED GROUP. a unit bootstrap over
                        many units from few subjects gives intervals far too
                        narrow, because the units are not independent

precision is part of the method for an iterative estimator
----------------------------------------------------------
the closed-form statistics here accumulate in float64 - CKA, the canonical
correlation, the permutation moments - because for a fixed formula float64 is
strictly more accurate and the published float32 values differ from it only in
the last bits.

the iterative ones do NOT upcast. a seed makes KMeans deterministic at a given
precision and not across precisions: the seed cohort's embeddings give the
published ARI as float32 and a different ARI as float64, same seed and same
n_init. so `cluster_agreement`, `linear_probe` and `knn_purity` take the
caller's dtype and record it, and a port that tidied the dtype would have
changed a published number with nothing to show for it.

what a caller must declare
--------------------------
every cohort-dependent parameter is a required argument with no default: the
seed, the number of clusters, the number of resamples, the permutation count,
the negative ratio, alpha. a default here is a choice made on one tissue that
then travels silently, which is what `EVALUATION.md` section 8 forbids. the one
exception is an epsilon guarding a division, which is numerical hygiene rather
than a decision about biology.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

Tail = Literal["greater", "less", "two_sided"]


class NotComparable(ValueError):
    """two numbers are about to be compared that were not computed the same way.

    raised rather than returned, because the result of ignoring it looks like a
    finding. the CKA case is the live example: one implementation took
    pre-centred input under the same name as one that centres internally, and
    handed raw data it returned 0.96 where the others returned 0.054.
    """


# ============================================================ machine 1 ====
def permutation_test(observed: float, nulls: Sequence[float] | np.ndarray, *,
                     tail: Tail) -> dict[str, Any]:
    """z and p for one statistic against its own permutation null.

    `p = (1 + #{null beyond observed}) / (1 + n)`. the `1 +` on both sides is
    deliberate and is not a continuity fudge: a p of exactly 0 is not a
    measurement, it is a statement that 500 permutations did not reach the
    observation. the floor is therefore 1/(1+n) and is reported as such.

    `tail` has no default. the one- and two-sided tails of a single DAG test
    were published as 80.3% and 78% and read as two results, which is what a
    defaulted tail buys.

    NaN nulls are dropped - a permutation that could not be computed is absent,
    not zero - and an empty null or a NaN observation gives NaN rather than a
    number with no evidence under it.
    """
    if tail not in ("greater", "less", "two_sided"):
        raise ValueError(f"tail must be greater, less or two_sided, got {tail!r}")
    x = np.asarray(nulls, dtype=np.float64).ravel()
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n == 0 or observed is None or np.isnan(observed):
        return {"z": float("nan"), "p": float("nan"), "n_null": n, "tail": tail,
                "null_mean": float("nan"), "null_sd": float("nan"),
                "p_floor": float("nan"),
                "reason": "empty null" if n == 0 else "observed is nan"}

    obs = float(observed)
    mu, sd = float(x.mean()), float(x.std())
    if tail == "greater":
        beyond = int(np.sum(x >= obs))
    elif tail == "less":
        beyond = int(np.sum(x <= obs))
    else:
        beyond = int(np.sum(np.abs(x) >= abs(obs)))
    # a null with no spread cannot place the observation. nan says so; a large
    # z from a divide-by-almost-zero says the opposite and looks like a result.
    z = (obs - mu) / sd if sd > 0 else float("nan")
    return {"z": float(z), "p": float((1 + beyond) / (1 + n)), "n_null": n,
            "tail": tail, "null_mean": mu, "null_sd": sd,
            "p_floor": float(1 / (1 + n)),
            "reason": "" if sd > 0 else "null has zero spread"}


# ============================================================ machine 2 ====
def bh_fdr(pvalues: Sequence[float] | np.ndarray, *, alpha: float,
           family: Sequence[str] | None = None) -> dict[str, Any]:
    """Benjamini-Hochberg over the family the caller passes.

    the family is an ARGUMENT. "corrected across the family tested" is a claim
    about which tests were in the family, and the two decisions the published
    work made about membership - excluding an underpowered set and reporting it
    separately, and testing 395 nodes rather than 5 - are exactly that claim.
    a function that corrected over "whatever was in scope" could not state it.

    NaN p-values are excluded from the family rather than treated as 1.0: a test
    that could not be computed was not a test, and counting it inflates the
    denominator that every other q is divided by.
    """
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    names = list(family) if family is not None else [str(i) for i in range(p.size)]
    if len(names) != p.size:
        raise ValueError(f"{len(names)} family name(s) for {p.size} p-value(s)")

    testable = ~np.isnan(p)
    m = int(testable.sum())
    q = np.full(p.shape, np.nan)
    if m:
        idx = np.flatnonzero(testable)
        order = idx[np.argsort(p[idx], kind="stable")]
        ranked = p[order] * m / np.arange(1, m + 1)
        # the monotone step: a q never exceeds the q of a larger p
        q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)

    rejected = (q <= alpha) & testable
    return {"q": q.tolist(), "rejected": rejected.tolist(), "alpha": float(alpha),
            "n_family": m, "n_excluded_not_testable": int(p.size - m),
            "n_rejected": int(rejected.sum()),
            "family": names,
            "rejected_names": [n for n, r in zip(names, rejected) if r]}


# ============================================================ machine 3 ====
def bootstrap_ci(statistic: Callable[[np.ndarray], float],
                 groups: Sequence[Any], *, n_resamples: int, seed: int,
                 ci: float) -> dict[str, Any]:
    """a confidence interval by resampling GROUPS, not units.

    `groups` is one label per unit, and a resample draws group labels with
    replacement and then takes every unit of each drawn group, with
    multiplicity. that is the published procedure, and the reason for it is not
    convention: units within a subject are not independent, so resampling units
    treats one subject's many units as many observations and returns an interval
    far too narrow.

    `statistic` receives unit INDICES, so it can be any of the functions below
    without any of them knowing a bootstrap is happening.
    """
    g = np.asarray(groups)
    unique = np.unique(g)
    if unique.size < 2:
        return {"point": float(statistic(np.arange(g.size))),
                "lo": float("nan"), "hi": float("nan"), "ci": float(ci),
                "n_groups": int(unique.size), "n_resamples": 0,
                "n_resamples_nan": 0,
                "reason": "a bootstrap over one group resamples the same units"}

    index_of = {u: np.flatnonzero(g == u) for u in unique}
    rng = np.random.default_rng(seed)
    point = float(statistic(np.arange(g.size)))
    vals = []
    for _ in range(n_resamples):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([index_of[u] for u in drawn])
        v = statistic(idx)
        if not np.isnan(v):
            vals.append(float(v))
    if not vals:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "ci": float(ci), "n_groups": int(unique.size), "n_resamples": 0,
                "n_resamples_nan": int(n_resamples),
                "reason": "every resample produced nan"}
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    lo, hi = np.quantile(vals, [lo_q, hi_q])
    return {"point": point, "lo": float(lo), "hi": float(hi), "ci": float(ci),
            "n_groups": int(unique.size), "n_resamples": len(vals),
            "n_resamples_nan": int(n_resamples - len(vals)), "reason": ""}


# ====================================================== paired retrieval ====
def retrieval(similarity: np.ndarray, *, ks: Sequence[int], negative_ratio: int,
              negative_seed: int) -> dict[str, Any]:
    """R@k, MRR, median rank, alignment gap and AUC from one similarity matrix.

    `similarity[i, j]` is unit i in one modality against unit j in the other,
    so the diagonal is the matched pair. the matrix is assumed to be cosine
    similarities - the caller normalises, because a function that silently
    normalised would make `similarity` mean two things.

    three details carried from `align.py:401` deliberately rather than improved:

    the rank of self is a DOUBLE ARGSORT, and numpy's default sort is not
    stable, so units tied with the matched pair break arbitrarily. the published
    numbers have that property; `n_tied_at_self` reports how many units it could
    have affected, which is the thing the original did not record

    `median_rank` is 1-indexed while `R@k` thresholds a 0-indexed rank. both are
    reproduced as published and named here so a reader is not left to find it

    the AUC's negatives are SUBSAMPLED - `negative_ratio` per positive, drawn
    without replacement under `negative_seed`. both were implicit in the script,
    and the seed there was not the seed the rest of the chain used. they are
    required arguments here for that reason: an AUC over a different negative
    sample is a different number, and nothing in the artifact said which.
    """
    sim = np.asarray(similarity, dtype=np.float64)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise NotComparable(
            f"retrieval needs a square matched matrix, got {sim.shape}. the "
            "diagonal IS the matched pair, so a rectangular matrix has no "
            "matched pairs to find")
    n = sim.shape[0]
    ranks = (-sim).argsort(axis=1).argsort(axis=1)[np.arange(n), np.arange(n)]
    matched = np.diag(sim)
    tied = int(np.sum([(sim[i] == matched[i]).sum() > 1 for i in range(n)]))

    off = sim[~np.eye(n, dtype=bool)]
    gap = float(matched.mean() - off.mean())

    size = min(len(matched) * int(negative_ratio), off.size)
    neg = np.random.default_rng(negative_seed).choice(off, size=size, replace=False)
    labels = np.concatenate([np.ones(len(matched)), np.zeros(len(neg))])
    scores = np.concatenate([matched, neg])
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = float("nan")

    out = {"n_queries": n, "mrr": float((1.0 / (ranks + 1)).mean()),
           "median_rank": float(np.median(ranks + 1)),
           "alignment_gap": gap, "auc": auc,
           "n_negatives_sampled": int(size), "negative_ratio": int(negative_ratio),
           "negative_seed": int(negative_seed), "n_tied_at_self": tied}
    for k in ks:
        out[f"r{int(k)}"] = float((ranks < int(k)).mean())
    return out


# ==================================================== geometric agreement ====
def linear_cka(X: np.ndarray, Y: np.ndarray, *, centre: bool = True,
               eps: float = 1e-12) -> float:
    """linear CKA between two representations of the same units, in row order.

    `‖XᵀY‖²_F / (‖XᵀX‖_F · ‖YᵀY‖_F)`, the HSIC form.

    `centre` is in the signature because this is the statistic the published
    chain implemented three times, and one of the three took PRE-CENTRED input
    under the same function name as one that centred internally. handed raw data
    that one returns 0.96 where the others return 0.054 - same formula, 17x
    apart, no error. passing `centre=False` is now an explicit claim that the
    caller already centred, rather than a fact about which module they imported.
    """
    A = np.asarray(X, dtype=np.float64)
    B = np.asarray(Y, dtype=np.float64)
    if A.shape[0] != B.shape[0]:
        raise NotComparable(
            f"CKA compares two views of the SAME units in the same order: "
            f"{A.shape[0]} rows against {B.shape[0]}")
    if centre:
        A = A - A.mean(axis=0, keepdims=True)
        B = B - B.mean(axis=0, keepdims=True)
    cross = A.T @ B
    num = float(np.sum(cross * cross))
    den = float(np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B))
    return num / (den + eps)


# ================================================== clustering agreement ====
def cluster_agreement(X: np.ndarray, labels: Sequence[Any], *, k: int,
                      n_init: int, seed: int, silhouette_sample: int
                      ) -> dict[str, Any]:
    """KMeans(k) against a label: ARI, plus silhouette on a subsample.

    `k` has no default. the published chain defined this function twice with the
    same name and a different k - 9 against one label and 14 against another -
    and the cited artifact came from one of them while the other stayed live.

    the silhouette is subsampled TWICE, as published: the caller's rows are
    drawn first and sklearn samples again inside, with the same seed. reproduced
    rather than corrected, and named here so the next reader does not take it
    for a single subsample.

    **ARI is geometry-biased and is not a reading of biology on its own.** a
    manifold embedding scores low against KMeans' globular assumption while a
    linear probe on the same space is at parity - that is a measured finding on
    the seed cohort, not a caveat. report `linear_probe` beside it.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    # NOT upcast. for a closed-form statistic float64 is strictly more accurate,
    # but KMeans is iterative and a seed only makes it deterministic AT A GIVEN
    # PRECISION: the same data as float32 and float64 lands in different optima.
    # measured on the seed cohort - float32 gives the published 0.2339287118 and
    # float64 gives 0.2338499031, same seed, same n_init. so the caller's dtype
    # is part of the method here and is recorded rather than chosen.
    A = np.asarray(X)
    y = np.asarray(labels)
    if A.shape[0] != y.size:
        raise NotComparable(f"{A.shape[0]} rows against {y.size} label(s)")
    if np.unique(y).size < 2 or y.size < k:
        return {"ari": float("nan"), "silhouette": float("nan"), "n": int(y.size),
                "k": int(k), "reason": "fewer than two classes, or fewer units than k"}

    preds = KMeans(n_clusters=int(k), n_init=int(n_init),
                   random_state=int(seed)).fit_predict(A)
    n_sil = min(int(silhouette_sample), y.size)
    idx = np.random.default_rng(int(seed)).choice(y.size, n_sil, replace=False)
    sil = float(silhouette_score(A[idx], y[idx], sample_size=n_sil,
                                 random_state=int(seed)))
    return {"ari": float(adjusted_rand_score(y, preds)), "silhouette": sil,
            "n": int(y.size), "k": int(k), "n_silhouette": n_sil,
            "dtype": str(A.dtype),
            "note": "ARI is geometry-biased; read it beside a linear probe",
            "reason": ""}


# ============================================================ decodability ====
def linear_probe(X: np.ndarray, labels: Sequence[Any], groups: Sequence[Any], *,
                 seed: int, max_iter: int, held_out_groups: int) -> dict[str, Any]:
    """held-out accuracy of a linear classifier, split BY GROUP.

    the split is by group and not by unit, for the same reason the bootstrap
    resamples by group: units from one subject are near-copies, so a unit-level
    split puts the same tissue on both sides and reports memorisation.

    this is the companion to `cluster_agreement`, and on the seed cohort it is
    what showed a low ARI to be metric-induced rather than biology missing.
    """
    from sklearn.linear_model import LogisticRegression

    # the caller's dtype is preserved, for the reason `cluster_agreement`
    # records: a solver's path depends on precision, so upcasting here would
    # silently answer a different question than the one the artifact answered.
    A = np.asarray(X)
    y = np.asarray(labels)
    g = np.asarray(groups)
    if not (A.shape[0] == y.size == g.size):
        raise NotComparable(f"{A.shape[0]} rows, {y.size} labels, {g.size} groups")
    unique = np.unique(g)
    if unique.size <= held_out_groups:
        return {"accuracy": float("nan"), "n_train_groups": 0, "n_test_groups": 0,
                "reason": f"{unique.size} group(s) cannot yield a held-out split "
                          f"of {held_out_groups}"}

    rng = np.random.default_rng(int(seed))
    test_groups = set(rng.choice(unique, size=int(held_out_groups), replace=False))
    te = np.array([x in test_groups for x in g])
    tr = ~te
    if np.unique(y[tr]).size < 2:
        return {"accuracy": float("nan"), "reason": "one class in the train split"}

    clf = LogisticRegression(max_iter=int(max_iter), random_state=int(seed))
    clf.fit(A[tr], y[tr])
    return {"accuracy": float(clf.score(A[te], y[te])),
            "n_train": int(tr.sum()), "n_test": int(te.sum()), "dtype": str(A.dtype),
            "n_train_groups": int(unique.size - held_out_groups),
            "n_test_groups": int(held_out_groups),
            "split": "by group", "reason": ""}


def knn_purity(X: np.ndarray, labels: Sequence[Any], *, ks: Sequence[int]
               ) -> dict[str, Any]:
    """for each k, the mean fraction of a unit's k neighbours sharing its label.

    a local measure, so it answers a question ARI cannot: whether the label is
    recoverable in a neighbourhood even when the classes are not globular.
    """
    from sklearn.neighbors import NearestNeighbors

    A = np.asarray(X)
    y = np.asarray(labels)
    if A.shape[0] != y.size:
        raise NotComparable(f"{A.shape[0]} rows against {y.size} label(s)")
    out: dict[str, Any] = {"n": int(y.size), "dtype": str(A.dtype)}
    for k in ks:
        k = int(k)
        if y.size <= k:
            out[f"purity_k{k}"] = float("nan")
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(A)
        _, idx = nn.kneighbors(A)
        same = (y[idx[:, 1:]] == y[:, None])
        out[f"purity_k{k}"] = float(same.mean())
    return out


# ================================================================ contrast ====
def two_sample_z(a: Sequence[float], b: Sequence[float], *, tail: Tail
                 ) -> dict[str, Any]:
    """Welch's z between two groups of scores: unequal variances, unpooled.

    the matrix's categorical compartment row cites this, one-sided, and the
    statistic had no home in the catalogue, the routing contract or the guard
    library - the number was published and the test was nowhere. Welch rather
    than Student because two compartments have no reason to share a variance,
    and the groups are routinely very different sizes.
    """
    if tail not in ("greater", "less", "two_sided"):
        raise ValueError(f"tail must be greater, less or two_sided, got {tail!r}")
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if x.size < 2 or y.size < 2:
        return {"z": float("nan"), "n_a": int(x.size), "n_b": int(y.size),
                "tail": tail, "reason": "each group needs at least two values"}
    # ddof=1 on each group: this is a sample variance of each group, not a
    # population deviation, and the two differ by enough to matter at n ~ 20
    se = np.sqrt(x.var(ddof=1) / x.size + y.var(ddof=1) / y.size)
    if not se > 0:
        return {"z": float("nan"), "n_a": int(x.size), "n_b": int(y.size),
                "tail": tail, "reason": "both groups have zero variance"}
    z = float((x.mean() - y.mean()) / se)
    return {"z": z if tail != "less" else -z, "mean_a": float(x.mean()),
            "mean_b": float(y.mean()), "n_a": int(x.size), "n_b": int(y.size),
            "se": float(se), "tail": tail, "ddof": 1, "reason": ""}


# ============================================================= association ====
def top_canonical_correlation(Z: np.ndarray, Y: np.ndarray, *,
                              with_weights: bool = False):
    """the top canonical correlation between two blocks, via QR then SVD.

    this is a CCA, not a Pearson r on a chosen axis: it finds the
    one-dimensional direction in each block that maximises their correlation.

    ONE implementation, with a flag, because the published chain had two - a
    full version and a value-only twin used inside the permutation loop - and
    two implementations of a statistic are two chances for the loop and the
    point estimate to drift apart. the weights cost a solve, so the loop passes
    `with_weights=False` and still calls this function.
    """
    A = np.asarray(Z, dtype=np.float64)
    B = np.asarray(Y, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    if A.shape[0] != B.shape[0]:
        raise NotComparable(
            f"CCA relates two blocks over the SAME units: {A.shape[0]} rows "
            f"against {B.shape[0]}")

    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    Qa, Ra = np.linalg.qr(Ac, mode="reduced")
    Qb, Rb = np.linalg.qr(Bc, mode="reduced")
    U, S, Vt = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    # a canonical correlation is a cosine and cannot exceed 1. values a hair
    # above it are floating point, and clipping says so rather than letting a
    # 1.0000000002 travel into a z.
    top = float(np.clip(S[0], -1.0, 1.0))
    if not with_weights:
        return top
    def _solve(R, v):
        try:
            return np.linalg.solve(R, v)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(R, v, rcond=None)[0]
    return top, _solve(Ra, U[:, 0]), _solve(Rb, Vt[0])


__all__ = ["NotComparable", "bh_fdr", "bootstrap_ci", "cluster_agreement",
           "knn_purity", "linear_cka", "linear_probe", "permutation_test",
           "retrieval", "top_canonical_correlation", "two_sample_z"]
