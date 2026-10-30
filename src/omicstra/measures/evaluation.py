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

    # a non-finite entry surfaces from the SVD as "did not converge", which
    # reads as a numerical problem and is a data problem - a label table that
    # does not cover every unit, usually. the message has to say which.
    for nm, M in (("Z", A), ("Y", B)):
        if not np.isfinite(M).all():
            bad = int((~np.isfinite(M)).any(axis=1).sum())
            raise NotComparable(
                f"{nm} has {bad} row(s) with a non-finite value, out of "
                f"{M.shape[0]}. a canonical correlation cannot be computed over "
                "them, and imputing them would invent the signal being tested - "
                "drop them, or declare the signal unevaluable at this split")
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



def projected_correlation(Z: np.ndarray, Y: np.ndarray, a: np.ndarray,
                          b: np.ndarray) -> float:
    """the correlation of two blocks along DIRECTIONS FITTED ELSEWHERE.

    this is the other half of a transfer test, and it is not a CCA: the
    directions are given, so there is nothing to maximise and the answer is a
    plain Pearson r between the two projections. a CCA computed on the same
    units would refit and report the best possible correlation, which is the
    number a transfer test exists to avoid.

    both blocks are centred on their OWN rows. centring on the training rows
    instead would leak the training mean into the held-out score.
    """
    A, B = np.asarray(Z, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    if A.shape[0] != B.shape[0]:
        raise NotComparable(
            f"a projection relates two blocks over the SAME units: "
            f"{A.shape[0]} rows against {B.shape[0]}")
    u = (A - A.mean(axis=0, keepdims=True)) @ np.asarray(a, dtype=np.float64)
    v = (B - B.mean(axis=0, keepdims=True)) @ np.asarray(b, dtype=np.float64)
    u, v = np.ravel(u), np.ravel(v)
    if u.std() == 0 or v.std() == 0:
        return float("nan")
    return float(np.corrcoef(u, v)[0, 1])


def transfer_correlation(Z_train: np.ndarray, Y_train: np.ndarray,
                         Z_test: np.ndarray, Y_test: np.ndarray):
    """fit the canonical directions on train, score them on held-out units.

    PROVES, in `EVALUATION.md`'s terms: a signal carries to units the direction
    was not fitted on. the within-cohort CCA proves something much weaker and
    on the seed cohort proved it for every arm in the grid at once - ten of ten
    runs significant, which orders nothing. the split is what makes the number
    discriminate, so it is part of the statistic rather than a wrapper around it.

    returns (r_test, r_train, a, b). `r_train` is not decoration: it is the
    number `r_test` has to be read against, because a direction that fits train
    poorly and test well has found noise in both.
    """
    a_b = top_canonical_correlation(Z_train, Y_train, with_weights=True)
    _, a, b = a_b
    return (projected_correlation(Z_test, Y_test, a, b),
            projected_correlation(Z_train, Y_train, a, b), a, b)

def transfer_null_distribution(
    Z_train: np.ndarray,
    Y_train: np.ndarray,
    Z_test: np.ndarray,
    Y_test: np.ndarray,
    n_perms: int,
    rng: np.random.Generator,
) -> list[float]:
    """null distribution for the transfer test, cached factorization.

    for univariate Y the canonical direction is proportional to pinv(Z_c) @ y_c.
    the QR+SVD in `transfer_correlation` recomputes the full factorization of
    Z_train on every call; for a permutation loop where only Y_train changes,
    precomputing pinv(Z_train_c) once and doing a matmul per permutation is
    ~100x faster on 30k x 512 matrices.

    the directions and projected correlations are numerically equivalent to
    calling `transfer_correlation` with the same shuffled Y_train.
    """
    Zt = np.asarray(Z_train, dtype=np.float64)
    Zt_c = Zt - Zt.mean(axis=0, keepdims=True)
    Zt_pinv = np.linalg.pinv(Zt_c)

    Ze = np.asarray(Z_test, dtype=np.float64)
    Ze_c = Ze - Ze.mean(axis=0, keepdims=True)

    y_te = np.asarray(Y_test, dtype=np.float64).ravel()
    y_te_c = y_te - y_te.mean()
    yn_te = float(np.linalg.norm(y_te_c))

    y_tr = np.asarray(Y_train, dtype=np.float64).ravel()
    y_tr_c = y_tr - y_tr.mean()
    n_tr = len(y_tr_c)

    nulls = []
    for _ in range(n_perms):
        idx = rng.permutation(n_tr)
        y_shuf = y_tr_c[idx]
        w = Zt_pinv @ y_shuf
        norm = float(np.linalg.norm(w))
        if norm < 1e-12 or yn_te < 1e-12:
            nulls.append(float("nan"))
            continue
        a = w / norm
        z_proj = Ze_c @ a
        zn = float(np.linalg.norm(z_proj))
        if zn < 1e-12:
            nulls.append(float("nan"))
            continue
        nulls.append(float(z_proj @ y_te_c) / (zn * yn_te))
    return nulls


def within_cohort_null_distribution(
    Q_z: np.ndarray,
    y: np.ndarray,
    n_perms: int,
    rng: np.random.Generator,
) -> list[float]:
    """null distribution for the within-cohort CCA, cached QR.

    for univariate Y, top_canonical_correlation(Z, y) = ||Q.T @ yc|| / ||yc||,
    where Q = QR(Z - Z.mean())[0]. Q depends only on Z, so for a loop where
    only y is permuted, precomputing Q once and doing a matmul per permutation
    avoids refactoring the same n x d matrix per draw.

    call `prepare_within_cohort_cache(Z)` once per view to get Q_z.
    """
    n = y.shape[0]
    nulls = []
    for _ in range(n_perms):
        idx = rng.permutation(n)
        y_s = np.asarray(y[idx], dtype=np.float64).ravel()
        yc = y_s - y_s.mean()
        yn = float(np.linalg.norm(yc))
        if yn < 1e-12:
            nulls.append(float("nan"))
            continue
        s = float(np.linalg.norm(Q_z.T @ yc))
        nulls.append(np.clip(s / yn, -1.0, 1.0))
    return nulls


def prepare_within_cohort_cache(Z: np.ndarray) -> np.ndarray:
    """QR factor of centered Z, reusable across signals and permutations."""
    A = np.asarray(Z, dtype=np.float64)
    Ac = A - A.mean(axis=0, keepdims=True)
    Q, _ = np.linalg.qr(Ac, mode="reduced")
    return Q


# ======================================================== multivariate CCA ====

def multivariate_top_canonical_correlation(
    Z: np.ndarray, Y: np.ndarray, *, with_weights: bool = False,
) -> float | tuple[float, np.ndarray, np.ndarray]:
    """top canonical correlation between two multivariate blocks via QR+SVD.

    unlike `top_canonical_correlation`, Y is matrix-valued (N x p). the CCA
    finds the directions a in R^d and b in R^p that maximise corr(Za, Yb).
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
    top = float(np.clip(S[0], -1.0, 1.0))
    if not with_weights:
        return top

    def _solve(R, v):
        try:
            return np.linalg.solve(R, v)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(R, v, rcond=None)[0]

    return top, _solve(Ra, U[:, 0]), _solve(Rb, Vt[0])


def multivariate_transfer_correlation(
    Z_train: np.ndarray, Y_train: np.ndarray,
    Z_test: np.ndarray, Y_test: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """fit multivariate canonical directions on train, score on held-out.

    returns (r_test, r_train, a, b). r_test is pearsonr(Z_te_c @ a, Y_te_c @ b)
    with Z_te_c and Y_te_c centred on their own means.
    """
    result = multivariate_top_canonical_correlation(
        Z_train, Y_train, with_weights=True)
    _, a, b = result
    r_test = _mv_projected_corr(Z_test, Y_test, a, b)
    r_train = _mv_projected_corr(Z_train, Y_train, a, b)
    return r_test, r_train, a, b


def _mv_projected_corr(
    Z: np.ndarray, Y: np.ndarray, a: np.ndarray, b: np.ndarray,
) -> float:
    """pearsonr of the projections onto given directions, centred on own data."""
    Zc = np.asarray(Z, dtype=np.float64)
    Yc = np.asarray(Y, dtype=np.float64)
    Zc = Zc - Zc.mean(axis=0, keepdims=True)
    Yc = Yc - Yc.mean(axis=0, keepdims=True)
    u = Zc @ np.asarray(a, dtype=np.float64)
    v = Yc @ np.asarray(b, dtype=np.float64)
    u, v = np.ravel(u), np.ravel(v)
    if u.std() == 0 or v.std() == 0:
        return float("nan")
    from scipy.stats import pearsonr as _pearsonr
    return float(_pearsonr(u, v)[0])


def multivariate_transfer_null(
    Z_train: np.ndarray, Y_train: np.ndarray,
    Z_test: np.ndarray, Y_test: np.ndarray,
    n_perms: int, rng: np.random.Generator,
) -> list[float]:
    """permutation null for multivariate transfer CCA.

    permutes Y_train rows, refits directions via QR+SVD, re-scores on held-out.
    Q_Z and R_Z are cached (Z_train is invariant across permutations).
    """
    from scipy.stats import pearsonr as _pearsonr

    Zt = np.asarray(Z_train, dtype=np.float64)
    Zt_c = Zt - Zt.mean(axis=0, keepdims=True)
    Qz, Rz = np.linalg.qr(Zt_c, mode="reduced")

    Ze = np.asarray(Z_test, dtype=np.float64)
    Ze_c = Ze - Ze.mean(axis=0, keepdims=True)

    Yt = np.asarray(Y_train, dtype=np.float64)
    Yt_c = Yt - Yt.mean(axis=0, keepdims=True)
    n_tr = Yt_c.shape[0]

    Ye = np.asarray(Y_test, dtype=np.float64)
    Ye_c = Ye - Ye.mean(axis=0, keepdims=True)

    nulls = []
    for _ in range(n_perms):
        idx = rng.permutation(n_tr)
        Yp = Yt_c[idx]
        Qy, Ry = np.linalg.qr(Yp, mode="reduced")
        U, _, Vt = np.linalg.svd(Qz.T @ Qy, full_matrices=False)
        try:
            a = np.linalg.solve(Rz, U[:, 0])
        except np.linalg.LinAlgError:
            a = np.linalg.lstsq(Rz, U[:, 0], rcond=None)[0]
        try:
            b = np.linalg.solve(Ry, Vt[0])
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(Ry, Vt[0], rcond=None)[0]
        u = Ze_c @ a
        v = Ye_c @ b
        u, v = np.ravel(u), np.ravel(v)
        if u.std() == 0 or v.std() == 0:
            nulls.append(float("nan"))
            continue
        nulls.append(float(_pearsonr(u, v)[0]))
    return nulls


def multivariate_full_null(
    Z: np.ndarray, Y: np.ndarray,
    n_perms: int, rng: np.random.Generator,
) -> list[float]:
    """permutation null for multivariate option-B CCA (fit on all units).

    permutes Y rows, recomputes top canonical correlation. Q_Z cached.
    """
    A = np.asarray(Z, dtype=np.float64)
    B = np.asarray(Y, dtype=np.float64)
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    Qz, _ = np.linalg.qr(Ac, mode="reduced")
    n = Bc.shape[0]
    nulls = []
    for _ in range(n_perms):
        idx = rng.permutation(n)
        Qy, _ = np.linalg.qr(Bc[idx], mode="reduced")
        s = float(np.linalg.svd(Qz.T @ Qy, full_matrices=False, compute_uv=False)[0])
        nulls.append(float(np.clip(s, -1.0, 1.0)))
    return nulls


def finalize_pathway_cca_fdr(
    rows: list[dict], *, alpha: float = 0.05,
) -> list[dict]:
    """apply cross-run BH-FDR to collected pathway_transfer_cca rows.

    the step runs per-run and produces per-run p-values. this function
    recomputes fdr_A over the TESTABLE z_he cells across all runs (the
    headline family) and fdr_B over all cells, matching the script's
    cross-run family.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    head = df[(df["testable"]) & (df["view"] == "z_he")].copy()
    df["fdr_A"] = np.nan
    if len(head):
        head_p = head["p_two_A"].values
        fdr_a = bh_fdr(head_p, alpha=alpha,
                        family=[f"{r['run_id']}/{r['pathway_name']}"
                                for _, r in head.iterrows()])
        df.loc[head.index, "fdr_A"] = fdr_a["q"]
    all_p_B = df["p_one_B"].values
    fdr_b = bh_fdr(all_p_B, alpha=alpha,
                    family=[f"{r['run_id']}/{r['view']}/{r['pathway_name']}"
                            for _, r in df.iterrows()])
    df["fdr_B"] = fdr_b["q"]
    df["sig_A_05"] = df["fdr_A"] < alpha
    df["sig_B_05"] = df["fdr_B"] < alpha
    return df.to_dict("records")


__all__ = ["NotComparable", "bh_fdr", "bootstrap_ci", "cluster_agreement",
           "finalize_pathway_cca_fdr",
           "knn_purity", "linear_cka", "linear_probe",
           "multivariate_full_null", "multivariate_top_canonical_correlation",
           "multivariate_transfer_correlation", "multivariate_transfer_null",
           "permutation_test",
           "prepare_within_cohort_cache", "projected_correlation", "retrieval",
           "top_canonical_correlation", "transfer_correlation",
           "transfer_null_distribution", "two_sample_z",
           "within_cohort_null_distribution"]
