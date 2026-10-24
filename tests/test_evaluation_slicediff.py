"""measures/evaluation.py against the numbers the routing matrix cites.

tier 2. every statistic here is recomputed from the inputs the published run
stored and diffed against the published value. runs where the cohort exists and
skips elsewhere - the run artifacts are git-ignored.

three criteria, and which one applies is a property of the statistic:

    EXACT          the published value was computed from stored inputs by a
                   closed-form path. the AUC, the permutation z and p, and the
                   BH q are exact to the last bit
    float32-bound  the script accumulated a closed-form statistic in float32
                   and this module accumulates in float64. the difference is
                   the SCRIPT's rounding, so the tolerance is 1e-6 relative and
                   is declared here rather than hidden in the function
    dtype-bound    an iterative estimator is deterministic at a given precision
                   and not across precisions. reproduced by passing the dtype
                   the artifact was computed at, which is float32

the cohort's parameters live in THIS FILE, read out of the artifacts' own
provenance. the module holds no default for any of them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omicstra.measures.evaluation import (
    bh_fdr,
    cluster_agreement,
    linear_cka,
    permutation_test,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/tnbc-92_v3"
NICHES = ROOT / "data/embeddings/niches_v3"
H3 = RUNS / "eval/H3/pathway_cca_gpath2vec_v3"
RUN = "R1_v3"

# read out of the artifacts, not chosen here:
#   negative_ratio   neg_sample_cosines.size / matched_cosines.size  = 10
#   seed, n_init     eval_h2_mc_coherence.py, and the artifact reproduces at 42/10
#   k                mc_coherence.parquet, KMeans(k=14) against mc_megacluster
#   silhouette       5000, the caller-side subsample
#   n_perms          provenance.json#n_perms = 500
#   alpha            0.05, the matrix's own cells
PUBLISHED = {"negative_ratio": 10, "seed": 42, "n_init": 10, "k": 14,
             "silhouette_sample": 5000, "n_perms": 500, "alpha": 0.05}

# the float32 accumulation band. the script computed CKA and the alignment gap
# over a materialised float32 similarity matrix; this module sums in float64.
F32 = 1e-6


def _present() -> bool:
    return ((RUNS / RUN / "h1_eval_data.npz").is_file()
            and (RUNS / "eval/H2/mc_coherence.parquet").is_file()
            and (H3 / "per_pathway_cca.parquet").is_file()
            and NICHES.is_dir())


pytestmark = pytest.mark.skipif(
    not _present(), reason="the run artifacts are git-ignored; tier 2 runs "
                           "where they exist")

_CACHE: dict[str, object] = {}


def _npz():
    if "npz" not in _CACHE:
        _CACHE["npz"] = np.load(RUNS / RUN / "h1_eval_data.npz")
    return _CACHE["npz"]


def _raw() -> dict:
    return json.loads((RUNS / RUN / "metrics_h1_raw.json").read_text())


def _embeddings() -> tuple[np.ndarray, np.ndarray]:
    if "emb" not in _CACHE:
        d = pd.read_parquet(RUNS / RUN / "embeddings_test.parquet")
        _CACHE["emb"] = (np.stack(d["z_he"].to_numpy()),
                         np.stack(d["z_st"].to_numpy()))
    return _CACHE["emb"]              # type: ignore[return-value]


def _labelled() -> tuple[np.ndarray, np.ndarray]:
    """z_he and mc_megacluster for the test niches, joined as the script joined.

    float32, which is the dtype the published ARI was computed at - see the
    dtype test below for why that is load-bearing rather than incidental.
    """
    if "lab" not in _CACHE:
        rd = pd.read_parquet(RUNS / RUN / "embeddings_test.parquet")
        df = rd.reset_index().copy()
        if "spot_id" not in df.columns:
            df["spot_id"] = rd.index.values
        frames = []
        for sub in df["subarray"].unique():
            p = NICHES / f"{sub}.parquet"
            if not p.is_file():
                continue
            nf = pd.read_parquet(p).reset_index()
            nf["subarray"] = sub
            frames.append(nf[["subarray", "spot_id", "mc_megacluster"]])
        m = df.merge(pd.concat(frames, ignore_index=True),
                     on=["subarray", "spot_id"], how="left")
        m = m[m["mc_megacluster"].notna()]
        _CACHE["lab"] = (np.stack([np.asarray(v, np.float32) for v in m["z_he"]]),
                         m["mc_megacluster"].astype(int).to_numpy())
    return _CACHE["lab"]              # type: ignore[return-value]


def _h3() -> pd.DataFrame:
    if "h3" not in _CACHE:
        _CACHE["h3"] = (pd.read_parquet(H3 / "per_pathway_cca.parquet")
                        .merge(pd.read_parquet(H3 / "perm_nulls.parquet"),
                               on=["run_id", "view", "pathway_name"]))
    return _CACHE["h3"]               # type: ignore[return-value]


# ------------------------------------------------------------------ EXACT ----
def test_the_auc_is_exact_from_the_stored_labels_and_scores():
    """the run stored the exact positives and the exact negative sample, so
    there is nothing to re-sample and nothing to round."""
    from sklearn.metrics import roc_auc_score

    z = _npz()
    assert roc_auc_score(z["labels"], z["scores"]) == _raw()["auc"]


def test_the_stored_negative_sample_is_the_declared_ratio():
    """the ratio was implicit in the script and is a required argument here."""
    z = _npz()
    ratio = z["neg_sample_cosines"].size // z["matched_cosines"].size
    assert ratio == PUBLISHED["negative_ratio"]


def test_the_negative_sample_seed_and_the_bootstrap_seed_are_different_numbers():
    """two seeds live in this run and they are not the same: the negative sample
    was drawn under 0 inside the training script, and the bootstrap under 42.
    the artifact records only the second, which is why the first is a required
    argument rather than a default."""
    assert int(_npz()["resample_seed"]) == PUBLISHED["seed"]
    ci = json.loads((RUNS / RUN / "metrics_h1_ci.json").read_text())
    assert "seed=42" in ci["cka_ci_method"]


@pytest.mark.parametrize("run_id", ["R1_v3", "R4_v3", "R6_v3"])
def test_the_permutation_moments_and_p_are_bit_exact(run_id):
    """the run stored its 500 nulls, their mean and their sd, so these are
    arithmetic over a fixed float64 array and no tolerance applies."""
    rows = _h3().query("run_id == @run_id and view == 'z_he'")
    assert len(rows) == 5
    for _, r in rows.iterrows():
        nulls = np.asarray(r["null_A"], dtype=np.float64)
        assert nulls.size == PUBLISHED["n_perms"]
        got = permutation_test(r["top_corr_test"], nulls, tail="two_sided")
        # the moments are bit-identical to the ones the artifact stored, which
        # also pins the convention: the published sd is the POPULATION sd
        assert got["null_mean"] == r["null_A_mean"], r["pathway_name"]
        assert got["null_sd"] == r["null_A_std"], r["pathway_name"]
        assert got["p"] == pytest.approx(r["p_two_A"], rel=1e-12), r["pathway_name"]
        # and the z is exactly the quotient of those two stored moments
        assert got["z"] == pytest.approx(
            (r["top_corr_test"] - r["null_A_mean"]) / r["null_A_std"], rel=1e-15)


def test_the_published_z_column_disagrees_with_its_own_stored_moments():
    """at 1e-10, and found by asking for bit-equality rather than assuming it.

    the artifact stores `null_A_mean`, `null_A_std` and `z_A`. the first two are
    reproduced exactly from the stored nulls; `z_A` is not exactly their
    quotient. the scale is far below anything that moves a verdict - every q in
    the family is unchanged - but it is the kind of drift that a single
    implementation removes and three implementations accumulate.
    """
    r = _h3().query("run_id == 'R1_v3' and view == 'z_he'").iloc[0]
    quotient = (r["top_corr_test"] - r["null_A_mean"]) / r["null_A_std"]
    assert r["z_A"] != quotient
    assert r["z_A"] == pytest.approx(quotient, rel=1e-9)


def test_the_null_sd_is_the_population_form_not_the_sample_form():
    """ddof is a convention and the two differ by a tenth of a percent at
    n=500, which is enough to move a borderline q."""
    r = _h3().query("run_id == 'R1_v3' and view == 'z_he'").iloc[0]
    nulls = np.asarray(r["null_A"], dtype=np.float64)
    assert nulls.std(ddof=0) == r["null_A_std"]
    assert nulls.std(ddof=1) != pytest.approx(r["null_A_std"], rel=1e-9)


def test_the_published_fdr_family_is_the_whole_grid_not_one_run():
    """this is the reason the family is an argument.

    the matrix's cell reads "BH-FDR over 4 testable", and over 4 the q for the
    one significant pathway is 0.0479. the ARTIFACT's fdr_A is 0.0227, which is
    BH over 36 - nine runs times four testable pathways, one view. both are
    valid corrections of different claims, and only one of them is what the
    stored number means.
    """
    h3 = _h3()
    family = h3.query("view == 'z_he' and testable")
    q = bh_fdr(family["p_two_A"].to_numpy(), alpha=PUBLISHED["alpha"],
               family=[f"{r}:{p}" for r, p in zip(family["run_id"],
                                                  family["pathway_name"])])
    assert q["n_family"] == 36
    assert np.allclose(q["q"], family["fdr_A"].to_numpy(), rtol=1e-12)
    assert q["rejected"] == family["sig_A_05"].tolist()


def test_correcting_over_one_run_gives_a_different_q_for_the_same_p():
    one = _h3().query("run_id == 'R1_v3' and view == 'z_he' and testable")
    q = bh_fdr(one["p_two_A"].to_numpy(), alpha=PUBLISHED["alpha"])
    immune = list(one["pathway_name"]).index("Immune_System")
    assert q["n_family"] == 4
    assert q["q"][immune] == pytest.approx(0.047904, abs=1e-6)
    assert q["q"][immune] != pytest.approx(one.iloc[immune]["fdr_A"], abs=1e-6)


# ---------------------------------------------------------- float32-bound ----
def test_the_cka_matches_the_published_value_within_the_scripts_rounding():
    he, st = _embeddings()
    assert linear_cka(he, st) == pytest.approx(_raw()["cka_after"], rel=F32)


def test_the_alignment_gap_matches_without_materialising_the_matrix():
    """mean(off-diagonal) is computed from the column sums rather than from a
    35,594 x 35,594 matrix - the same number, and five orders of magnitude less
    memory. the published value came from the materialised float32 version."""
    he, st = _embeddings()
    n = he.shape[0]
    total = float(np.sum(he.sum(0) * st.sum(0)))
    diag = float(np.sum(he * st, axis=1).sum())
    gap = diag / n - (total - diag) / (n * n - n)
    assert gap == pytest.approx(_raw()["alignment_gap"], rel=F32)


def test_two_scripts_recorded_the_same_cka_to_different_precision():
    """an argument for one implementation: the same statistic on the same run
    sits in two artifacts with different last digits, because two scripts
    accumulated it separately."""
    ci = json.loads((RUNS / RUN / "metrics_h1_ci.json").read_text())
    assert ci["cka"] != _raw()["cka_after"]
    assert ci["cka"] == pytest.approx(_raw()["cka_after"], rel=F32)


# ------------------------------------------------------------ dtype-bound ----
def test_the_ari_and_silhouette_match_at_the_dtype_the_artifact_used():
    he32, y = _labelled()
    got = cluster_agreement(he32, y, k=PUBLISHED["k"], n_init=PUBLISHED["n_init"],
                            seed=PUBLISHED["seed"],
                            silhouette_sample=PUBLISHED["silhouette_sample"])
    want = (pd.read_parquet(RUNS / "eval/H2/mc_coherence.parquet")
            .query("run_id == @RUN and view == 'z_he'").iloc[0])
    assert got["dtype"] == "float32"
    assert got["n"] == int(want["n"])
    assert got["ari"] == pytest.approx(want["ari"], rel=1e-9)
    assert got["silhouette"] == pytest.approx(want["silhouette"], rel=1e-6)


def test_upcasting_the_same_data_changes_the_published_ari():
    """why `cluster_agreement` does not upcast. a seed makes KMeans
    deterministic at a given precision, not across precisions - so a port that
    tidied the dtype would have moved a published number with nothing to show."""
    he32, y = _labelled()
    kw = {"k": PUBLISHED["k"], "n_init": PUBLISHED["n_init"],
          "seed": PUBLISHED["seed"],
          "silhouette_sample": PUBLISHED["silhouette_sample"]}
    at32 = cluster_agreement(he32, y, **kw)
    at64 = cluster_agreement(he32.astype(np.float64), y, **kw)
    assert at64["dtype"] == "float64"
    assert at32["ari"] != at64["ari"]
    assert at32["ari"] == pytest.approx(at64["ari"], abs=1e-3)   # same answer, other optimum


# ------------------------------------------------------ the CI convention ----
def test_the_bootstrap_quantile_convention_reproduces_the_published_interval():
    """`bootstrap_ci` resamples internally, so what is checked against the
    artifact is the convention it uses to turn draws into an interval: the
    central ci, two-tailed. the run stored its 200 draws."""
    z = _npz()
    ci = json.loads((RUNS / RUN / "metrics_h1_ci.json").read_text())
    for draws, lo_key, hi_key in (("auc_bootstrap", "auc_bootstrap_ci_low",
                                   "auc_bootstrap_ci_high"),
                                  ("cka_bootstrap", "cka_ci_low", "cka_ci_high")):
        want_ci = 0.95
        lo_q, hi_q = (1 - want_ci) / 2, 1 - (1 - want_ci) / 2
        # the draws are stored as float32 while the script took its quantiles
        # over the float64 values, so this is the same float32 band as the CKA
        lo, hi = np.quantile(z[draws], [lo_q, hi_q])
        assert lo == pytest.approx(ci[lo_key], rel=F32), draws
        assert hi == pytest.approx(ci[hi_key], rel=F32), draws


def test_the_pack_cites_the_bootstrap_interval_and_not_the_parametric_one():
    """the run computes two AUC intervals and the evidence pack uses one. the
    other is a Hanley-McNeil closed form that this module does not implement,
    because nothing routes on it."""
    ci = json.loads((RUNS / RUN / "metrics_h1_ci.json").read_text())
    pack = json.loads((ROOT / "projects/tnbc-92/routing_evidence.json").read_text())
    cited = next(c for c in pack["tasks"]["cross_modal_retrieval"]["candidates"]
                 if c["id"] == RUN)
    assert cited["ci"][0] == pytest.approx(ci["auc_bootstrap_ci_low"], abs=5e-5)
    assert cited["ci"][0] != pytest.approx(ci["auc_ci_low"], abs=5e-5)
    assert ci["auc_ci_method"] == "hanley_mcneil_1982"
