"""EVALUATE_STEPS on the real cohort, at cohort scale.

tier 2 for the CHAIN, which is a smaller claim than tier 2 for the measures
was, and `script_map/evaluation_maths.md` section 8 records why. of the numbers
the chain produces:

    the ARI and silhouette      HAVE an oracle - `mc_coherence.parquet` - and
                                are diffed exactly here, THROUGH the chain
    the AUC, CKA, gap           have an oracle - `metrics_h1_raw.json` - and are
                                diffed within the script's own float32 band
    the within-subject AUC      NO oracle. it is the input G2 was recorded as
                                `not_run` for wanting
    the ARI permutation null    NO oracle, for the same reason
    the axis specificity        NO oracle. `pathway_axes.parquet` carries no
                                vectors, so the cosine has never been computed
                                for this cohort
    the transfer r              not diffable without re-deriving the script's
                                data prep - a 457 MB embedding pkl and the
                                Reactome DAG, which is the CALLER's assembly and
                                not the chain's statistic

so the three with no oracle are asserted to be well-formed and are reported as
new. a test that cannot diff a number says so rather than inventing a
tolerance.

the null here is 20 permutations, the guard's own floor, against the published
500. a reduced null is NOT a candidate for a published number - the same rule
`CLAUDE.md` states for a reduced grid - and nothing below diffs it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omicstra.protocols import build_protocol
from omicstra.protocols.evaluate import EVALUATE_STEPS

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/tnbc-92_v3"
NICHES = ROOT / "data/embeddings/niches_v3"
RUN = "R1_v3"

# read out of the artifacts, as the measures slice-diff established
PUBLISHED = {"k": 14, "n_init": 10, "seed": 42, "silhouette_sample": 5000,
             "negative_ratio": 10, "negative_seed": 0}
F32 = 1e-6


AUCELL = ROOT / "data/embeddings/biological_signals/niche_aucell_5targets.parquet"


def _present() -> bool:
    return ((RUNS / RUN / "embeddings_test.parquet").is_file()
            and (RUNS / "eval/H2/mc_coherence.parquet").is_file()
            and (RUNS / RUN / "metrics_h1_raw.json").is_file()
            and NICHES.is_dir())


pytestmark = pytest.mark.skipif(
    not _present(), reason="the run artifacts are git-ignored; tier 2 runs "
                           "where they exist")

_CACHE: dict = {}


def _ctx_and_params():
    """the cohort's arrays in the chain's shape. a cohort adapter, not package code."""
    if "ctx" in _CACHE:
        return _CACHE["ctx"]

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
        frames.append(nf[["subarray", "spot_id", "mc_megacluster", "tls",
                          "mc_weights_niche"]])
    m = df.merge(pd.concat(frames, ignore_index=True),
                 on=["subarray", "spot_id"], how="left")
    m = m[m["mc_megacluster"].notna()]

    # float32, the dtype the published ARI was computed at
    he = np.stack([np.asarray(v, np.float32) for v in m["z_he"]])
    st = np.stack([np.asarray(v, np.float32) for v in m["z_st"]])
    mcw = np.stack([np.asarray(v, np.float32) for v in m["mc_weights_niche"]])

    ctx = {
        "spaces": {"z_he": he},
        "paired": {"z_he": st},
        "labels": {"mc_megacluster": m["mc_megacluster"].astype(int).to_numpy()},
        "confounder": m["patient_id"].to_numpy(),
        # continuous signals, handed over AS THEY ARE. these are SUPERVISION
        # columns used as TARGETS and never as features - the distinction G6
        # enforces, and what the published H2-C does with its signature scores.
        #
        # two of the three do not cover every unit: the supervision table is
        # absent for 4,353 of them. the adapter does NOT fill that in. imputing
        # a target invents the signal being tested for every unit it was absent
        # from, so the step is what decides, and it refuses those signals by
        # name while evaluating the one that is complete.
        "scores": {"tls": m["tls"].to_numpy(float),
                   "mc1": mcw[:, 0].astype(float),
                   "mc12": mcw[:, 11].astype(float)},
    }
    params = {
        "label_granularity": {"labels": ("mc_megacluster",),
                              "tau_label_nmi": 0.5},
        "paired_identifiability": {
            "ks": (1, 5, 10), "negative_ratio": PUBLISHED["negative_ratio"],
            "negative_seed": PUBLISHED["negative_seed"]},
        "structure_and_its_null": {
            "k": PUBLISHED["k"], "n_init": PUBLISHED["n_init"],
            "seed": PUBLISHED["seed"],
            "silhouette_sample": PUBLISHED["silhouette_sample"],
            "n_permutations": 20, "probe_max_iter": 200, "held_out_groups": 2},
        "held_out_comparison": {
            "ks": (1, 5, 10), "negative_ratio": PUBLISHED["negative_ratio"],
            "negative_seed": PUBLISHED["negative_seed"],
            "min_units_per_subject": 50},
        "association_and_specificity": {
            "n_permutations": 20, "alpha": 0.05, "tau_specificity": 0.5,
            "tail": "two_sided", "held_out_groups": 3,
            "transfer_seed": PUBLISHED["seed"],
            "within_cohort_seed": PUBLISHED["seed"],
            "min_units_per_side": 100},
        "hypothesis_verdict": {"hypotheses": {
            "H1": ("paired_identifiability", "held_out_comparison"),
            "H2": ("structure_and_its_null",),
            "H3": ("association_and_specificity",)}},
    }
    _CACHE["ctx"] = (ctx, params)
    return _CACHE["ctx"]


def _records():
    if "recs" not in _CACHE:
        ctx, params = _ctx_and_params()
        _CACHE["recs"] = build_protocol(EVALUATE_STEPS, "evaluate").invoke(
            {**ctx, "params": params})
    return _CACHE["recs"]


def _obs(step: str) -> dict:
    return _records()[step]["observed"]


def _raw() -> dict:
    return json.loads((RUNS / RUN / "metrics_h1_raw.json").read_text())


# ------------------------------------------------------- the chain at scale ---
def test_every_step_either_passes_or_states_why_it_could_not_run():
    """H3 is `not_run` on this cohort and the test below says why.
    confounder_suppression is `not_run` because the synthetic ctx carries
    no biology_labels or confounder_suppression params (it needs the
    adapter and project.json declaration). every other step passes."""
    recs = _records()
    expected_not_run = {"association_and_specificity", "confounder_suppression"}
    for step in expected_not_run:
        assert recs[step]["status"] == "not_run", (step, recs[step])
    others = {k: r["result"] for k, r in recs.items()
              if k not in expected_not_run and r["status"] != "pass"}
    assert not others, others
    assert not [k for k, r in recs.items() if r["status"] == "error"]


def test_the_chain_ran_on_every_labelled_niche():
    ctx, _ = _ctx_and_params()
    want = int(pd.read_parquet(RUNS / "eval/H2/mc_coherence.parquet")
               .query("run_id == @RUN and view == 'z_he'").iloc[0]["n"])
    assert ctx["spaces"]["z_he"].shape[0] == want


# ----------------------------------------------------- what HAS an oracle -----
def test_the_chains_ari_and_silhouette_match_the_published_table():
    """through the CHAIN, not through the measure directly - so this also pins
    the chain's assembly of the view and the label."""
    block = next(iter(_obs("structure_and_its_null")["per_pair"].values()))
    want = (pd.read_parquet(RUNS / "eval/H2/mc_coherence.parquet")
            .query("run_id == @RUN and view == 'z_he'").iloc[0])
    assert block["dtype"] == "float32"
    assert block["ari"] == pytest.approx(want["ari"], rel=1e-9)
    assert block["silhouette"] == pytest.approx(want["silhouette"], rel=1e-6)


def test_the_chains_retrieval_and_cka_match_the_published_metrics():
    v = _obs("paired_identifiability")["per_view"]["z_he"]
    raw = _raw()
    assert v["auc"] == pytest.approx(raw["auc"], rel=1e-3)
    assert v["cka"] == pytest.approx(raw["cka_after"], rel=F32)
    assert v["alignment_gap"] == pytest.approx(raw["alignment_gap"], rel=F32)
    assert v["n_queries"] == raw["n_queries"]


def test_the_recall_depths_match_the_published_rollup():
    v = _obs("paired_identifiability")["per_view"]["z_he"]
    raw = _raw()
    for k in (1, 5, 10):
        assert v[f"r{k}"] == pytest.approx(raw[f"r{k}"], rel=F32), k
    assert v["mrr"] == pytest.approx(raw["mrr"], rel=F32)
    assert v["median_rank"] == pytest.approx(raw["median_rank"], rel=F32)


# ------------------------------------------------- what has NO oracle ---------
def test_the_within_subject_score_is_new_and_is_what_g2_was_waiting_for():
    """no published counterpart: this is the input the pack-reading gate names
    in `_not_run("G2_held_out_honesty", "a within-subject score to compare")`.
    asserted to be well-formed and reported, not diffed."""
    v = _obs("held_out_comparison")["per_view"]["z_he"]
    assert v["guard_g2"] in ("pass", "fail")
    assert 0.0 < v["within_subject_auc"] <= 1.0
    assert v["cross_subject_auc"] == pytest.approx(_raw()["auc"], rel=1e-3)
    assert v["n_subjects_scored"] >= 2


def test_the_ari_permutation_null_is_new_and_well_formed():
    """the other guard the gate records as not_run. 20 permutations is the
    guard's floor and is NOT the published 500 - nothing here diffs it."""
    block = next(iter(_obs("structure_and_its_null")["per_pair"].values()))
    null = block["null"]
    assert null["n_null"] == 20
    assert null["p"] >= null["p_floor"] == pytest.approx(1 / 21)
    assert not np.isnan(null["z"])
    assert block["guard_g5"] in ("pass", "fail")


def test_no_continuous_signal_in_the_niche_table_covers_every_unit():
    """a finding about the cohort, not a defect in the chain.

    the three continuous columns the niche table carries all have gaps - the
    scalar signal for 22,391 of 35,594 units and the supervision vector for
    4,353 - so none of them is evaluable at a split shared with the others.
    the published H3 used pathway cosines built from a 457 MB embedding pkl and
    the Reactome DAG, which is the CALLER's assembly and not this chain's
    statistic.
    """
    o = _obs("association_and_specificity")
    assert o["unevaluable_signals"] == ["mc1", "mc12", "tls"]
    rows = {r["signal"]: r for r in o["per_view"]["z_he"]["per_signal"]}
    assert "22391 of 35594" in rows["tls"]["g5_not_run_because"].replace(",", "")
    assert "4353 of 35594" in rows["mc1"]["g5_not_run_because"].replace(",", "")
    for r in rows.values():
        assert r["unevaluable"] is True and r["guard_g5"] == "not_run"


def test_a_step_where_every_signal_was_unevaluable_is_not_a_pass():
    """0 of 0 significant is not a clean sweep. the repo's own rule - a check
    that measures nothing is not a pass - applied to the step that would
    otherwise report one."""
    rec = _records()["association_and_specificity"]
    assert rec["status"] == "not_run"
    assert "every declared signal is unevaluable" in rec["result"]
    assert _obs("association_and_specificity")["n_out"] == 0


def test_the_cca_names_a_non_finite_input_as_data_not_as_numerics():
    """handed the gaps directly, the SVD says 'did not converge', which reads
    as a numerical problem and is a coverage problem."""
    from omicstra.measures.evaluation import NotComparable, top_canonical_correlation

    ctx, _ = _ctx_and_params()
    with pytest.raises(NotComparable, match="non-finite"):
        top_canonical_correlation(ctx["spaces"]["z_he"], ctx["scores"]["mc1"])


def test_the_axis_specificity_is_new_because_the_artifact_holds_no_vectors():
    """`pathway_axes.parquet` carries pathway_name, pathway_id, coverage_frac,
    set_size and testable. the cosine between axes has never been computed for
    this cohort, so there is nothing to diff against."""
    axes = pd.read_parquet(
        RUNS / "eval/H3/pathway_cca_gpath2vec_v3/pathway_axes.parquet")
    assert not any(c.startswith(("axis", "weight", "a_", "vector"))
                   for c in axes.columns), "the artifact gained vectors; diff them"

    # and on THIS cohort no declared signal is evaluable, so there are no axes
    # to compare. a coverage fact, reported as `not_run` with the reason - not
    # a specificity claim either way.
    blk = _obs("association_and_specificity")["per_view"]["z_he"]
    assert blk["specificity_status"] == "not_run"
    assert "fewer than two axes" in blk["specificity_missing"]


def test_the_split_is_still_taken_by_whole_subjects_before_anything_refuses():
    """the refusal is about the signals, not about the split - so the split is
    computed and reported either way, which is what makes the reason legible."""
    o = _obs("association_and_specificity")
    ctx, _ = _ctx_and_params()
    assert o["n_train"] + o["n_test"] == ctx["spaces"]["z_he"].shape[0]
    assert o["n_test"] > 0 and o["n_train"] > 0


# --------------------------------------------------------------- the verdict ---
def test_the_verdict_is_computed_for_every_hypothesis_on_the_real_cohort():
    v = _obs("hypothesis_verdict")["verdicts"]
    assert set(v) == {"H1", "H2", "H3"}
    for h in ("H1", "H2"):
        assert v[h]["verdict"] in ("biological", "diagnostic", "unguarded"), v[h]
        assert v[h]["guards"], f"{h} reached a verdict with no guard"
    # H3 has no statistic on this cohort, so it is `not_run` - which is a
    # different answer from "no guard failed" and is why the vocabulary has four
    # states rather than two
    assert v["H3"]["verdict"] == "not_run"
    assert "association_and_specificity" in v["H3"]["step_status"]


def test_the_three_closed_guards_all_ran_on_real_data():
    """the card's done-criterion, against the cohort rather than a fixture."""
    ran = {g["guard"] for step in (s.id for s in EVALUATE_STEPS)
           for g in _obs(step).get("guards", [])
           if g["status"] in ("pass", "fail")}
    # G4 is absent on this cohort and the test above says why: one evaluable
    # signal gives one axis, and a specificity guard needs two to compare.
    assert {"G1_label_granularity", "G2_held_out_honesty",
            "G5_noise_floor"} <= ran


# --------------------------------------------------- what the guards conclude --
def test_g1_reproduces_the_published_archetype_diagnostic():
    """the matrix states NMI(archetype, patient) ~ 0.89 as the reason that label
    may not carry a biology claim. the chain's G1 computes it."""
    from omicstra.protocols.evaluate import nmi_label_confounder

    rd = pd.read_parquet(RUNS / RUN / "embeddings_test.parquet")
    g = nmi_label_confounder(rd["archetype"].to_numpy(),
                             rd["patient_id"].to_numpy(), tau=0.5)
    assert g.value == pytest.approx(0.89, abs=0.005)
    assert not g.passed


def test_the_recommended_label_also_sits_above_the_declared_bar():
    """a new number, and a close one: `mc_megacluster` is the label the matrix
    routes to INSTEAD of archetype, and its NMI against the subject is 0.539
    against a declared tau of 0.5 - far below archetype's 0.893 and still over.

    so by this chain's own guard the clustering claim on this cohort is
    diagnostic rather than biological. the verdict turns on the bar, 0.539
    against 0.5, and the bar is a declaration - `EVALUATION.md` section 8's
    middle kind, to be re-derived per cohort rather than inherited. this test
    pins the measurement, not a conclusion about the biology.
    """
    o = _obs("label_granularity")
    assert o["nmi"]["mc_megacluster"] == pytest.approx(0.539, abs=0.005)
    assert o["tau"] == 0.5
    assert o["confounded"] == ["mc_megacluster"]


def test_h2_is_diagnostic_on_this_cohort_and_the_score_is_still_reported():
    v = _obs("hypothesis_verdict")["verdicts"]["H2"]
    assert v["verdict"] == "diagnostic"
    assert any("mc_megacluster" in f for f in v["guards_failed"])
    block = next(iter(_obs("structure_and_its_null")["per_pair"].values()))
    assert not np.isnan(block["ari"]) and block["claims_biology"] is False


def test_the_within_subject_auc_is_lower_than_the_cross_subject_one():
    """the direction a reader would assume, and it is the other way.

    a rank statistic is inflated within subject because the pool is smaller. a
    discrimination AUC can fall, because the mismatched pairs drawn within one
    subject are HARD negatives - same patient, adjacent tissue. the guard
    measures the drop and does not assume a sign, which is why it passes here.
    """
    v = _obs("held_out_comparison")["per_view"]["z_he"]
    assert v["within_subject_auc"] == pytest.approx(0.736, abs=0.005)
    assert v["cross_subject_auc"] > v["within_subject_auc"]
    assert v["guard_g2"] == "pass"
    assert v["g2"]["value"] < 0, "a negative drop is a rise"


# ------------------------------- the same verdict, reached through the graph ---
# the chain above is invoked directly. these reach it the way a CLIENT does -
# through the graph's compute arm, with the arrays assembled by the declared
# reader rather than by this file's own adapter. if the two disagree, the
# reachable path is measuring something else.
def _through_the_graph():
    if "graph" not in _CACHE:
        import json as _json

        from omicstra.graphs.evaluate import build_evaluate_graph

        cohort = _json.loads((ROOT / "projects/tnbc-92/cohort.json").read_text())
        if "evaluate_params" not in cohort:
            pytest.skip("this cohort declares no evaluate_params")
        _CACHE["graph"] = build_evaluate_graph().invoke(
            {"project_id": "tnbc-92", "arm": "evaluate", "compute": True,
             "run": RUN, "paired_view": "z_st", "signals": ["tls"],
             "cohort": cohort})
    return _CACHE["graph"]


def test_the_graphs_compute_arm_measures_the_same_units():
    r = _through_the_graph()["report"]
    assert r["status"] == "measured"
    assert r["run"] == RUN and r["view"] == "z_he"
    ctx, _ = _ctx_and_params()
    assert r["n_units"] == ctx["spaces"]["z_he"].shape[0]
    assert r["n_samples"] == 38


def test_the_graph_reaches_the_same_verdicts_as_the_chain_directly():
    """the reachability claim, as a test: a client's path and a notebook's path
    answer the same question with the same answer."""
    theirs = _through_the_graph()["report"]["verdicts"]
    mine = _obs("hypothesis_verdict")["verdicts"]
    assert {h: d["verdict"] for h, d in theirs.items()} == \
           {h: d["verdict"] for h, d in mine.items()}


def test_the_declared_reader_finds_the_sentinel_the_join_writes():
    """882 units carry -1 for the label, which the join writes for ABSENCE while
    its own comment says the eval code filters `< 0`. it does not - the
    published eval filters `notna()` only. the reader reports it rather than
    scoring it unknowingly."""
    r = _through_the_graph()["report"]
    assert r["label_sentinels"] == {"mc_megacluster": 882}
    assert "ABSENCE" in r["label_note"]


def test_the_cohorts_declared_params_are_what_the_arm_used():
    """not the package's. every threshold travelled from cohort.json."""
    import json as _json

    cohort = _json.loads((ROOT / "projects/tnbc-92/cohort.json").read_text())
    p = cohort["evaluate_params"]
    assert p["structure_and_its_null"]["k"] == PUBLISHED["k"]
    assert p["structure_and_its_null"]["n_init"] == PUBLISHED["n_init"]
    assert p["paired_identifiability"]["negative_ratio"] == PUBLISHED["negative_ratio"]
    assert p["paired_identifiability"]["negative_seed"] == PUBLISHED["negative_seed"]
    # the null is the GUARD's floor, not the published 500 - and says so
    assert p["structure_and_its_null"]["n_permutations"] == 20
    assert "reduced null is never a candidate" in p["note"]


def test_each_guards_note_is_its_own_finding():
    """G5's note read `why_diagnostic`, which is G1's finding - so the guard
    table said the permutation null passed BECAUSE the label encodes the
    confounder, reporting two independent guards as one reason."""
    rows = [g for step in (s.id for s in EVALUATE_STEPS)
            for g in _obs(step).get("guards", [])]
    g5 = [g for g in rows if g["guard"] == "G5_noise_floor"
          and g["status"] in ("pass", "fail")]
    assert g5, "G5 did not run, so there is no note to check"
    for g in g5:
        assert "encodes the confounder" not in (g.get("note") or ""), g
        assert "permutation" in (g.get("note") or ""), g


# ------------------------------- H3 through the graph with pathway scores ---
@pytest.mark.skipif(not AUCELL.is_file(),
                    reason="niche_aucell_5targets.parquet is git-ignored")
def test_h3_through_the_graph_with_pathway_scores():
    """the graph path with include_pathway_scores=True finds evaluable signals
    and produces an H3 verdict that is not not_run."""
    import json as _json

    from omicstra.graphs.evaluate import build_evaluate_graph

    cohort = _json.loads((ROOT / "projects/tnbc-92/cohort.json").read_text())
    if "evaluate_params" not in cohort:
        pytest.skip("this cohort declares no evaluate_params")
    result = build_evaluate_graph().invoke(
        {"project_id": "tnbc-92", "arm": "evaluate", "compute": True,
         "run": RUN, "paired_view": "z_st", "signals": ["tls"],
         "include_pathway_scores": True, "cohort": cohort})
    r = result["report"]
    assert r["status"] == "measured"
    v = r["verdicts"]
    assert v["H3"]["verdict"] in ("biological", "diagnostic")
    steps = r.get("steps", {})
    assert steps.get("association_and_specificity") == "pass"
