"""the molecular assay's pathway chains, as chains.

`design/analysis_pipeline_steps.md` §5 documents ONE of them. `scripts/
niche_aucell.py` is a second, with a different normalisation, and §5 does not
mention it - so porting "§5" alone would have silently dropped a chain still in
use. the two share only the niche aggregate.
"""
from __future__ import annotations

import pytest

from omicstra.protocols import build_protocol
from omicstra.protocols.pathway import NICHE_MEMBERSHIP_STEPS

CTX = {"genes": ["MT-CO1", "MTRNR2L1", "ACTB", "EPCAM"],
       "niche_members": {"n1": list(range(7)), "n2": list(range(5))}}
DECLARED = {"mito_filter": {"pattern": "^(MT-|MTRNR)"},
            "niche_membership": {"k_neighbors": 6}}


def _run(params, **ctx):
    return build_protocol(NICHE_MEMBERSHIP_STEPS, "niche_membership").invoke(
        {**CTX, **ctx, "params": params})


# --- declared, not assumed --------------------------------------------------
def test_an_undeclared_threshold_does_not_run_and_does_not_guess():
    """a number chosen on one cohort looking at one tissue must not travel to
    the next silently. EVALUATION.md section 8 calls that the third kind.

    there are TWO ways to be undeclared and they have different causes, so they
    read differently: the cohort named the step and left the field out, or it
    never named the step at all. the framework owns the second.
    """
    # the step is named, the field is missing - the step says which field
    r = _run({"mito_filter": {}, "niche_membership": {"k_neighbors": 6}})
    assert r["mito_filter"]["status"] == "not_run"
    assert "pattern is not declared" in r["mito_filter"]["result"]
    assert "pattern" in r["mito_filter"]["decision"]

    # the step is not named at all - the framework says so, before the step runs
    r = _run({"niche_membership": {"k_neighbors": 6}})
    assert r["mito_filter"]["status"] == "not_run"
    assert "no params for this step" in r["mito_filter"]["result"]


def test_a_step_cannot_read_a_parameter_it_did_not_declare():
    """params_space is enforced by the framework, so whatever else a cohort's
    file carries never reaches the measure."""
    r = _run({**DECLARED, "niche_membership": {"k_neighbors": 6, "pattern": "junk"}})
    assert r["niche_membership"]["status"] == "pass"
    assert r["niche_membership"]["observed"]["k_neighbors"] == 6


def test_an_upstream_failure_makes_the_next_step_not_run_not_error():
    r = _run({})
    assert r["niche_membership"]["status"] == "not_run"
    assert "upstream" in r["niche_membership"]["result"]


# --- the chain's own rules --------------------------------------------------
def test_mitochondrial_genes_are_dropped_before_aggregation():
    """before, not after: a niche's library size is computed on what remains,
    and summing first would change what every later normalisation divides by.
    """
    r = _run(DECLARED)
    o = r["mito_filter"]["observed"]
    assert (o["n_in"], o["n_out"]) == (4, 2) and o["reason"] == "mitochondrial"


def test_the_niche_width_comes_from_the_declaration():
    """"7 spots" is one platform's number - k=6 plus a centre. MODALITY_TEMPLATE
    says a new cohort must not have to edit code to change it."""
    for k, width in ((6, 7), (4, 5), (8, 9)):
        r = _run({**DECLARED, "niche_membership": {"k_neighbors": k}})
        assert r["niche_membership"]["observed"]["spots_per_niche"] == width


def test_membership_does_not_aggregate():
    """the two chains fork AT aggregation: one sums the spots of a niche into a
    single profile, the other keeps them apart as several observations of one
    niche. collapsing here would hand both the same row and silently make the
    variable-gene chain impossible."""
    r = _run(DECLARED)
    o = r["niche_membership"]["observed"]
    assert o["n_in"] == o["n_out"], "membership collapses nothing"
    assert "spots_per_niche" in o, "it reports the width, it does not apply it"


def test_a_partial_niche_is_counted_here_not_dropped():
    """dropping is a qc decision on declared thresholds, and it belongs to the
    step that owns those thresholds."""
    r = _run(DECLARED)
    o = r["niche_membership"]["observed"]
    assert (o["n_full"], o["n_partial"]) == (1, 1)
    assert o["n_in"] == o["n_out"], "the aggregate drops nothing"


def test_every_step_reports_a_funnel():
    """an unexplained drop is an error - section 6's rule. a step that filters
    says how many went in, how many came out, and why."""
    r = _run(DECLARED)
    for sid, rec in r.items():
        o = rec["observed"]
        assert {"n_in", "n_out"} <= set(o), f"{sid} reports no funnel"
        if o["n_in"] != o["n_out"]:
            assert o.get("reason"), f"{sid} drops rows without a reason"


# --- it is a chain, not functions that happen to use the types -------------
def test_the_order_is_asserted_at_build_time():
    """a reordered DAG is a different and wrong method carrying a confident
    trace, so it must fail before any data is touched."""
    from omicstra.protocols import ProtocolOrderError

    with pytest.raises(ProtocolOrderError):
        build_protocol(list(reversed(NICHE_MEMBERSHIP_STEPS)), "backwards")


def test_each_step_is_its_own_traced_span():
    """one span per step, named by step id, is what makes a trace readable as
    the method rather than as one opaque call."""
    from langchain_core.callbacks.base import BaseCallbackHandler

    seen: list[str] = []

    class Spy(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, **kw):
            if kw.get("name"):
                seen.append(kw["name"])

    build_protocol(NICHE_MEMBERSHIP_STEPS, "niche_membership").invoke(
        {**CTX, "params": DECLARED}, config={"callbacks": [Spy()]})
    assert {"mito_filter", "niche_membership"} <= set(seen), seen


# --- the variable-gene chain ----------------------------------------------
# transcribed from the code that wrote `niche_madmean_genesets.json`, with the
# parameters read out of that notebook's constants and CHECKED against the
# artifact itself: 3,251 sampled niches, median 55 genes, minimum exactly 20.
import numpy as np

from omicstra.protocols.pathway import VARIABLE_GENE_STEPS

VG_PARAMS = {"niche_qc": {"min_niche_umi": 1000, "min_spot_umi": 200,
                          "min_detected_spots": 3},
             "normalise": {"target_sum": 1e6},
             "variability": {},
             "background": {"min_background": 100},
             "foreground": {"top_fraction": 0.25, "min_foreground": 20},
             "gene_sets": {}}


def _niches(n=20, g=400, shallow=0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        depth = 40 if i < shallow else 400
        c = rng.poisson(depth / g, size=(7, g)).astype("float32") * (depth / 100)
        out.append((f"n{i}", c))
    return out


def _vg(params=None, niches=None, g=400):
    return build_protocol(VARIABLE_GENE_STEPS, "variable_gene").invoke(
        {"params": params if params is not None else VG_PARAMS,
         "niches": niches if niches is not None else _niches(g=g),
         "gene_symbols": [f"G{j}" for j in range(g)]})


def test_the_chain_runs_end_to_end_and_every_step_reports_a_funnel():
    r = _vg(niches=_niches(n=40, shallow=5))
    assert all(v["status"] == "pass" for v in r.values()), \
        {k: v["status"] for k, v in r.items()}
    assert r["niche_qc"]["observed"]["n_in"] == 40
    assert r["niche_qc"]["observed"]["n_out"] == 35, "the shallow five are dropped"
    for sid, rec in r.items():
        o = rec["observed"]
        assert {"n_in", "n_out"} <= set(o), sid
        if o["n_in"] != o["n_out"]:
            assert o.get("reason"), f"{sid} drops without naming a cause"


def test_a_drop_is_attributed_to_its_cause():
    """two qc gates ask different things: total depth over the niche, and how
    many spots carry signal. a niche can clear the first on one bright spot."""
    r = _vg(niches=_niches(n=10, shallow=4))
    why = r["niche_qc"]["observed"]["reason"]
    assert why["total_umi"] == 4 and why["detected_spots"] == 0


# --- the three details prose would have got wrong -------------------------
def test_the_mad_is_about_the_median_and_divided_by_the_MEAN():
    """not MAD/median, and no 1.4826 consistency factor - nothing here assumes
    normality. a ratio computed the other way would rank different genes."""
    from omicstra.protocols.pathway import variability

    z = np.array([[1.0, 10.0], [2.0, 10.0], [9.0, 10.0]], dtype="float32")
    _, upd = variability({"params": {}, "normalised": [("n", z)]})
    _, mean, ratio = upd["variability"][0]

    med = np.median(z, axis=0)
    expect = np.median(np.abs(z - med), axis=0) / (z.mean(axis=0) + 1e-9)
    assert np.allclose(ratio, expect)
    # and it is NOT MAD/median, which would differ on the first gene
    assert not np.isclose(ratio[0], expect[0] * z.mean(axis=0)[0] / med[0])


def test_the_percentile_runs_over_the_non_zero_ratios_only():
    """a zero-variance gene is locally constant and uninformative, and ranking
    it adds arbitrary tie-break noise on a niche with few spots."""
    from omicstra.protocols.pathway import foreground

    n_genes = 200
    ratio = np.zeros(n_genes, dtype="float32")
    ratio[:40] = np.linspace(0.1, 1.0, 40)       # only 40 are non-zero
    bg = np.ones(n_genes, dtype=bool)
    _, upd = foreground({"params": {"top_fraction": 0.25, "min_foreground": 5},
                         "background": [("n", bg, ratio)]})
    picked = upd["foreground"][0][1]
    assert picked.size == 10, "25% of the 40 NON-ZERO, not of all 200"
    assert (ratio[picked] > 0).all(), "a zero-variance gene was selected"


def test_min_foreground_is_a_floor_on_the_count_not_only_a_skip():
    """measured against the published artifact: its smallest gene set is
    exactly min_foreground, and none is below it."""
    from omicstra.protocols.pathway import foreground

    ratio = np.zeros(200, dtype="float32")
    ratio[:24] = np.linspace(0.1, 1.0, 24)       # 25% of 24 is 6, below the floor
    _, upd = foreground({"params": {"top_fraction": 0.25, "min_foreground": 20},
                         "background": [("n", np.ones(200, dtype=bool), ratio)]})
    assert upd["foreground"][0][1].size == 20, "the floor raises the count"


def test_the_background_is_the_normalised_mean_not_the_raw_counts():
    """`mean > 0` on the NORMALISED profile. the background is what a selection
    test is judged against, and it is per niche on purpose."""
    from omicstra.protocols.pathway import background

    mean = np.array([0.0, 0.0, 1.0, 2.0], dtype="float32")
    ratio = np.array([0.0, 0.0, 0.5, 0.7], dtype="float32")
    _, upd = background({"params": {"min_background": 2},
                         "variability": [("n", mean, ratio)]})
    bg = upd["background"][0][1]
    assert bg.tolist() == [False, False, True, True]


def test_a_niche_below_the_background_floor_is_dropped_with_its_cause():
    from omicstra.protocols.pathway import background

    mean = np.array([0.0, 1.0], dtype="float32")
    rec, upd = background({"params": {"min_background": 2},
                           "variability": [("n", mean, mean)]})
    assert upd["background"] == []
    assert rec.observed["reason"]["background_below_floor"] == 1


# --- declared, not assumed, at every threshold ----------------------------
def test_every_threshold_refuses_when_undeclared():
    for step, field in (("niche_qc", "min_niche_umi"),
                        ("normalise", "target_sum"),
                        ("background", "min_background"),
                        ("foreground", "top_fraction")):
        params = {k: dict(v) for k, v in VG_PARAMS.items()}
        params[step].pop(field)
        r = _vg(params)
        assert r[step]["status"] == "not_run", f"{step} ran without {field}"
        assert field in r[step]["result"], r[step]["result"]


def test_the_per_niche_cutoff_is_measured_and_recorded():
    """the FRACTION is declared; the threshold it implies is measured per niche.
    EVALUATION.md section 8: a null-derived or data-derived cut is re-derived,
    never inherited."""
    r = _vg()
    o = r["foreground"]["observed"]
    assert o["top_fraction"] == 0.25
    assert o["cutoff_median"] is not None and o["cutoff_median"] > 0


# --- the AUCell chain ------------------------------------------------------
# transcribed from `scripts/niche_aucell.py`. the notebook's prose says "AUCell
# on the variable-gene-filtered niche profile"; the script does not filter, and
# the script is right.
from omicstra.protocols.pathway import AUCELL_STEPS, aucell

AU_PARAMS = {"niche_qc": {"min_niche_umi": 1000, "min_spot_umi": 200,
                          "min_detected_spots": 3},
             "niche_sum": {}, "library_normalise": {"target_sum": 1e4},
             "pathway_net": {"min_size": 3, "max_size": 500},
             "aucell": {}, "zscore_within_group": {"group_by": "subarray"}}


def _au(params=None, G=300, n=30):
    rng = np.random.default_rng(1)
    genes = [f"G{j}" for j in range(G)]
    niches = [(f"s{i // 10}::n{i}", (rng.poisson(3, size=(7, G)) * 20).astype("float32"))
              for i in range(n)]
    ctx = {"params": params if params is not None else AU_PARAMS,
           "niches": niches, "gene_symbols": genes,
           "pathway_sets": {"P_small": genes[:2], "P_hit": genes[:40],
                            "P_big": genes[:280], "P_mid": genes[100:160]},
           "niche_groups": {nid: nid.split("::")[0] for nid, _ in niches}}
    return build_protocol(VARIABLE_GENE_STEPS[:1] + AUCELL_STEPS, "aucell").invoke(ctx)


def test_the_aucell_chain_runs_and_funnels():
    r = _au()
    assert all(v["status"] == "pass" for v in r.values()), \
        {k: v["status"] for k, v in r.items()}
    o = r["pathway_net"]["observed"]
    assert (o["n_in"], o["n_out"]) == (4, 3), "the 2-gene set is below min_size"
    assert o["reason"]["outside_size_bounds"] == 1


def test_the_score_tracks_the_ranking():
    """a set made of the top-expressed genes scores high, one made of the
    bottom scores low. without this the chain could be plumbing."""
    genes = [f"G{i}" for i in range(100)]
    profile = np.linspace(10.0, 0.0, 100)          # G0 highest, G99 lowest
    _, upd = aucell({"params": {}, "gene_symbols": genes,
                     "profiles": [("n", profile)],
                     "net": {"top": genes[:10], "bottom": genes[-10:]}})
    names, scores, _ = upd["aucell"]
    s = dict(zip(names, scores[0]))
    assert s["top"] > 0.9 and s["bottom"] < 0.1, s


def test_restricting_the_universe_changes_every_score():
    """THE reason AUCell takes no variable-gene filter. the universe is the
    denominator: rank the same profile over fewer genes and the same pathway
    gets a different number, so two niches filtered differently are no longer
    comparable - which is the one thing the scores are for.
    """
    genes = [f"G{i}" for i in range(100)]
    profile = np.linspace(10.0, 0.0, 100)
    net = {"p": genes[20:30]}

    full = aucell({"params": {}, "gene_symbols": genes, "net": net,
                   "profiles": [("n", profile)]})[1]["aucell"][1][0][0]
    # the same pathway, the same expression, ranked over half the universe
    half = aucell({"params": {}, "gene_symbols": genes[:50], "net": net,
                   "profiles": [("n", profile[:50])]})[1]["aucell"][1][0][0]
    assert abs(full - half) > 0.05, (
        f"a restricted universe gave the same score ({full:.3f} vs {half:.3f}) - "
        "if that were true, per-unit filtering would be harmless and it is not")


def test_the_zscore_is_within_a_group_never_across_it():
    """a section is a batch - its own staining, depth and capture efficiency.
    comparing a score across sections without this compares batches."""
    from omicstra.protocols.pathway import zscore_within_group

    names = ["p"]
    nids = ["a::1", "a::2", "b::1", "b::2"]
    scores = [[1.0], [3.0], [101.0], [103.0]]      # group b shifted by 100
    _, upd = zscore_within_group(
        {"params": {"group_by": "subarray"}, "aucell": (names, scores, nids),
         "niche_groups": {n: n.split("::")[0] for n in nids}})
    z = [row[0] for row in upd["aucell_z"][1]]
    assert z[:2] == pytest.approx(z[2:], abs=1e-9), \
        "a constant shift between groups must vanish, or the batch is in the score"


def test_the_raw_scores_survive_the_zscore():
    """the z-score destroys absolute level and some questions need it, so the
    raw matrix is kept beside it rather than replaced."""
    r = _au()
    assert r["aucell"]["status"] == "pass"
    assert r["zscore_within_group"]["status"] == "pass"
    # the z-score step PRODUCES a new artifact rather than replacing the raw
    # one, so both are on the context for whatever reads them next
    produced = {p for s in AUCELL_STEPS for p in s.produces}
    assert {"aucell", "aucell_z"} <= produced


def test_the_aucell_chain_refuses_its_undeclared_thresholds():
    for step, field in (("library_normalise", "target_sum"),
                        ("pathway_net", "min_size"),
                        ("zscore_within_group", "group_by")):
        params = {k: dict(v) for k, v in AU_PARAMS.items()}
        params[step].pop(field)
        r = _au(params)
        assert r[step]["status"] == "not_run", f"{step} ran without {field}"
        assert field in r[step]["result"]


def test_the_two_chains_fork_at_aggregation():
    """one collapses a niche's spots into a single profile before anything
    else; the other never collapses them. that is the fork - not normalisation,
    and not any enrichment_method flag."""
    au = {s.id for s in AUCELL_STEPS}
    vg = {s.id for s in VARIABLE_GENE_STEPS}
    assert "niche_sum" in au and "niche_sum" not in vg
    assert "variability" in vg and "variability" not in au
    assert au & vg == set(), "the chains share no step past qc"
