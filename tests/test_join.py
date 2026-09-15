"""the niche join: its declared semantics, its funnel, and the guard that stops
a score from measuring itself.

the oracle is `data/embeddings/niches_v3` - the 259 parquets every published run
reads. the acceptance numbers are that build's own: 259 subarrays, 208,786
niches, 67,131 dropped for pathway coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from omicstra.protocols.join import (
    FEATURE_COLUMNS,
    SEMANTICS,
    SUPERVISION_ONLY,
    CircularSupervision,
    Funnel,
    assert_no_circular_supervision,
    semantics_record,
)

ROOT = Path(__file__).resolve().parents[1]
JOIN = ROOT / "data" / "embeddings" / "niches_v3"

# the build's own numbers, from its manifest. pinned here so a re-port that
# quietly changes the cohort fails rather than producing a different-sized table.
N_SUBARRAYS = 259
N_NICHES = 208_786
N_DROPPED_PATHWAY = 67_131


# --- the funnel -------------------------------------------------------------
def test_a_drop_without_a_reason_is_refused():
    """the defect the funnel exists to prevent. `itemised_funnel` has been an
    absent advisory in the EDA contract since April for exactly this."""
    f = Funnel()
    with pytest.raises(ValueError, match="no reason"):
        f.add("mystery", n_in=100, n_out=40)


def test_a_stage_that_drops_nothing_needs_no_reason():
    f = Funnel()
    f.add("enumerate", 259, 259)
    assert f.stages[0].dropped == 0


def test_the_funnel_accounts_for_every_unit():
    f = Funnel()
    f.add("spots", 286_250, 275_917, "spots absent from one modality")
    f.add("pathway_coverage", 275_917, N_NICHES, "no significantly enriched pathway")
    r = f.record()
    assert r.observed["stages"][-1]["dropped"] == N_DROPPED_PATHWAY
    assert "75.7% retained" in r.result or "retained" in r.result
    assert "pathway_coverage" in r.decision, "the decision must name what was lost where"


# --- guard G6 ---------------------------------------------------------------
def test_supervision_as_an_input_feature_is_a_hard_error():
    """not a caution. the result it produces looks like a triumph - a retrieval
    score measuring whether a vector can find itself."""
    with pytest.raises(CircularSupervision, match="find itself"):
        assert_no_circular_supervision(
            ["virchow2_niche", "novae_niche", "mc_weights_niche"], "mc_weights_niche")


def test_any_supervision_column_among_the_inputs_is_caught():
    """even when it is not the declared supervision for THIS run - an evaluation
    label in the input vector is the same defect wearing a different name."""
    for col in ("mc_megacluster", "archetype", "tls", "compartment"):
        with pytest.raises(CircularSupervision):
            assert_no_circular_supervision(["virchow2_niche", col], "mc_weights_niche")


def test_the_published_feature_set_passes_the_guard():
    assert_no_circular_supervision(sorted(FEATURE_COLUMNS), "mc_weights_niche")


def test_no_column_is_both_a_feature_and_supervision():
    assert not (FEATURE_COLUMNS & SUPERVISION_ONLY)


# --- the declared semantics -------------------------------------------------
def test_every_semantic_decision_declares_its_reason():
    for name, d in SEMANTICS.items():
        assert d["as_built"], f"{name} has no as-built value"
        assert d["why"], f"{name} is declared without a reason"


def test_the_alternatives_are_declared_and_not_run():
    alts = {k: v["declared_alternative"] for k, v in SEMANTICS.items()
            if v["declared_alternative"]}
    assert "pathway_coverage" in alts, "union-vs-intersect must be on the record"
    for name, a in alts.items():
        assert a["run"] is False, f"{name}'s alternative claims to have been run"
        assert a["question"], f"{name}'s alternative states no question"


def test_the_per_modality_neighbour_lists_are_on_the_record():
    """the H&E and ST niches pool over neighbour lists computed separately.

    on the seed cohort they coincide, so nothing published moves; the declaration
    is for the cohort where they do not.
    """
    d = SEMANTICS["neighbour_lists"]
    assert d["as_built"] == "computed_per_modality"
    assert d["declared_alternative"]["value"] == "one_neighbour_list_for_both"


def test_the_union_alternative_names_what_it_would_change():
    a = SEMANTICS["pathway_coverage"]["declared_alternative"]
    assert "67,131" in a["question"], "the alternative must carry the unit count at stake"


def test_the_semantics_reach_the_ledger_as_a_record():
    r = semantics_record()
    assert r.observed["pathway_coverage"] == "drop_niche_on_nan_gpath2vec"
    assert "new arm, not a fix" in r.decision


# --- the oracle -------------------------------------------------------------
def _manifest():
    m = JOIN / "manifest.json"
    if not m.is_file():
        pytest.skip("the niche join is not on this machine")
    return json.loads(m.read_text())


def test_the_built_join_matches_its_acceptance_numbers():
    m = _manifest()
    assert m["n_subarrays_enumerated"] == N_SUBARRAYS
    assert m["n_ok"] == N_SUBARRAYS and m["n_failed"] == 0
    assert m["n_niches_total_post_intersection"] == N_NICHES
    assert m["n_niches_dropped_no_gpath2vec_coverage"] == N_DROPPED_PATHWAY


def test_the_parquet_count_matches_the_manifest():
    m = _manifest()
    assert len(list(JOIN.glob("*.parquet"))) == m["n_ok"]


def test_the_join_carries_supervision_and_features_in_one_table():
    """which is WHY the guard is a static assertion rather than a convention:
    a training script needs both, so they sit together and nothing but an
    assertion keeps them apart."""
    m = _manifest()
    pytest.importorskip("pyarrow")
    import pandas as pd

    f = min(JOIN.glob("*.parquet"))      # deterministic pick, any row set will do
    cols = set(pd.read_parquet(f).columns)
    assert FEATURE_COLUMNS <= cols, "a declared feature column is missing from the join"
    assert cols & SUPERVISION_ONLY, "supervision travels in the same table - the guard's premise"
    assert "patient_id" in cols, "the split axis must travel with every row"
    assert m["columns"]["novae_niche_normalization"] == "z-score per subarray"


# --- is the drop even across the confounder axis? ---------------------------
def test_an_even_drop_is_not_flagged():
    """a uniform drop has overdispersion ~1 by construction; flagging it would
    make the check noise."""
    from omicstra.protocols.join import assert_funnel_not_confounded, drop_distribution

    groups = {f"p{i}": (100, 400) for i in range(20)}       # exactly 25% each
    d = drop_distribution(groups)
    assert d["concentration_detected"] is False and d["overdispersion"] < 2
    assert assert_funnel_not_confounded(d) == []


def test_a_concentrated_drop_is_flagged_with_its_magnitude():
    from omicstra.protocols.join import assert_funnel_not_confounded, drop_distribution

    groups = {f"p{i}": ((380 if i < 3 else 20), 400) for i in range(20)}
    d = drop_distribution(groups)
    assert d["concentration_detected"] is True
    (caution,) = assert_funnel_not_confounded(d)
    assert "overdispersion" in caution and str(d["overdispersion"]) in caution


def test_eliminating_a_group_is_an_error_not_a_caution():
    """a drop that removes a patient entirely does not thin the cohort, it
    changes which cohort was studied."""
    from omicstra.protocols.join import (
        ConfounderBiasedFunnel,
        assert_funnel_not_confounded,
        drop_distribution,
    )

    groups = {"p1": (400, 400), "p2": (100, 400), "p3": (100, 400)}
    with pytest.raises(ConfounderBiasedFunnel, match="lost every"):
        assert_funnel_not_confounded(drop_distribution(groups))


def test_too_few_groups_says_so_rather_than_inventing_a_verdict():
    from omicstra.protocols.join import assert_funnel_not_confounded, drop_distribution

    d = drop_distribution({"only": (10, 100)})
    assert d["testable"] is False
    assert assert_funnel_not_confounded(d) == []


def test_the_real_join_drop_is_uneven_across_patients_and_says_so():
    """THE measured finding, pinned. 24.3% of niches leave the cohort at this
    stage and they do not leave evenly - the drop tracks patient_id, which is the
    axis every honest split is held out on. no patient is eliminated, so this is
    a caution that belongs in the writeup, not an error.
    """
    from collections import defaultdict

    from omicstra.protocols.join import assert_funnel_not_confounded, drop_distribution

    m = _manifest()
    by = defaultdict(lambda: [0, 0])
    for s in m["per_subarray"]:
        by[str(s["patient_id"])][0] += s["n_intersection_dropped_no_gpath2vec"]
        by[str(s["patient_id"])][1] += s["n_niches_pre_intersection"]

    d = drop_distribution({k: tuple(v) for k, v in by.items()})
    assert d["n_groups"] == 92
    assert not d["eliminated_groups"], "no patient may be removed entirely"
    assert d["min_retained_units"] > 0
    assert d["concentration_detected"] is True, (
        "this drop is known to be concentrated (overdispersion ~159x). if it stops "
        "being detectable the join changed, and the writeup's caveat is now wrong")
    assert len(assert_funnel_not_confounded(d)) == 1


def test_the_flag_claims_detection_not_uniformity():
    """`uniform: True` claimed the stronger thing - absence of evidence dressed
    as evidence of absence. a small cohort can easily fail to show concentration
    that is really there, and the field name has to admit that.
    """
    from omicstra.protocols.join import drop_distribution

    d = drop_distribution({f"p{i}": (25, 100) for i in range(4)})
    assert "uniform" not in d, "the field must not claim uniformity"
    assert d["concentration_detected"] is False


# --- the maths, against the 259-parquet oracle ------------------------------
JOIN_PICKS = ["TNBC10_CN5_D2", "TNBC1_CN1_C1", "TNBC10_CN5_E2", "TNBC22_CN11_E2"]


def test_niche_membership_pads_with_the_centre_not_a_gap():
    """a repeated centre says 'this niche is smaller than k+1', which is true.
    a zero row would be a statement about morphology no tissue made."""
    from omicstra.measures import niche_members

    assert niche_members("a", ["b", "c"], 6, {"a", "b", "c"}) == \
        ["a", "b", "c", "a", "a", "a", "a"]
    full = niche_members("a", list("bcdefg"), 6, set("abcdefg"))
    assert full == ["a", *"bcdefg"] and len(full) == 7


def test_the_zscore_accumulates_in_float64():
    """THE port bug the slice-diff caught, and the only thing that would have.

    the oracle stores each pooled vector through `.tolist()`, so its np.stack is
    float64 and mu/sd are computed in double. accumulating in float32 gives the
    same formula a different answer - 3.8e-06 on z-scores whose sd is 1.
    """
    import numpy as np

    from omicstra.measures import zscore_per_group

    rng = np.random.default_rng(0)
    m = rng.normal(size=(900, 64)).astype(np.float32)
    z = zscore_per_group(m)
    exact = (m.astype(np.float64) - m.astype(np.float64).mean(0)) / m.astype(np.float64).std(0)
    assert np.abs(z - exact.astype(np.float32)).max() == 0.0
    assert z.dtype == np.float32, "float64 is for the accumulation, not the output"


def test_a_constant_column_is_left_alone():
    """dividing by ~0 turns a constant into whatever numerical noise it carried,
    which is the kind of value that looks like signal downstream."""
    import numpy as np

    from omicstra.measures import zscore_per_group

    assert np.abs(zscore_per_group([[5.0]] * 4)).max() == 0.0


@pytest.mark.parametrize("sid", JOIN_PICKS)
def test_the_join_maths_reproduces_the_built_parquets_exactly(sid):
    """bit-identity, not a tolerance. this is indexing, stacking and a mean -
    unlike the encoder port, there is no backend-dependent kernel to excuse a
    difference, so anything but 0.00e+00 is a defect rather than float32.
    """
    pytest.importorskip("pyarrow")
    import numpy as np
    import pandas as pd

    from omicstra.measures import niche_members, stack_tokens, zscore_per_group

    V, C = ROOT / "data/embeddings/virchow2_niche", ROOT / "data/embeddings/virchow2_cell"
    N = ROOT / "data/embeddings/novae_niche_full"
    need = [V / f"{sid}_meta.tsv", C / f"{sid}.npy",
            N / f"{sid}_novae_embeddings.parquet", JOIN / f"{sid}.parquet"]
    if not all(p.is_file() for p in need):
        pytest.skip("the upstream caches are not on this machine")

    meta = pd.read_csv(need[0], sep="\t", index_col=0)
    vc, nv, join = np.load(need[1]), pd.read_parquet(need[2]), pd.read_parquet(need[3])
    row_of = {s: i for i, s in enumerate(meta.index)}
    cols = [c for c in nv.columns if c.startswith("novae_")]
    kept = list(join.index)
    mem = {s: niche_members(s, list(nv.loc[s, "neighbor_spot_ids"]), 6, set(meta.index))
           for s in kept}

    tok = np.stack([stack_tokens(vc, row_of, mem[s]).ravel() for s in kept])
    want = np.stack([np.asarray(v, np.float32) for v in join.loc[kept, "virchow2_cell_tokens"]])
    assert np.abs(tok - want).max() == 0.0, f"{sid}: token stack differs from the oracle"

    pooled = np.stack([nv[cols].reindex(mem[s]).mean(axis=0).values.astype(np.float32)
                       for s in kept])
    z = zscore_per_group(pooled)
    wz = np.stack([np.asarray(v, np.float32) for v in join.loc[kept, "novae_niche"]])
    assert np.abs(z - wz).max() == 0.0, f"{sid}: novae z-score differs from the oracle"
