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
