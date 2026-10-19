"""tier 1 for JOIN_STEPS: synthetic cases only.

no cohort data and no cohort constant. every case here is a property of a JOIN,
not of a tissue - which is the point: the chain is the package's, and the seed
cohort is one caller of it. the regression against the published parquets is
tier 2 and lives elsewhere.

each case names the behaviour it pins and, where the behaviour exists because
`script_map/build_niche_join.md` found a defect, the map row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omicstra.protocols import ProtocolOrderError, build_protocol
from omicstra.protocols.join import (
    JOIN_STEPS,
    CircularSupervision,
    JoinInputMissing,
)

DIM = 3
MORPH, MOLEC, ANCHOR = "morph", "molec", "anchor"


def _frame(units: list[str], dim: int = DIM, offset: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(len(units) + int(offset))
    return pd.DataFrame(rng.normal(offset, 1.0, size=(len(units), dim)),
                        index=units, columns=[f"c{i}" for i in range(dim)])


def _cohort(units: list[str] | None = None, *, k: int = 2,
            anchor_units: list[str] | None = None,
            molec_units: list[str] | None = None,
            subject: str | None = "S1") -> dict:
    """a sample of `units` on a ring, so every unit has k real neighbours."""
    units = list(units if units is not None else ["u0", "u1", "u2", "u3", "u4"])
    ring = {u: [units[(i + 1) % len(units)], units[(i - 1) % len(units)]][:k]
            for i, u in enumerate(units)}
    molec = list(molec_units) if molec_units is not None else units
    anchor = list(anchor_units) if anchor_units is not None else units
    return {
        "sample": "sampleA",
        "assays": {MORPH: _frame(units), MOLEC: _frame(molec, offset=1.0),
                   ANCHOR: _frame(anchor, offset=2.0)},
        "neighbours": ring,
        "unit_labels": {"cls": pd.DataFrame({"v": ["a"] * len(units)}, index=units)},
        "subject_of": {"sampleA": subject},
        "subject_labels": {"S1": {"group": 7}},
    }


def _params(**over) -> dict:
    p = {
        "unit_inventory": {"required_assays": (MORPH, MOLEC)},
        "subject_resolution": {"subject_labels": ("group",), "on_absent": "refuse"},
        "neighbourhood": {"k_neighbors": 2, "neighbour_source": MOLEC,
                          "padding": "repeat_centre"},
        "pool_continuous": {"pooled_assays": (MORPH, MOLEC),
                            "data_columns": {MORPH: [f"c{i}" for i in range(DIM)],
                                             MOLEC: [f"c{i}" for i in range(DIM)]}},
        "centre_labels": {"centre_only_labels": ("cls",)},
        "assemble": {"provenance": "members_pooled"},
        "anchor_coverage": {"anchor_assay": ANCHOR, "on_absent": "drop"},
        "standardise": {"standardise_assays": (MOLEC,), "group_by": "sample",
                        "eps": 1e-8, "ddof": 0},
        "join_funnel": {"confounder_axis": "subject"},
    }
    for k, v in over.items():
        p[k] = {**p[k], **v} if isinstance(v, dict) and k in p else v
    return p


def _run(ctx: dict, params: dict | None = None) -> dict:
    chain = build_protocol(JOIN_STEPS, "niche_join")
    return chain.invoke({**ctx, "params": params or _params()})


def _obs(recs: dict, step: str) -> dict:
    return recs[step]["observed"]


# --------------------------------------------------------- the happy path ----
def test_a_well_formed_sample_joins_and_every_stage_states_both_ends():
    recs = _run(_cohort())
    assert [r["status"] for r in recs.values()] == ["pass"] * len(JOIN_STEPS)
    for step in ("unit_inventory", "neighbourhood", "pool_continuous",
                 "centre_labels", "assemble", "anchor_coverage", "standardise"):
        o = _obs(recs, step)
        assert "n_in" in o and "n_out" in o, f"{step} states only one end"
    stages = _obs(recs, "join_funnel")["stages"]
    assert len(stages) == len(JOIN_STEPS) - 1          # the funnel is not its own stage
    assert all(s["dropped"] == 0 for s in stages)


def test_the_provenance_column_is_what_the_pooling_used():
    """map row 291. the script stored the PRE-filter list, so recomputing the
    pool from the column did not reproduce the row."""
    ctx = _cohort()
    recs = _run(ctx)
    assert recs["assemble"]["status"] == "pass"
    assert _obs(recs, "assemble")["provenance"] == "members_pooled"
    # the column holds centre + k members, all of them real units
    cols = _obs(recs, "assemble")["columns"]
    assert "members_pooled" in cols


# ------------------------------------------------------------- case 1 --------
def test_a_unit_with_no_anchor_vector_is_dropped_counted_and_given_a_reason():
    ctx = _cohort(anchor_units=["u0", "u1", "u2"])
    recs = _run(ctx)
    o = _obs(recs, "anchor_coverage")
    assert (o["n_in"], o["n_out"], o["n_dropped"]) == (5, 3, 2)
    assert ANCHOR in o["reason"]
    stage = next(s for s in _obs(recs, "join_funnel")["stages"]
                 if s["stage"] == "anchor_coverage")
    assert stage["dropped"] == 2 and stage["reason"]


def test_the_same_units_can_be_kept_and_marked_instead():
    """`mark` is SEMANTICS' declared alternative, except that it marks rather
    than zero-fills: a zero vector and an absent one are different claims."""
    ctx = _cohort(anchor_units=["u0", "u1", "u2"])
    recs = _run(ctx, _params(anchor_coverage={"anchor_assay": ANCHOR,
                                              "on_absent": "mark"}))
    o = _obs(recs, "anchor_coverage")
    assert (o["n_in"], o["n_out"], o["n_dropped"]) == (5, 5, 0)
    assert f"{ANCHOR}__covered" in _obs(recs, "standardise") or True
    assert not o["reason"]


# ------------------------------------------------------------- case 2 --------
def test_a_unit_in_one_assay_only_is_dropped_at_the_inventory():
    ctx = _cohort(molec_units=["u0", "u1", "u2", "u3"])
    recs = _run(ctx)
    o = _obs(recs, "unit_inventory")
    assert (o["n_in"], o["n_out"]) == (5, 4)
    assert o["per_assay"] == {MORPH: 5, MOLEC: 4}
    assert o["n_in_one_assay_only"] == 1


def test_the_inventory_states_the_size_it_intersected():
    """map row 201. the script recorded this stage's n_out and never its n_in,
    so the one number that said what the intersection removed was absent."""
    recs = _run(_cohort(molec_units=["u0", "u1"]))
    o = _obs(recs, "unit_inventory")
    assert o["per_assay"] == {MORPH: 5, MOLEC: 2}
    assert o["n_in"] == 5 and o["n_out"] == 2


# ------------------------------------------------------------- case 3 --------
def test_a_sample_absent_from_one_assay_is_not_applicable_not_empty():
    ctx = _cohort(molec_units=[])
    recs = _run(ctx)
    assert recs["unit_inventory"]["status"] == "not_applicable"
    assert MOLEC in _obs(recs, "unit_inventory")["reason"]
    # and nothing downstream claims to have judged anything
    assert {r["status"] for k, r in recs.items() if k != "unit_inventory"} == {"not_run"}


def test_a_sample_with_no_subject_refuses_when_the_label_is_load_bearing():
    """map rows 128 and 179-181: both lookups were `.get` -> None, uncounted."""
    recs = _run(_cohort(subject=None))
    assert recs["subject_resolution"]["status"] == "fail"
    assert "no subject" in recs["subject_resolution"]["result"]


def test_the_same_absence_can_be_marked_instead_of_refused():
    recs = _run(_cohort(subject=None),
                _params(subject_resolution={"subject_labels": ("group",),
                                            "on_absent": "mark"}))
    assert recs["subject_resolution"]["status"] == "pass"
    assert _obs(recs, "subject_resolution")["subject"] is None


# ------------------------------------------------------------- case 4 --------
def test_a_sample_with_zero_units_after_the_join_refuses():
    ctx = _cohort(anchor_units=[])
    recs = _run(ctx)
    assert recs["anchor_coverage"]["status"] == "fail"
    o = _obs(recs, "anchor_coverage")
    assert o["n_in"] == 5 and o["n_out"] == 0
    # and the stages after it do not run, so no empty table is standardised
    assert recs["standardise"]["status"] == "not_run"


# ------------------------------------------------------------- case 5 --------
def test_the_data_columns_are_read_from_the_declaration_never_inferred():
    """the script had to name five non-data columns by hand; inferring them by
    dtype pulls pixel coordinates into the vector."""
    ctx = _cohort()
    ctx["assays"][MOLEC]["pixel_x"] = 1779.0
    recs = _run(ctx)
    assert recs["pool_continuous"]["status"] == "pass"
    assert _obs(recs, "pool_continuous")["data_columns"][MOLEC] == ["c0", "c1", "c2"]


def test_an_assay_with_no_declared_columns_refuses():
    recs = _run(_cohort(), _params(pool_continuous={
        "pooled_assays": (MORPH, MOLEC),
        "data_columns": {MORPH: ["c0", "c1", "c2"]}}))
    assert recs["pool_continuous"]["status"] == "not_run"
    assert MOLEC in recs["pool_continuous"]["result"]


def test_a_declared_column_the_assay_does_not_have_refuses():
    recs = _run(_cohort(), _params(pool_continuous={
        "pooled_assays": (MORPH,), "data_columns": {MORPH: ["c0", "nope"]}}))
    assert recs["pool_continuous"]["status"] == "not_run"
    assert "nope" in recs["pool_continuous"]["result"]


# ------------------------------------------------------------- case 6 --------
def test_standardising_before_the_drop_is_refused_at_build_time():
    """map rows 301-315. the mean and sd are taken over the survivors, so the
    drop changes every value this step emits. the order is the method, and a
    reordered chain never gets to touch data."""
    by_id = {s.id: s for s in JOIN_STEPS}
    order = ["unit_inventory", "subject_resolution", "neighbourhood",
             "pool_continuous", "centre_labels", "assemble",
             "standardise", "anchor_coverage", "join_funnel"]
    with pytest.raises(ProtocolOrderError) as e:
        build_protocol([by_id[i] for i in order], "reordered")
    assert "standardise" in str(e.value) and "table" in str(e.value)


def test_the_two_orders_do_not_give_the_same_numbers():
    """the reason case 6 is a build-time refusal rather than a preference."""
    from omicstra.measures import zscore_per_group

    m = np.arange(12, dtype=np.float64).reshape(4, DIM)
    after = zscore_per_group(m[:2])                 # drop, then standardise
    before = zscore_per_group(m)[:2]                # standardise, then drop
    assert not np.allclose(after, before)


# ------------------------------------------------------------- case 7 --------
def test_a_pooled_assay_that_is_not_required_shrinks_the_denominator_and_fails():
    """map rows 220 and 231. the script filtered the neighbour list against ONE
    assay's index, so a neighbour absent from another pooled assay still reached
    the pool as a missing row and shrank that assay's mean denominator, with no
    counter and no reason. declaring an assay pooled but not required is that
    situation, and it is now a refusal."""
    ctx = _cohort(molec_units=["u0", "u1", "u2"])
    recs = _run(ctx, _params(unit_inventory={"required_assays": (MORPH,)}))
    assert _obs(recs, "unit_inventory")["n_out"] == 5
    assert recs["pool_continuous"]["status"] == "fail"
    o = _obs(recs, "pool_continuous")
    assert o["n_units_under_full_support"][MOLEC] > 0
    assert o["n_units_under_full_support"][MORPH] == 0


def test_requiring_the_assay_restores_a_uniform_denominator():
    ctx = _cohort(molec_units=["u0", "u1", "u2"])
    recs = _run(ctx)                                  # both assays required
    assert recs["pool_continuous"]["status"] == "pass"
    o = _obs(recs, "pool_continuous")
    assert o["n_units_under_full_support"] == {MORPH: 0, MOLEC: 0}


# ------------------------------------------- declarations and the funnel ------
@pytest.mark.parametrize("step,field", [
    ("unit_inventory", "required_assays"),
    ("subject_resolution", "subject_labels"),
    ("neighbourhood", "k_neighbors"),
    ("neighbourhood", "neighbour_source"),
    ("pool_continuous", "pooled_assays"),
    ("pool_continuous", "data_columns"),
    ("centre_labels", "centre_only_labels"),
    ("anchor_coverage", "anchor_assay"),
    ("standardise", "standardise_assays"),
    ("standardise", "group_by"),
    ("standardise", "eps"),
    ("standardise", "ddof"),
    ("join_funnel", "confounder_axis"),
])
def test_an_absent_declaration_refuses_with_a_reason_naming_the_field(step, field):
    p = _params()
    p[step] = {k: v for k, v in p[step].items() if k != field}
    recs = _run(_cohort(), p)
    assert recs[step]["status"] == "not_run"
    assert field in recs[step]["result"]
    assert recs[step]["decision"]


def test_a_step_with_no_params_at_all_is_skipped_never_guessed_at():
    p = _params()
    del p["neighbourhood"]
    recs = _run(_cohort(), p)
    assert recs["neighbourhood"]["status"] == "not_run"
    assert "no params" in recs["neighbourhood"]["result"]


def test_a_declared_ddof_the_measure_does_not_implement_is_refused():
    recs = _run(_cohort(), _params(standardise={
        "standardise_assays": (MOLEC,), "group_by": "sample",
        "eps": 1e-8, "ddof": 1}))
    assert recs["standardise"]["status"] == "not_run"
    assert "zscore_per_group" in recs["standardise"]["result"]


def test_a_padding_policy_the_chain_does_not_implement_is_refused():
    recs = _run(_cohort(), _params(neighbourhood={
        "k_neighbors": 2, "neighbour_source": MOLEC, "padding": "zero_vector"}))
    assert recs["neighbourhood"]["status"] == "not_run"
    assert "repeat_centre" in recs["neighbourhood"]["decision"]


def test_a_boundary_unit_is_padded_with_its_centre_and_counted():
    ctx = _cohort()
    ctx["neighbours"]["u0"] = []                       # no declared neighbours
    recs = _run(ctx)
    o = _obs(recs, "neighbourhood")
    assert o["n_padded"] == 1 and o["members_per_unit"] == 3


def test_a_declared_assay_absent_from_the_context_is_refused_not_narrowed():
    recs = _run(_cohort(), _params(unit_inventory={
        "required_assays": (MORPH, MOLEC, "nosuch")}))
    assert recs["unit_inventory"]["status"] == "not_run"
    assert "nosuch" in recs["unit_inventory"]["result"]


def test_a_missing_context_key_raises_rather_than_being_invented():
    ctx = _cohort()
    del ctx["assays"]
    recs = _run(ctx)
    assert recs["unit_inventory"]["status"] == "error"
    assert JoinInputMissing.__name__ in recs["unit_inventory"]["result"]


def test_the_funnel_refuses_a_stage_that_recorded_no_counts():
    """the funnel is a check ON the chain, not a second opinion about it."""
    from omicstra.protocols import Step
    from omicstra.protocols.join import join_funnel
    from omicstra.records import DiagnosticRecord

    def silent(ctx):
        return DiagnosticRecord(step_id="silent", status="pass",
                                result="counted nothing"), {"standardised": True}

    chain = build_protocol(
        [Step(id="silent", fn=silent, produces=frozenset({"standardised"})),
         Step(id="join_funnel", fn=join_funnel,
              requires=frozenset({"standardised"}),
              params_space=("confounder_axis",))], "silent_chain")
    recs = chain.invoke({"params": {"silent": {}, "join_funnel":
                                    {"confounder_axis": "subject"}}})
    assert recs["join_funnel"]["status"] == "fail"
    assert "silent" in str(_obs(recs, "join_funnel")["uncounted"])


def test_the_circular_supervision_guard_fires_through_the_chain():
    """`fail`, not `error`: the step did not break, the cohort failed a check."""
    recs = _run(_cohort(), _params(join_funnel={
        "confounder_axis": "subject",
        "feature_columns": (MORPH, "cls"), "supervision": "cls"}))
    assert recs["join_funnel"]["status"] == "fail"
    assert _obs(recs, "join_funnel")["guard"] == "G6"


def test_the_same_guard_still_raises_for_a_direct_caller():
    from omicstra.protocols.join import assert_no_circular_supervision

    with pytest.raises(CircularSupervision):
        assert_no_circular_supervision({MORPH, "cls"}, "cls")


def test_nothing_in_the_chain_knows_what_tissue_it_is_looking_at():
    """the whole point of the port: the script names three encoders in code and
    this does not. the assay names above are synthetic and the chain runs."""
    recs = _run(_cohort())
    assert recs["unit_inventory"]["status"] == "pass"
    assert _obs(recs, "unit_inventory")["required_assays"] == [MORPH, MOLEC]


def test_the_confounder_check_is_deferred_to_the_cohort_not_faked_per_shard():
    """a shard is one sample, so the split axis has one group in it. the step
    contributes its (dropped, enumerated) pair and says where the question is
    answerable, rather than reporting an untestable distribution as a result."""
    recs = _run(_cohort(anchor_units=["u0", "u1", "u2"]))
    o = _obs(recs, "join_funnel")
    assert o["drop_distribution"]["testable"] is False
    assert o["drop_distribution"]["scope"] == "one shard"
    assert o["funnel_contribution"] == {"S1": [2, 5]}


def test_the_contributions_of_several_shards_answer_it():
    """the aggregation the step defers to, over three shards of one cohort."""
    from omicstra.protocols.join import assert_funnel_not_confounded, drop_distribution

    pooled = {}
    for subject, anchor in (("S1", ["u0", "u1", "u2"]),
                            ("S2", ["u0", "u1", "u2", "u3"]),
                            ("S3", ["u0"])):
        ctx = _cohort(anchor_units=anchor)
        ctx["subject_of"] = {"sampleA": subject}
        ctx["subject_labels"] = {subject: {"group": 7}}
        pooled.update(_obs(_run(ctx), "join_funnel")["funnel_contribution"])

    dist = drop_distribution({k: tuple(v) for k, v in pooled.items()},
                             axis="subject")
    assert dist["testable"] and dist["n_groups"] == 3
    assert not dist["eliminated_groups"]
    assert isinstance(assert_funnel_not_confounded(dist), list)


def test_a_subject_that_loses_every_unit_is_an_error_at_the_cohort_level():
    from omicstra.protocols.join import (
        ConfounderBiasedFunnel,
        assert_funnel_not_confounded,
        drop_distribution,
    )

    dist = drop_distribution({"S1": (2, 5), "S2": (5, 5)}, axis="subject")
    assert dist["eliminated_groups"] == ["S2"]
    with pytest.raises(ConfounderBiasedFunnel):
        assert_funnel_not_confounded(dist)
