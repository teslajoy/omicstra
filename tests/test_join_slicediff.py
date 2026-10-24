"""JOIN_STEPS against the parquets it was ported from.

tier 2. the 259 published parquets are the ORACLE - every alignment arm in the
grid read them - so the port is verified by diffing against them rather than by
re-deriving them. this runs where the cohort exists and skips elsewhere: every
input it needs is git-ignored.

the criterion is EXACT on the vectors. these are the same float32 arrays passed
through the same measures, so there is no tolerance to declare - a port that
pooled a different set of members, or standardised in a different order, differs
in the last bits and that is a real difference. the one declared tolerance is
`tls`, where the script stored a python float and the chain stores float32, so
the oracle is cast before the comparison rather than the comparison loosened.

the cohort's declarations live in THIS FILE. the package holds no default for
any of them, which is the whole point of the port: `DECLARED` below is what the
seed cohort's project.json will carry, and a second cohort carries its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omicstra.protocols import build_protocol
from omicstra.protocols.join import JOIN_STEPS

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data/embeddings"
BIO = EMB / "biological_signals"
ORACLE = EMB / "niches_v3"
GPATH = (EMB / "gpath2vec/fisher_madmean_low_dim512_e5_s1234"
             / "fisher_madmean_low_cluster_embeddings.parquet")

# the assay names are the oracle's OWN column names, so a mismatch is visible in
# the diff rather than needing a translation table.
HE, CELL, ST = "virchow2_niche", "virchow2_cell_tokens", "novae_niche"
PATH, TLS, MCW = "gpath2vec_niche", "tls", "mc_weights_niche"

# the five non-data columns of the pathway parquet, named by the build script on
# its own line 71. inferring them by dtype pulls pixel coordinates into the
# vector, which is case 5 of the tier-1 suite.
GPATH_META = ("subarray", "patient", "spot_id", "pixel_x", "pixel_y")

# four sections chosen for what they exercise, not for being representative:
# full label coverage; no tls table at all; 454 units with no discrete label;
# the lowest retention in the cohort at 10.9%.
SECTIONS = ["TNBC1_CN1_C1", "TNBC10_CN5_D2", "TNBC10_CN5_E1", "TNBC26_CN13_E2"]

# the cohort's declarations. every one of these is read from the build script or
# measured out of the artifact - none is a default, and none lives in the wheel.
DECLARED = {
    "k_neighbors": 6,                 # build_niche_join.py:68
    "required_assays": (HE, ST),      # :201, the canonical unit set
    "pooled_assays": (ST, TLS, MCW),  # :231, :241-251, :253-258
    "per_unit_assays": (HE, PATH),    # :285 pre-pooled at extraction; :234-238
    "token_assays": (CELL,),          # :227-228, the 7-token stack
    "centre_only_labels": ("compartment", "mc_megacluster"),   # :260-276
    "anchor_assay": PATH,             # :301-305, the intersection split
    "subject_labels": ("archetype",),
    # only the ST features' denominator is load-bearing. the supervision table
    # covers 1,039 of 1,075 units on the first section, so its pooled vector
    # averages over a varying denominator - a true property of the published
    # artifact, which the script never recorded and this chain now does.
    "require_full_support": (ST,),
    "eps": 1e-8,                      # :312
    "ddof": 0,                        # :311, numpy's default
}


def _present() -> bool:
    return (ORACLE.is_dir() and GPATH.is_file()
            and all((ORACLE / f"{s}.parquet").is_file() for s in SECTIONS)
            and (BIO / "mc_weights.tsv").is_file())


pytestmark = pytest.mark.skipif(
    not _present(), reason="the cohort's embeddings are git-ignored; tier 2 "
                           "runs where they exist")


# the inputs are tens of megabytes and every test below needs the same ones, so
# a section is read once per session. the ctx is handed out by reference and the
# steps never mutate the assay frames - `assemble` builds new rows - so sharing
# it is safe. `standardise` DOES mutate the rows it is given, which is why the
# table is rebuilt per walk rather than cached with the ctx.
_CACHE: dict[str, tuple[dict, dict, pd.DataFrame]] = {}


def _ctx(sub: str) -> tuple[dict, dict, pd.DataFrame]:
    """this cohort's inputs in the chain's shape. a cohort adapter, not package code."""
    if sub not in _CACHE:
        _CACHE[sub] = _read(sub)
    return _CACHE[sub]


def _read(sub: str) -> tuple[dict, dict, pd.DataFrame]:
    meta = pd.read_csv(EMB / f"virchow2_niche/{sub}_meta.tsv", sep="\t", index_col=0)
    vn = np.load(EMB / f"virchow2_niche/{sub}.npy")
    vc = np.load(EMB / f"virchow2_cell/{sub}.npy")
    nv = pd.read_parquet(EMB / f"novae_niche_full/{sub}_novae_embeddings.parquet")
    st_cols = [c for c in nv.columns if c.startswith("novae_")]

    gp = pd.read_parquet(GPATH, filters=[("subarray", "==", sub)])
    gp_cols = [c for c in gp.columns if c not in GPATH_META]

    slide_pos = sub.split("_", 1)[1]          # the cohort's second key space
    mcw = pd.read_csv(BIO / "mc_weights.tsv", sep="\t")
    mcw = mcw[mcw.subarray == slide_pos]
    mclab = pd.read_csv(BIO / "mc_labels.tsv", sep="\t")
    mclab = mclab[mclab.subarray == slide_pos]
    tls = pd.read_csv(BIO / "tls_scores.tsv", sep="\t")
    tls = tls[tls.subarray_id == sub]
    morph = pd.read_csv(BIO / "morphology_labels.tsv", sep="\t")
    morph = morph[morph.subarray_id == sub]

    oracle = pd.read_parquet(ORACLE / f"{sub}.parquet")
    subject = int(mcw.patient_id.iloc[0])
    mc_cols = [f"mc{k}" for k in range(1, 15)]

    assays = {
        HE: pd.DataFrame(vn, index=meta.index,
                         columns=[f"he{i}" for i in range(vn.shape[1])]),
        CELL: pd.DataFrame(vc, index=meta.index,
                           columns=[f"ce{i}" for i in range(vc.shape[1])]),
        ST: nv[st_cols],
        PATH: gp.set_index("spot_id")[gp_cols],
        TLS: tls.set_index("spot_id")[["tls_score"]],
        MCW: mcw.set_index("spot_id")[mc_cols],
    }
    ctx = {
        "sample": sub,
        "assays": assays,
        "neighbours": {s: list(nv.loc[s, "neighbor_spot_ids"]) for s in nv.index},
        "unit_labels": {
            "compartment": morph.set_index("spot_id")[["dominant_18class"]],
            "mc_megacluster": mclab.set_index("spot_id")[["megacluster"]],
        },
        "subject_of": {sub: subject},
        "subject_labels": {subject: {"archetype": int(oracle.archetype.iloc[0])}},
    }
    cols = {a: list(f.columns) for a, f in assays.items()}
    d = DECLARED
    params = {
        "unit_inventory": {"required_assays": d["required_assays"]},
        "subject_resolution": {"subject_labels": d["subject_labels"],
                               "on_absent": "refuse"},
        "neighbourhood": {"k_neighbors": d["k_neighbors"],
                          "neighbour_source": ST, "padding": "repeat_centre"},
        "pool_continuous": {"pooled_assays": d["pooled_assays"],
                            "data_columns": {a: cols[a] for a in d["pooled_assays"]},
                            "require_full_support": d["require_full_support"]},
        "carry_per_unit": {"per_unit_assays": d["per_unit_assays"],
                           "data_columns": {a: cols[a] for a in d["per_unit_assays"]}},
        "stack_unit_tokens": {"token_assays": d["token_assays"],
                              "data_columns": {a: cols[a] for a in d["token_assays"]}},
        "centre_labels": {"centre_only_labels": d["centre_only_labels"]},
        "assemble": {"provenance": "members_pooled"},
        "anchor_coverage": {"anchor_assay": d["anchor_assay"], "on_absent": "drop"},
        "standardise": {"standardise_assays": (ST,), "group_by": "sample",
                        "eps": d["eps"], "ddof": d["ddof"]},
        "join_funnel": {"confounder_axis": "subject"},
    }
    return ctx, params, oracle


def _walk(ctx: dict, params: dict) -> tuple[dict, list[dict]]:
    """the chain's records, and the table it produced.

    `build_protocol` hands back records by design - a chain is judged on what it
    recorded. the table is threaded through ctx, so the steps are walked a
    second time to hold it. both paths run the same functions on the same ctx.
    """
    recs = build_protocol(JOIN_STEPS, "niche_join").invoke({**ctx, "params": params})
    c = dict(ctx)
    for st in JOIN_STEPS:
        if st.id not in params:
            continue
        out = st.fn({**c, "params": params[st.id]})
        if isinstance(out, tuple):
            c.update(out[1])
    return recs, c.get("table", [])


def _eq(have, want) -> tuple[int, float]:
    """rows differing, and the worst absolute difference. nan == nan."""
    bad, worst = 0, 0.0
    for h, w in zip(have, want):
        h = np.asarray(h, np.float32).ravel()
        w = np.asarray(w, np.float32).ravel()
        assert h.shape == w.shape, f"shape {h.shape} vs oracle {w.shape}"
        both_nan = np.isnan(h) & np.isnan(w)
        d = np.abs(h - w)
        d[both_nan] = 0.0
        m = float(np.nanmax(d)) if d.size else 0.0
        worst = max(worst, m)
        bad += int(m > 0)
    return bad, worst


@pytest.mark.parametrize("sub", SECTIONS)
def test_every_step_passes_on_a_published_section(sub):
    recs, _ = _walk(*_ctx(sub)[:2])
    failed = {k: r["result"] for k, r in recs.items() if r["status"] != "pass"}
    assert not failed, f"{sub}: {failed}"


@pytest.mark.parametrize("sub", SECTIONS)
def test_the_surviving_unit_set_is_identical(sub):
    ctx, params, oracle = _ctx(sub)
    _, table = _walk(ctx, params)
    assert [r["unit_id"] for r in table] == list(oracle.index)


@pytest.mark.parametrize("sub", SECTIONS)
@pytest.mark.parametrize("col", [HE, CELL, PATH, ST, MCW])
def test_every_vector_column_is_exact(sub, col):
    """no tolerance: the same float32 arrays through the same measures."""
    ctx, params, oracle = _ctx(sub)
    _, table = _walk(ctx, params)
    bad, worst = _eq([r[col] for r in table], list(oracle[col]))
    assert (bad, worst) == (0, 0.0), f"{sub}/{col}: {bad} rows differ, worst {worst:.3e}"


@pytest.mark.parametrize("sub", SECTIONS)
def test_the_scalar_column_is_exact_at_float32(sub):
    """the script stored a python float; the chain stores float32. the oracle is
    CAST rather than the comparison loosened - a tolerance would hide a real
    difference of the same size."""
    ctx, params, oracle = _ctx(sub)
    _, table = _walk(ctx, params)
    have = np.asarray([r[TLS][0] for r in table], np.float32)
    want = np.asarray(oracle[TLS].to_numpy(), np.float32)
    assert np.array_equal(have, want, equal_nan=True)


@pytest.mark.parametrize("sub", SECTIONS)
def test_every_label_column_matches_including_its_absences(sub):
    ctx, params, oracle = _ctx(sub)
    _, table = _walk(ctx, params)
    for col, absent_in_oracle in (("compartment", (None,)), ("mc_megacluster", (-1,))):
        for r in table:
            o = oracle.at[r["unit_id"], col]
            if r[col] is None:
                # the chain marks absence in its own column; the script wrote a
                # sentinel. the two agree about WHICH units are absent.
                assert (o in absent_in_oracle or (isinstance(o, float) and np.isnan(o))), \
                    f"{sub}/{col}: chain says absent, oracle says {o!r}"
                assert r[f"{col}__present"] is False
            else:
                assert r[col] == o, f"{sub}/{col}/{r['unit_id']}: {r[col]!r} vs {o!r}"


@pytest.mark.parametrize("sub", SECTIONS)
def test_the_funnel_matches_the_counts_the_build_recorded(sub):
    """the manifest is an independent witness: it was written by the script, so
    agreeing with it is not the chain agreeing with itself."""
    man = json.loads((ORACLE / "manifest.json").read_text())
    row = next(r for r in man["per_subarray"] if r["subarray"] == sub)
    recs, _ = _walk(*_ctx(sub)[:2])

    inv = recs["unit_inventory"]["observed"]
    anc = recs["anchor_coverage"]["observed"]
    nbr = recs["neighbourhood"]["observed"]
    assert inv["n_out"] == row["n_niches_pre_intersection"]
    assert anc["n_dropped"] == row["n_intersection_dropped_no_gpath2vec"]
    assert anc["n_out"] == row["n_niches"]
    assert nbr["n_padded"] == row["n_boundary_niches_padded"]
    assert recs["carry_per_unit"]["observed"]["n_absent_per_assay"][PATH] == \
        row["n_missing_gpath2vec"]


@pytest.mark.parametrize("sub", SECTIONS)
def test_the_funnel_accounts_for_every_unit(sub):
    recs, _ = _walk(*_ctx(sub)[:2])
    stages = recs["join_funnel"]["observed"]["stages"]
    assert all(s["reason"] for s in stages if s["dropped"]), \
        "a stage dropped units with no reason"
    first, last = stages[0], stages[-1]
    assert first["n_in"] - last["n_out"] == sum(s["dropped"] for s in stages)


def test_the_cohort_declares_the_published_parameters_and_the_package_none():
    """the port's claim: every number above is the cohort's, read from the build
    script, and no step carries one of its own."""
    import inspect

    from omicstra.protocols import join

    src = inspect.getsource(join)
    for n in (6, 512, 1280, 1000, 0.25):
        assert f"= {n}" not in src, f"{n} reached the package"
    assert DECLARED["k_neighbors"] == 6 and DECLARED["ddof"] == 0


# ----------------------------------- the WIRED path, through run_niche_join ---
# everything above drives JOIN_STEPS with an adapter written in this file. these
# go through `run_niche_join(compute=True)` - the function the align graph calls
# - with the arrays assembled by the DECLARED reader. if the two disagree, the
# reachable path builds something else.
WIRED = ["TNBC1_CN1_C1", "TNBC26_CN13_E2"]


def _wired(tmp_path_factory):
    if "wired" not in _CACHE:
        from omicstra.contracts.project import ProjectConfig
        from omicstra.protocols.align import run_niche_join

        out = tmp_path_factory.mktemp("wired_join")
        nj, rec = run_niche_join(ProjectConfig.load("tnbc-92"), "tnbc-92",
                                 compute=True, out_dir=out, subarrays=WIRED)
        _CACHE["wired"] = (out, nj, rec)
    return _CACHE["wired"]


def test_the_wired_join_builds_without_the_cohort_specific_script(tmp_path_factory):
    """`compute=True` used to shell out to scripts/build_niche_join.py, which
    names this cohort 27 times. it runs JOIN_STEPS now."""
    out, nj, rec = _wired(tmp_path_factory)
    assert rec.params.get("resolves_only") is False
    assert nj.n_subarrays == len(WIRED)
    man = json.loads((out / "manifest.json").read_text())
    assert man["built_by"] == "omicstra.protocols.join.JOIN_STEPS"
    assert man["sources"]["declared_by"] == "project.json#join_inputs"


@pytest.mark.parametrize("sub", WIRED)
@pytest.mark.parametrize("col", [HE, CELL, PATH, ST, MCW])
def test_the_wired_build_is_exact_against_the_published_parquet(
        tmp_path_factory, sub, col):
    out, _, _ = _wired(tmp_path_factory)
    mine = pd.read_parquet(out / f"{sub}.parquet")
    oracle = pd.read_parquet(ORACLE / f"{sub}.parquet")
    assert list(mine.index) == list(oracle.index)
    bad, worst = _eq([mine.at[u, col] for u in oracle.index],
                     list(oracle.loc[oracle.index, col]))
    assert (bad, worst) == (0, 0.0), f"{sub}/{col}: {bad} rows, worst {worst:.3e}"


@pytest.mark.parametrize("sub", WIRED)
def test_the_wired_scalar_column_is_exact_at_float32(tmp_path_factory, sub):
    """and it is scalar even where the label table does not cover the sample.
    the column selection did not apply to an empty slice, so an uncovered
    sample came back three wide and the pooled value was 3 NaNs where the
    artifact has one."""
    out, _, _ = _wired(tmp_path_factory)
    mine = pd.read_parquet(out / f"{sub}.parquet")
    oracle = pd.read_parquet(ORACLE / f"{sub}.parquet")
    got = np.asarray([mine.at[u, TLS] for u in oracle.index], np.float32).ravel()
    want = np.asarray(oracle.loc[oracle.index, TLS].to_numpy(), np.float32)
    assert got.shape == want.shape
    assert np.array_equal(got, want, equal_nan=True)


@pytest.mark.parametrize("sub", WIRED)
def test_the_wired_build_marks_absence_where_the_script_wrote_a_sentinel(
        tmp_path_factory, sub):
    """the ONE intended difference. the script wrote -1 into the label's own
    value space - open defect #7 in docs/open_items.md - and the chain writes
    NaN plus a `{label}__present` column. the two must agree about WHICH units
    are absent, and about every unit that is present."""
    out, _, _ = _wired(tmp_path_factory)
    mine = pd.read_parquet(out / f"{sub}.parquet").loc[
        pd.read_parquet(ORACLE / f"{sub}.parquet").index]
    oracle = pd.read_parquet(ORACLE / f"{sub}.parquet")

    sentinel = oracle["mc_megacluster"] == -1
    assert (mine["mc_megacluster"].isna() == sentinel).all()
    present = ~sentinel
    assert (mine.loc[present, "mc_megacluster"].astype(int)
            == oracle.loc[present, "mc_megacluster"]).all()
    assert int((~mine["mc_megacluster__present"]).sum()) == int(sentinel.sum())


def test_the_token_block_shape_is_recorded_so_it_can_be_restored(tmp_path_factory):
    """8,960 numbers on disk are 7 x 1,280, and nothing in the column says so."""
    out, _, _ = _wired(tmp_path_factory)
    man = json.loads((out / "manifest.json").read_text())
    assert man["flattened_shapes"] == {CELL: [7, 1280]}


def test_the_wired_funnel_matches_the_published_counts(tmp_path_factory):
    out, _, _ = _wired(tmp_path_factory)
    man = json.loads((out / "manifest.json").read_text())
    published = json.loads((ORACLE / "manifest.json").read_text())["per_subarray"]
    for row in man["per_subarray"]:
        want = next(r for r in published if r["subarray"] == row["subarray"])
        assert row["n_niches"] == want["n_niches"]
        assert row["n_niches_pre_intersection"] == want["n_niches_pre_intersection"]
        assert (row["n_intersection_dropped_no_gpath2vec"]
                == want["n_intersection_dropped_no_gpath2vec"])


def test_a_cohort_that_declares_no_join_params_is_refused(tmp_path):
    from omicstra.contracts.project import ProjectConfig
    from omicstra.protocols.align import ComputeUnavailable, run_niche_join

    cfg = ProjectConfig.load("tnbc-92")
    with pytest.raises(ComputeUnavailable, match="join_params"):
        import omicstra.protocols.align as A
        orig = A._cohort
        A._cohort = lambda pid=None: {}
        try:
            run_niche_join(cfg, "tnbc-92", compute=True, out_dir=tmp_path,
                           subarrays=["TNBC1_CN1_C1"])
        finally:
            A._cohort = orig


# --------------------------------- every oracle column must be accounted for --
# the wired build silently omitted `patient_id` and `archetype` for a while,
# because the subject source was never declared and the tests above compare a
# NAMED LIST of columns. a list cannot catch an absence. so this maps every
# column of the published parquet onto the wired one, and a column in neither
# the map nor the reproduced set fails.
ORACLE_TO_WIRED = {
    # reproduced under the same name
    "virchow2_niche": "virchow2_niche",
    "virchow2_cell_tokens": "virchow2_cell_tokens",
    "gpath2vec_niche": "gpath2vec_niche",
    "novae_niche": "novae_niche",
    "mc_weights_niche": "mc_weights_niche",
    "tls": "tls",
    "compartment": "compartment",
    "mc_megacluster": "mc_megacluster",
    # renamed, because the chain names things for what they are rather than for
    # one cohort's vocabulary
    "patient_id": "subject",
    "archetype": "spatial_archetype",
    "subarray": "sample",
    # CORRECTED, not renamed: the script stored the pre-filter neighbour list,
    # so recomputing the pool from it did not reproduce the row. the chain
    # stores the members it actually pooled.
    "neighbor_spot_ids": "members_pooled",
}


def test_every_published_column_is_accounted_for(tmp_path_factory):
    out, _, _ = _wired(tmp_path_factory)
    oracle = pd.read_parquet(ORACLE / f"{WIRED[0]}.parquet")
    mine = pd.read_parquet(out / f"{WIRED[0]}.parquet")

    unmapped = sorted(set(oracle.columns) - set(ORACLE_TO_WIRED))
    assert not unmapped, f"published columns with no stated counterpart: {unmapped}"

    absent = sorted(c for c, w in ORACLE_TO_WIRED.items() if w not in mine.columns)
    assert not absent, f"mapped to a column the wired build does not have: {absent}"


def test_the_renamed_subject_columns_carry_the_published_values(tmp_path_factory):
    """the omission these two tests exist for: declared nowhere, so built
    silently without them."""
    out, _, _ = _wired(tmp_path_factory)
    for sub in WIRED:
        oracle = pd.read_parquet(ORACLE / f"{sub}.parquet")
        mine = pd.read_parquet(out / f"{sub}.parquet").loc[oracle.index]
        assert (mine["subject"].astype(int)
                == oracle["patient_id"].astype(int)).all()
        assert (mine["spatial_archetype"].astype(int)
                == oracle["archetype"].astype(int)).all()
        assert (mine["sample"] == oracle["subarray"]).all()


def test_the_provenance_column_is_the_corrected_one_not_a_rename(tmp_path_factory):
    """`members_pooled` is what the pooling used; the script's
    `neighbor_spot_ids` was the raw declared list, before its own filter and
    before padding, so a reader recomputing from it did not get the row back."""
    out, _, _ = _wired(tmp_path_factory)
    oracle = pd.read_parquet(ORACLE / f"{WIRED[0]}.parquet")
    mine = pd.read_parquet(out / f"{WIRED[0]}.parquet").loc[oracle.index]
    k = oracle.index[0]
    assert len(mine.at[k, "members_pooled"]) == 7, "centre plus k members"
    assert len(oracle.at[k, "neighbor_spot_ids"]) == 6, "the script stored k"
    assert mine.at[k, "members_pooled"][0] == k, "centre first"
