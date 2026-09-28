"""the ST arm: the gene axis, the compute, and the arm a handle belongs to.

`test_st_graph.py` covers the edge scale as a declaration. this covers the other
half of what the ST encoder needs before it can run - the gene identifier space -
and the port that turns the two declarations into vectors.

the acceptance against the cache lives at the bottom and skips where the cohort
is absent, which is every clone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omicstra.adapters.canonical import observed_gene_id
from omicstra.protocols.encode import GeneAxis, StGraph, apply_gene_axis, run_one_st

ROOT = Path(__file__).resolve().parents[1]
# the DECLARATION, which is the annotation release and its policy - and nothing
# else. `have` is observed from the ingest record and `strip_version` is derived
# from it, so neither appears here.
DECLARED = {"map": "m.tsv", "map_key": "ensembl", "map_value": "symbol",
            "duplicates": "sum", "unmapped": "drop"}
OBSERVED = "ensembl_versioned"        # what the ingest recorded for a section


# --- the declaration -------------------------------------------------------
def test_an_unrecorded_gene_axis_is_refused_not_assumed():
    """"the ingest did not record it" is not "it is on symbols".

    THE correction. an earlier version asked the cohort to DECLARE which space
    its sections were indexed by, and that declaration was wrong for 68 of 108
    sections of the second cohort - which pools three studies that disagree. the
    space is a per-section fact the ingest observes; only the release is declared.
    """
    with pytest.raises(ValueError, match="ingest did not record"):
        GeneAxis.resolve({"gene_mapping": DECLARED}, want="symbol", observed=None)


def test_a_section_already_on_the_encoders_axis_needs_no_declaration():
    """40 of that cohort's sections are already on symbols. they map nothing, so
    there is no release to choose and requiring one would be a question with a
    single answer."""
    axis = GeneAxis.resolve({}, want="symbol", observed="symbol")
    assert axis.is_identity() and axis.map_path is None


def test_a_section_needing_a_map_with_none_declared_is_refused():
    with pytest.raises(ValueError, match="declares no gene_mapping"):
        GeneAxis.resolve({}, want="symbol", observed=OBSERVED)


def test_a_table_keyed_on_the_wrong_space_cannot_map_a_section():
    """a cohort with sections in more than one space needs a table per space, and
    the sections it does not cover are skipped naming this."""
    with pytest.raises(ValueError, match="cannot map it"):
        GeneAxis.resolve({"gene_mapping": dict(DECLARED, map_key="entrez")},
                         want="symbol", observed=OBSERVED)


def test_strip_version_is_derived_from_the_record_not_declared():
    """the ingest already distinguishes ensembl_versioned from ensembl, so cutting
    at the dot follows from two facts. a third declaration could disagree."""
    assert GeneAxis.resolve({"gene_mapping": DECLARED}, want="symbol",
                            observed="ensembl_versioned").strip_version is True
    assert GeneAxis.resolve({"gene_mapping": DECLARED}, want="symbol",
                            observed="ensembl").strip_version is False


def test_the_cohort_and_the_encoder_must_agree_on_the_target():
    """the two halves are declared in the two places that own them. a cohort
    asserting it supplies what the encoder wants, and being believed, is what
    this pair exists to stop."""
    with pytest.raises(ValueError, match="the table does not lead"):
        GeneAxis.resolve({"gene_mapping": dict(DECLARED, map_value="entrez")},
                         want="symbol", observed=OBSERVED)


@pytest.mark.parametrize("field,bad", [("duplicates", "concat"), ("unmapped", "invent")])
def test_the_closed_sets_are_closed(field, bad):
    with pytest.raises(ValueError, match=field):
        GeneAxis.resolve({"gene_mapping": dict(DECLARED, **{field: bad})},
                         want="symbol", observed=OBSERVED)


def test_a_mapping_with_no_table_is_refused():
    d = {k: v for k, v in DECLARED.items() if k != "map"}
    with pytest.raises(ValueError, match="no map table"):
        GeneAxis.resolve({"gene_mapping": d}, want="symbol", observed=OBSERVED)


def test_the_declaration_reaches_the_record():
    """two sections on different axes must not read alike, and a run that changed
    `duplicates` is a different run."""
    axis = GeneAxis.resolve({"gene_mapping": DECLARED}, want="symbol", observed=OBSERVED)
    p = axis.params()
    assert p["gene_axis_duplicates"] == "sum" and p["gene_axis_map"] == "m.tsv"
    assert p["gene_axis_strip_version"] is True


# --- the transform ---------------------------------------------------------
def _tiny(tmp_path, dtype="int32"):
    """three spots, four ensembl ids, two of which are one symbol."""
    import anndata as ad
    import pandas as pd
    from scipy.sparse import csr_matrix

    X = csr_matrix(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=dtype))
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=["s1", "s2", "s3"]),
                   var=pd.DataFrame(index=["ENSG1.9", "ENSG2.3", "ENSG3.1", "ENSG4.2"]))
    tbl = tmp_path / "m.tsv"
    tbl.write_text("ensembl_id\tgene_symbol\nENSG1\tAAA\nENSG2\tBBB\nENSG3\tAAA\n")
    return a, tbl


def _axis(tbl, observed=OBSERVED, **over):
    return GeneAxis.resolve({"gene_mapping": dict(DECLARED, map=str(tbl), **over)},
                            want="symbol", observed=observed)


def test_duplicates_are_summed_and_the_unmapped_dropped(tmp_path):
    a, tbl = _tiny(tmp_path)
    out, rec = apply_gene_axis(a, _axis(tbl))
    assert list(out.var_names) == ["AAA", "BBB"], "sorted target names, as a groupby gives"
    # ENSG1 + ENSG3 -> AAA; ENSG2 -> BBB; ENSG4 unmapped and dropped
    assert out.X.toarray().tolist() == [[4, 2], [12, 6], [20, 10]]
    assert rec["n_unmapped"] == 1 and rec["n_collapsed"] == 1


def test_the_counts_dtype_survives_the_collapse(tmp_path):
    """THE finding, and it is not cosmetic. a float64 indicator upcasts an int32
    matrix; the sums stay exact, and then the ENCODER's own normalize_total and
    log1p run at a different precision. that reached the embeddings as a 4.8e-07
    disagreement with the cache - exactly the size someone accepts as float noise
    and records as a tolerance. it was a type defect with a fix.
    """
    a, tbl = _tiny(tmp_path, dtype="int32")
    out, _ = apply_gene_axis(a, _axis(tbl))
    assert out.X.dtype == np.int32, f"counts came back as {out.X.dtype}"


def test_mean_and_first_are_different_matrices(tmp_path):
    """which is why `duplicates` has no default."""
    a, tbl = _tiny(tmp_path)
    mean = apply_gene_axis(a, _axis(tbl, duplicates="mean"))[0].X.toarray()
    first = apply_gene_axis(a, _axis(tbl, duplicates="first"))[0].X.toarray()
    assert mean[0].tolist() == [2.0, 2.0]          # (1+3)/2, 2
    assert first[0].tolist() == [1.0, 2.0]         # ENSG1 wins AAA
    assert not np.array_equal(mean, first)


def test_keeping_the_unmapped_keeps_them_under_their_own_id(tmp_path):
    a, tbl = _tiny(tmp_path)
    out, rec = apply_gene_axis(a, _axis(tbl, unmapped="keep"))
    assert "ENSG4" in set(out.var_names)
    assert rec["n_unmapped"] == 1


def test_a_section_recorded_as_unversioned_maps_nothing_here(tmp_path):
    """`ENSG1.9` is not a key in a table of `ENSG1`, so a section the ingest
    recorded as plain `ensembl` while its ids carry versions maps nothing. the
    record and the data have to agree; this is what disagreement looks like."""
    a, tbl = _tiny(tmp_path)
    _, rec = apply_gene_axis(a, _axis(tbl, observed="ensembl"))
    assert rec["n_unmapped"] == 4


def test_an_identity_axis_returns_the_matrix_untouched(tmp_path):
    a, _ = _tiny(tmp_path)
    axis = GeneAxis.resolve({}, want="symbol", observed="symbol")
    out, rec = apply_gene_axis(a, axis)
    assert out is a and rec["route"] == "identity"


# --- the compute refuses on its own account --------------------------------
def test_an_image_encoder_is_not_an_st_encoder(tmp_path):
    a, tbl = _tiny(tmp_path)
    g = StGraph(method="delaunay", radius_cap_px=None, scale_to_microns=1.0)
    with pytest.raises(ValueError, match="declares no gene axis"):
        run_one_st(a, np.zeros((3, 2)), g, _axis(tbl), encoder="virchow2")


def test_a_coordinate_count_that_does_not_match_the_spots_is_refused(tmp_path):
    """the adapter joins them by spot_id, so a mismatch means the join was
    skipped - not that a spot is missing."""
    pytest.importorskip("novae", reason="the encode extra is not installed")
    a, tbl = _tiny(tmp_path)
    g = StGraph(method="delaunay", radius_cap_px=None, scale_to_microns=1.0)
    with pytest.raises(ValueError, match="coordinates against"):
        run_one_st(a, np.zeros((2, 2)), g, _axis(tbl), encoder="novae")


def test_only_the_declared_graph_method_is_implemented(tmp_path):
    """a radius cap is a declared ALTERNATIVE arm with its own run id, not a
    variant of this one."""
    pytest.importorskip("novae", reason="the encode extra is not installed")
    a, tbl = _tiny(tmp_path)
    g = StGraph(method="knn", radius_cap_px=None, scale_to_microns=1.0)
    with pytest.raises(ValueError, match="only 'delaunay' is"):
        run_one_st(a, np.zeros((3, 2)), g, _axis(tbl), encoder="novae")


# --- the shard carries its own declarations --------------------------------
def test_a_shard_carries_the_graph_and_the_axis(tmp_path):
    from omicstra.protocols.encode import st_shards

    _, tbl = _tiny(tmp_path)
    g = StGraph(method="delaunay", radius_cap_px=None, scale_to_microns=1.2195)
    sh = st_shards([("s1", "c.h5ad", "s.parquet")], tmp_path, g, _axis(tbl))[0]
    assert sh.params["graph"]["scale_to_microns"] == 1.2195
    assert sh.params["axis"]["duplicates"] == "sum"
    assert sh.params["encoder"] == "novae"


def test_a_body_given_a_bare_shard_refuses_rather_than_defaulting(tmp_path):
    from omicstra.dispatch import Shard
    from omicstra.protocols.encode import st_shard_body

    with pytest.raises(KeyError, match="carries no graph"):
        st_shard_body(Shard(id="s1", output=tmp_path / "s1.npy",
                            params={"counts": "c", "coords": "s"}), lambda *_: None)


# --- the seed cohort: the acceptance --------------------------------------
FIVE = ["TNBC6_CN3_E1", "TNBC42_CN21_D2", "TNBC36_CN18_D2",
        "TNBC82_CN41_E2", "TNBC69_CN35_C1"]
CANON = ROOT / "projects" / "tnbc-92" / "data" / "canonical"
NOVAE_CACHE = ROOT / "data" / "embeddings" / "novae_niche_full"


@pytest.mark.parametrize("sid", FIVE)
def test_the_ported_st_encoder_reproduces_the_cache_exactly(sid):
    """the step 3 acceptance, and it is EXACT rather than a tolerance.

    the same five subarrays the 2026-09-15 edge-scale measurement used, at the
    scale that measurement recorded the cache was built with. the route here is
    the canonical adapter - .h5ad plus a coordinates parquet - where the cache
    came through Rscript and selection.RData, so this is the four-way split being
    tested and not only the maths.

    non-zero is a port defect, not device noise: novae runs on cpu float32 with
    no image decode in the loop, so there is nothing here to be 3e-06 about.
    """
    pytest.importorskip("novae", reason="the encode extra is not installed")
    import anndata as ad
    import pandas as pd

    h5, sp = CANON / f"{sid}.h5ad", CANON / f"{sid}_spots.parquet"
    ref_p = NOVAE_CACHE / f"{sid}_novae_embeddings.parquet"
    if not (h5.is_file() and sp.is_file() and ref_p.is_file()):
        pytest.skip("no canonical counts or novae cache on this machine")

    root = ROOT / "projects" / "tnbc-92"
    cohort = json.loads((root / "cohort.json").read_text())
    axis = GeneAxis.resolve(cohort, want="symbol",
                            observed=observed_gene_id(root, sid), root=root)
    graph = StGraph.from_platform(
        json.loads((root / "platform.json").read_text()), "original_st")

    a = ad.read_h5ad(h5)
    xy = pd.read_parquet(sp).set_index("spot_id").loc[
        a.obs_names.astype(str), ["x", "y"]].to_numpy()
    vecs, rec = run_one_st(a, xy, graph, axis, encoder="novae")

    cache = pd.read_parquet(ref_p)
    ids = list(cache.index.astype(str))
    got = pd.DataFrame(vecs, index=a.obs_names.astype(str)).loc[ids].to_numpy()
    ref = cache.loc[:, [f"novae_{i}" for i in range(64)]].to_numpy().astype(np.float32)

    assert np.array_equal(got, ref), (
        f"{sid}: max_abs {float(np.abs(got - ref).max()):.3e}. the acceptance is "
        "exact - report the cause rather than widening this.")
    assert rec.params["scale_to_microns"] == graph.scale_to_microns
