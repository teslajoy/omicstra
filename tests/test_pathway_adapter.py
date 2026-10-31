"""the pathway context loader: canonical h5ad + niche parquet -> ctx.

tier 1: the adapter returns what the protocol steps need, for a synthetic
sample. that the generated notebook RUNS is tier 2.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def pathway_cohort(tmp_path):
    """a minimal cohort with canonical data and a niche parquet."""
    root = tmp_path / "cohort"
    canon = root / "data" / "canonical"
    canon.mkdir(parents=True)
    niches = root / "data" / "niches"
    niches.mkdir(parents=True)

    n_spots, n_genes = 20, 50
    rng = np.random.default_rng(42)
    X = rng.poisson(5, size=(n_spots, n_genes)).astype("float32")
    spot_ids = [f"{i}x{j}" for i in range(5) for j in range(4)]

    anndata = pytest.importorskip("anndata")
    import scipy.sparse as sp
    adata = anndata.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({"spot_id": spot_ids, "patient_id": "P1",
                           "sample_id": "S1"}, index=spot_ids),
        var=pd.DataFrame(index=[f"ENSG{i:011d}.1" for i in range(n_genes)]))
    sample = "S1"
    adata.write_h5ad(canon / f"{sample}.h5ad")
    (canon / "ingest.json").write_text(json.dumps({"samples": [sample]}))

    # gene mapping
    mapping = pd.DataFrame({
        "ensembl": [f"ENSG{i:011d}" for i in range(n_genes)],
        "symbol": [f"GENE{i}" for i in range(n_genes)]})
    mapping.to_csv(root / "gene_map.tsv", sep="\t", index=False)

    # niche parquet: 10 niches, each with 6 neighbours
    centres = spot_ids[:10]
    neighbours = []
    for i, c in enumerate(centres):
        nbrs = [spot_ids[j] for j in range(n_spots)
                if j != i and spot_ids[j] != c][:6]
        neighbours.append(np.array(nbrs))
    niche_df = pd.DataFrame({"neighbor_spot_ids": neighbours}, index=centres)
    niche_df.to_parquet(niches / f"{sample}.parquet")

    # project.json and cohort.json
    (root / "project.json").write_text(json.dumps({
        "project_id": "test", "niches_dir": "data/niches",
        "platform": "synthetic", "work_dir": "work"}))
    (root / "cohort.json").write_text(json.dumps({
        "gene_mapping": {"map": "gene_map.tsv", "map_key": "ensembl",
                          "map_value": "symbol"}}))
    return root, sample


def test_arrays_returns_all_required_context_keys(pathway_cohort):
    from omicstra.settings import settings

    root, sample = pathway_cohort
    before = settings.project_dir
    settings.project_dir = root
    try:
        from omicstra.adapters.pathway import arrays
        ctx = arrays(str(root), sample)
    finally:
        settings.project_dir = before

    assert "genes" in ctx
    assert "gene_symbols" in ctx
    assert "niche_members" in ctx
    assert "niches" in ctx
    assert "niche_groups" in ctx
    assert len(ctx["genes"]) == 50
    assert len(ctx["gene_symbols"]) == 50
    assert all(s.startswith("GENE") for s in ctx["gene_symbols"])


def test_niches_carry_counts_as_2d_arrays(pathway_cohort):
    from omicstra.settings import settings

    root, sample = pathway_cohort
    before = settings.project_dir
    settings.project_dir = root
    try:
        from omicstra.adapters.pathway import arrays
        ctx = arrays(str(root), sample)
    finally:
        settings.project_dir = before

    for nid, counts in ctx["niches"]:
        assert isinstance(counts, np.ndarray)
        assert counts.ndim == 2
        assert counts.shape[1] == 50
        assert counts.shape[0] >= 1


def test_niche_members_match_niches(pathway_cohort):
    from omicstra.settings import settings

    root, sample = pathway_cohort
    before = settings.project_dir
    settings.project_dir = root
    try:
        from omicstra.adapters.pathway import arrays
        ctx = arrays(str(root), sample)
    finally:
        settings.project_dir = before

    assert len(ctx["niche_members"]) == len(ctx["niches"])
    for nid, counts in ctx["niches"]:
        members = ctx["niche_members"][nid]
        assert counts.shape[0] == len(members)


def test_niche_groups_are_the_sample(pathway_cohort):
    from omicstra.settings import settings

    root, sample = pathway_cohort
    before = settings.project_dir
    settings.project_dir = root
    try:
        from omicstra.adapters.pathway import arrays
        ctx = arrays(str(root), sample)
    finally:
        settings.project_dir = before

    for nid, group in ctx["niche_groups"].items():
        assert group == sample


def test_missing_h5ad_is_refused(pathway_cohort):
    from omicstra.adapters.pathway import PathwayDataMissing
    from omicstra.settings import settings

    root, _ = pathway_cohort
    before = settings.project_dir
    settings.project_dir = root
    try:
        from omicstra.adapters.pathway import arrays
        with pytest.raises(PathwayDataMissing, match="no canonical h5ad"):
            arrays(str(root), "NONEXISTENT")
    finally:
        settings.project_dir = before


def test_missing_niche_parquet_is_refused(pathway_cohort):
    import os
    from omicstra.adapters.pathway import PathwayDataMissing
    from omicstra.settings import settings

    root, sample = pathway_cohort
    niche_path = root / "data" / "niches" / f"{sample}.parquet"
    os.remove(niche_path)
    before = settings.project_dir
    settings.project_dir = root
    try:
        from omicstra.adapters.pathway import arrays
        with pytest.raises(PathwayDataMissing, match="no niche parquet"):
            arrays(str(root), sample)
    finally:
        settings.project_dir = before


def test_pathway_units_are_generable():
    """all three pathway units now have context loaders."""
    from omicstra.notebook import UNITS
    for name in ("assay_pathway_genesets", "assay_pathway_aucell",
                 "assay_niche_membership"):
        assert UNITS[name].context is not None, f"{name} has no context loader"


def test_pathway_unit_notebooks_build_with_empty_records():
    """the notebook generator can produce the structure (header + footer) for
    each pathway unit, even with no records."""
    from omicstra.notebook import UNITS, build

    for name in ("assay_pathway_genesets", "assay_pathway_aucell",
                 "assay_niche_membership"):
        nb = build({"unit": name, "project_id": "c", "project_dir": "/tmp/c",
                     "run": None, "run_id": "test",
                     "params": {}, "records": [], "sample": "S1"})
        assert nb["metadata"]["omicstra"]["unit"] == name
        setup = nb["cells"][1]["source"]
        assert "from omicstra.adapters.pathway import arrays" in setup
        assert "ctx = arrays(PROJECT," in setup


# --- integration: assay node -> records -> notebook -------------------------


def test_aucell_chain_builds_with_shared_qc_step():
    """the aucell chain cannot build standalone because niche_sum requires
    qc_niches. composing niche_qc into the chain resolves this."""
    from omicstra.protocols import build_protocol
    from omicstra.protocols.pathway import AUCELL_STEPS, VARIABLE_GENE_STEPS

    qc = VARIABLE_GENE_STEPS[0]
    assert qc.id == "niche_qc"
    chain = build_protocol([qc, *AUCELL_STEPS], "aucell_with_qc")
    assert chain is not None


def test_assay_skips_when_no_pathway_flags_set():
    """the assay node is a no-op when neither include_pathway_scores nor
    include_pathway_cosines is set."""
    from omicstra.graph import pathway_assay as assay

    state = {"task_id": "cross_modal_retrieval"}
    out = assay(state)
    assert out["stages"][0]["status"] == "nothing_to_run"
    assert "records" not in out


def test_assay_reuses_declared_pathway_scores(pathway_cohort):
    """when include_pathway_scores is set and the declared file exists,
    the assay resolves without computation."""
    from omicstra.graph import pathway_assay as assay
    from omicstra.settings import settings

    root, sample = pathway_cohort
    # create a declared pathway_scores artifact on disk
    scores_path = root / "data" / "scores" / "aucell.parquet"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame({"pathway_A": [0.1, 0.2]}).to_parquet(scores_path)

    # update project.json with the declaration
    import json
    proj = json.loads((root / "project.json").read_text())
    proj["pathway_scores"] = {"path": "data/scores/aucell.parquet",
                               "columns": ["pathway_A"]}
    (root / "project.json").write_text(json.dumps(proj))

    before = settings.project_dir
    settings.project_dir = root
    try:
        state = {"include_pathway_scores": True, "project_id": "test"}
        out = assay(state)
    finally:
        settings.project_dir = before

    assert out["stages"][0]["status"] == "resolved"
    assert "pathway_scores" in out["stages"][0]["report"]["reused"]


def test_assay_refuses_missing_cosines_with_no_producer(pathway_cohort):
    """when include_pathway_cosines is set but the declared artifact is absent,
    the assay refuses because no chain produces gpath2vec embeddings."""
    from omicstra.graph import pathway_assay as assay
    from omicstra.settings import settings

    root, _ = pathway_cohort
    before = settings.project_dir
    settings.project_dir = root
    try:
        state = {"include_pathway_cosines": True, "project_id": "test"}
        out = assay(state)
    finally:
        settings.project_dir = before

    assert out["stages"][0]["status"] == "refused"
    assert "pathway_node_embeddings" in str(out["stages"][0]["report"])


def test_assay_refuses_missing_scores_without_params(pathway_cohort):
    """when include_pathway_scores is set, the artifact is absent, and no
    pathway_params are declared, the assay refuses explicitly."""
    from omicstra.graph import pathway_assay as assay
    from omicstra.settings import settings

    root, _ = pathway_cohort
    before = settings.project_dir
    settings.project_dir = root
    try:
        state = {"include_pathway_scores": True,
                 "project_id": "test", "cohort": {}}
        out = assay(state)
    finally:
        settings.project_dir = before

    assert out["stages"][0]["status"] == "refused"
    assert "pathway_params" in out["stages"][0]["report"]["why"]


def test_assay_reuse_generates_notebook_from_resolved_stage(pathway_cohort):
    """the resolved stage report (artifact reused) carries enough information
    for the notebook generator to produce provenance."""
    from omicstra.graph import pathway_assay as assay
    from omicstra.settings import settings

    root, sample = pathway_cohort
    # create a declared pathway_scores artifact
    scores_path = root / "data" / "scores" / "aucell.parquet"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame({"pathway_A": [0.1, 0.2]}).to_parquet(scores_path)

    import json
    proj = json.loads((root / "project.json").read_text())
    proj["pathway_scores"] = {"path": "data/scores/aucell.parquet",
                               "columns": ["pathway_A"]}
    (root / "project.json").write_text(json.dumps(proj))

    before = settings.project_dir
    settings.project_dir = root
    try:
        state = {"include_pathway_scores": True, "project_id": "test"}
        out = assay(state)
    finally:
        settings.project_dir = before

    stage = out["stages"][0]
    assert stage["status"] == "resolved"
    assert stage["report"]["reused"]["pathway_scores"] == str(scores_path)
    assert "pathway_scores" in stage["report"]["required"]


def test_from_records_filters_by_sample(pathway_cohort):
    """records with different samples don't overwrite each other."""
    from omicstra.notebook import from_records
    from omicstra.settings import settings

    root, _ = pathway_cohort
    before = settings.project_dir
    settings.project_dir = root
    try:
        recs = [
            {"step_id": "niche_qc", "status": "pass", "sample": "S1",
             "result": "s1", "observed": {"n_out": 10}, "params": {}},
            {"step_id": "niche_qc", "status": "pass", "sample": "S2",
             "result": "s2", "observed": {"n_out": 20}, "params": {}},
        ]
        b1 = from_records("assay_pathway_genesets", recs, "test", sample="S1")
        b2 = from_records("assay_pathway_genesets", recs, "test", sample="S2")
    finally:
        settings.project_dir = before

    assert b1["records"][0]["result"] == "s1"
    assert b2["records"][0]["result"] == "s2"
