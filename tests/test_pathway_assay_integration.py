"""integration: pathway_assay through the graph, blocking, and notebook.

proves:
- a resolved artifact lets the chain continue to align
- a refused artifact blocks align from running
- the resolved stage produces provenance the notebook generator consumes
- a COMPUTED artifact produces records, reaches downstream, and generates a notebook
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from omicstra.graph import OmicstraState, build_omicstra_graph


# --- stub subgraphs --------------------------------------------------------
# each returns a minimal stage report so the chain walks to pathway_assay.


def _stub_eda(state):
    return {"report": {"status": "ready"}, "records": []}


def _stub_encode(state):
    return {"report": {"status": "resolved"}, "records": []}


def _stub_st(state):
    return {"report": {"status": "resolved"}, "records": []}


_align_invocations: list[dict] = []


def _stub_align(state):
    _align_invocations.append(dict(state))
    return {"report": {"status": "resolved"}, "records": []}


def _stub_evaluate(state):
    return {"report": {"status": "evaluated"}, "records": []}


def _build_graph():
    return build_omicstra_graph(
        checkpointer=InMemorySaver(),
        eda=_stub_eda,
        encode=_stub_encode,
        st=_stub_st,
        align=_stub_align,
        evaluate=_stub_evaluate,
    )


# --- fixtures ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_align_log():
    _align_invocations.clear()


@pytest.fixture
def pathway_project(tmp_path):
    """a minimal project with project.json, cohort.json, routing evidence,
    and a routing contract that declares pathway_transfer."""
    root = tmp_path / "proj"
    root.mkdir()

    (root / "project.json").write_text(json.dumps({
        "project_id": "test", "work_dir": "work",
        "platform": "synthetic",
    }))
    (root / "cohort.json").write_text(json.dumps({}))

    # routing evidence with a pathway_transfer entry so discover picks compute
    (root / "routing_evidence.json").write_text(json.dumps({"tasks": {}}))

    # routing contract declaring pathway_transfer
    import importlib.resources
    rc_path = importlib.resources.files("omicstra.configs") / "routing_contract.json"
    rc = json.loads(rc_path.read_text())
    (root / "routing_contract.json").write_text(json.dumps(rc))

    return root


@pytest.fixture
def pathway_project_with_scores(pathway_project):
    """adds a declared pathway_scores artifact on disk."""
    root = pathway_project
    scores_dir = root / "data" / "scores"
    scores_dir.mkdir(parents=True)
    scores_path = scores_dir / "aucell.parquet"
    pd.DataFrame({"pathway_A": [0.1, 0.2]}).to_parquet(scores_path)

    proj = json.loads((root / "project.json").read_text())
    proj["pathway_scores"] = {"path": "data/scores/aucell.parquet",
                               "columns": ["pathway_A"]}
    (root / "project.json").write_text(json.dumps(proj))
    return root


# --- acceptance tests -------------------------------------------------------

def test_resolved_artifact_lets_chain_reach_align(pathway_project_with_scores):
    """when include_pathway_scores is set and the artifact exists on disk,
    pathway_assay resolves and the chain continues to align."""
    from omicstra.settings import settings

    root = pathway_project_with_scores
    before = settings.project_dir
    settings.project_dir = root
    try:
        g = _build_graph()
        result = g.invoke(
            {"task_id": "pathway_transfer",
             "project_id": "test",
             "include_pathway_scores": True},
            config={"configurable": {"thread_id": "t1"}},
        )
    finally:
        settings.project_dir = before

    # pathway_assay resolved (the node's own stage + the keeper's copy)
    pa_stages = [s for s in result.get("stages", [])
                 if s.get("stage") == "pathway_assay"]
    assert len(pa_stages) >= 1
    node_stage = pa_stages[0]
    assert node_stage["status"] == "resolved"
    assert "pathway_scores" in node_stage["report"]["reused"]

    # align was reached (stub was called)
    assert len(_align_invocations) == 1, "align was not reached after resolved artifact"


def test_missing_unproducible_artifact_blocks_align(pathway_project):
    """when include_pathway_cosines is set and the artifact is absent with no
    producer, pathway_assay refuses and the chain stops before align."""
    from omicstra.settings import settings

    root = pathway_project
    before = settings.project_dir
    settings.project_dir = root
    try:
        g = _build_graph()
        result = g.invoke(
            {"task_id": "pathway_transfer",
             "project_id": "test",
             "include_pathway_cosines": True},
            config={"configurable": {"thread_id": "t2"}},
        )
    finally:
        settings.project_dir = before

    # pathway_assay refused
    pa_stages = [s for s in result.get("stages", [])
                 if s.get("stage") == "pathway_assay"]
    assert len(pa_stages) >= 1
    assert pa_stages[0]["status"] == "refused"

    # align was NOT reached - the refusal blocked the chain
    assert len(_align_invocations) == 0, \
        "align ran despite pathway_assay refusing - the chain did not block"


def test_no_pathway_flags_lets_chain_pass_through(pathway_project):
    """when neither pathway flag is set, pathway_assay is nothing_to_run
    and the chain continues to align."""
    from omicstra.settings import settings

    root = pathway_project
    before = settings.project_dir
    settings.project_dir = root
    try:
        g = _build_graph()
        result = g.invoke(
            {"task_id": "cross_modal_retrieval",
             "project_id": "test"},
            config={"configurable": {"thread_id": "t3"}},
        )
    finally:
        settings.project_dir = before

    pa_stages = [s for s in result.get("stages", [])
                 if s.get("stage") == "pathway_assay"]
    assert len(pa_stages) >= 1
    assert pa_stages[0]["status"] == "nothing_to_run"

    assert len(_align_invocations) == 1, "align was not reached for a non-pathway task"


def test_resolved_stage_generates_notebook_provenance(pathway_project_with_scores):
    """the resolved pathway_assay stage carries enough provenance for the
    notebook generator to produce a notebook with the reused artifact."""
    from omicstra.notebook import build
    from omicstra.settings import settings

    root = pathway_project_with_scores
    before = settings.project_dir
    settings.project_dir = root
    try:
        g = _build_graph()
        result = g.invoke(
            {"task_id": "pathway_transfer",
             "project_id": "test",
             "include_pathway_scores": True},
            config={"configurable": {"thread_id": "t4"}},
        )

        pa_stages = [s for s in result.get("stages", [])
                     if s.get("stage") == "pathway_assay"]
        assert len(pa_stages) >= 1

        # the stage report carries provenance: which artifacts were reused
        report = pa_stages[0]["report"]
        assert "pathway_scores" in report["reused"]
        reused_path = report["reused"]["pathway_scores"]

        # build a notebook from the stage info
        nb = build({
            "unit": "assay_pathway_aucell",
            "project_id": "test",
            "project_dir": str(root),
            "run": None,
            "run_id": "test-provenance",
            "params": {},
            "records": [],
            "sample": None,
        })
    finally:
        settings.project_dir = before

    assert nb["metadata"]["omicstra"]["unit"] == "assay_pathway_aucell"
    # the notebook structure exists (header + setup + footer cells)
    assert len(nb["cells"]) >= 3
    # the setup cell references the pathway context loader
    setup = nb["cells"][1]["source"]
    assert "from omicstra.adapters.pathway import arrays" in setup


# --- acceptance: computation -> downstream -> notebook ----------------------
# the fixture below provides everything the AUCell chain needs to actually run
# on synthetic data: a canonical h5ad, a niche parquet, a gene mapping, pathway
# gene sets, and pathway_params with entries for every step in the chain.


@pytest.fixture
def pathway_project_computable(tmp_path):
    """a project where pathway_scores is NOT on disk, but all inputs and params
    are present so the AUCell chain can compute it from scratch."""
    root = tmp_path / "proj_compute"
    root.mkdir()

    # --- canonical data: a tiny h5ad with 20 spots x 50 genes ---------------
    canon = root / "data" / "canonical"
    canon.mkdir(parents=True)

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
        var=pd.DataFrame(
            index=[f"ENSG{i:011d}.1" for i in range(n_genes)]))
    sample = "S1"
    adata.write_h5ad(canon / f"{sample}.h5ad")
    pd.DataFrame({"spot_id": spot_ids,
                   "x": range(n_spots), "y": range(n_spots)}
                 ).to_parquet(canon / f"{sample}_spots.parquet")
    (canon / "ingest.json").write_text(json.dumps({"samples": [sample]}))

    # --- gene mapping: ensembl -> symbol ------------------------------------
    mapping = pd.DataFrame({
        "ensembl": [f"ENSG{i:011d}" for i in range(n_genes)],
        "symbol": [f"GENE{i}" for i in range(n_genes)]})
    mapping.to_csv(root / "gene_map.tsv", sep="\t", index=False)

    # --- niche parquet: 10 niches, each with 6 neighbours -------------------
    niches_dir = root / "data" / "niches"
    niches_dir.mkdir(parents=True)
    centres = spot_ids[:10]
    neighbours = []
    for i, c in enumerate(centres):
        nbrs = [spot_ids[j] for j in range(n_spots)
                if j != i and spot_ids[j] != c][:6]
        neighbours.append(np.array(nbrs))
    niche_df = pd.DataFrame({"neighbor_spot_ids": neighbours}, index=centres)
    niche_df.to_parquet(niches_dir / f"{sample}.parquet")

    # --- pathway gene sets: small sets referencing the fixture's symbols -----
    pathway_sets = {
        "test_pathway_A": ["GENE0", "GENE1", "GENE2", "GENE3", "GENE4"],
        "test_pathway_B": ["GENE10", "GENE11", "GENE12", "GENE13"],
        "test_pathway_C": ["GENE20", "GENE21", "GENE22"],
    }
    sets_dir = root / "data" / "pathway_sets"
    sets_dir.mkdir(parents=True)
    sets_path = sets_dir / "genesets.json"
    sets_path.write_text(json.dumps(pathway_sets))

    # --- pathway_params for every step in the chain -------------------------
    pathway_params = {
        "niche_qc": {"min_niche_umi": 10, "min_spot_umi": 1,
                     "min_detected_spots": 1},
        "niche_sum": {},
        "library_normalise": {"target_sum": 1e4},
        "pathway_net": {"min_size": 2, "max_size": 500},
        "aucell": {},
        "zscore_within_group": {"group_by": "sample"},
    }

    # --- project.json: no pathway_scores declared ---------------------------
    (root / "project.json").write_text(json.dumps({
        "project_id": "test_compute", "work_dir": "work",
        "platform": "synthetic",
        "niches_dir": "data/niches",
        "pathway_params": {**pathway_params,
                           "pathway_sets_file": "data/pathway_sets/genesets.json"},
    }))
    (root / "cohort.json").write_text(json.dumps({
        "gene_mapping": {"map": "gene_map.tsv", "map_key": "ensembl",
                          "map_value": "symbol"},
        "pathway_params": {**pathway_params,
                           "pathway_sets_file": "data/pathway_sets/genesets.json"},
    }))

    # routing evidence: pathway_transfer not yet run
    (root / "routing_evidence.json").write_text(json.dumps({"tasks": {}}))

    # routing contract
    import importlib.resources
    rc_path = importlib.resources.files("omicstra.configs") / "routing_contract.json"
    rc = json.loads(rc_path.read_text())
    (root / "routing_contract.json").write_text(json.dumps(rc))

    return root, sample


def test_computed_artifact_produces_records_reaches_downstream_generates_notebook(
        pathway_project_computable):
    """the acceptance case: pathway_assay computes AUCell scores on synthetic
    data, produces records with sample identity, the records reach the downstream
    align stage, and a notebook with provenance is generated from them.

    this is the end-to-end chain that the reuse and blocking tests do not cover:
    computation -> saved records -> downstream consumption -> notebook.
    """
    from omicstra.notebook import build, from_records
    from omicstra.settings import settings

    root, sample = pathway_project_computable
    before = settings.project_dir
    settings.project_dir = root
    try:
        g = _build_graph()
        result = g.invoke(
            {"task_id": "pathway_transfer",
             "project_id": "test_compute",
             "include_pathway_scores": True},
            config={"configurable": {"thread_id": "t_compute"}},
        )

        # --- 1. pathway_assay computed (not resolved, not refused) -----------
        pa_stages = [s for s in result.get("stages", [])
                     if s.get("stage") == "pathway_assay"]
        assert len(pa_stages) >= 1
        stage = pa_stages[0]
        assert stage["status"] == "computed", (
            f"expected computed, got {stage['status']}: {stage.get('report')}")
        assert sample in stage["report"]["samples_run"]

        # --- 2. records carry sample identity and step provenance ------------
        records = result.get("records", [])
        pathway_records = [r for r in records
                           if r.get("unit") == "assay_pathway_aucell"]
        assert len(pathway_records) >= 1, "no AUCell records produced"
        for rec in pathway_records:
            assert rec["sample"] == sample, (
                f"record missing sample identity: {rec.get('step_id')}")

        step_ids = {r["step_id"] for r in pathway_records}
        expected_steps = {"niche_qc", "niche_sum", "library_normalise",
                          "pathway_net", "aucell", "zscore_within_group"}
        assert expected_steps <= step_ids, (
            f"missing steps: {expected_steps - step_ids}")

        for rec in pathway_records:
            assert rec["status"] in ("pass", "not_applicable"), (
                f"step {rec['step_id']} status={rec['status']}: "
                f"{rec.get('result')}")

        aucell_rec = next(r for r in pathway_records
                         if r["step_id"] == "aucell")
        assert aucell_rec["observed"]["n_pathways"] >= 1
        assert aucell_rec["observed"]["n_out"] >= 1

        # --- 3. downstream align received the computed records ---------------
        assert len(_align_invocations) == 1, (
            "align was not reached after computed pathway_assay")
        align_state = _align_invocations[0]
        align_records = align_state.get("records", [])
        align_pw = [r for r in align_records
                    if r.get("unit") == "assay_pathway_aucell"]
        assert len(align_pw) == len(pathway_records), (
            f"align received {len(align_pw)} AUCell records, "
            f"expected {len(pathway_records)}")

        # --- 4. notebook generation from computed records --------------------
        bundle = from_records("assay_pathway_aucell", records,
                              "test_compute", sample=sample,
                              run_id="acceptance-compute")
        assert len(bundle["records"]) >= 1, \
            "from_records filtered out all records"

        nb = build(bundle)
        assert nb["metadata"]["omicstra"]["unit"] == "assay_pathway_aucell"
        assert nb["metadata"]["omicstra"]["produced_by_run"] == \
            "acceptance-compute"

        assert len(nb["cells"]) >= 3
        setup = nb["cells"][1]["source"]
        assert "from omicstra.adapters.pathway import arrays" in setup
        assert f"'{sample}'" in setup, \
            "setup cell does not reference the sample"

        assert "PARAMS" in setup
        params_in_nb = bundle["params"]
        has_real_params = any(bool(v) for v in params_in_nb.values())
        assert has_real_params, (
            "notebook params are all empty - records did not carry "
            "their parameters")
    finally:
        settings.project_dir = before
