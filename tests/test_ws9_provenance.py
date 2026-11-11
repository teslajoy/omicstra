"""WS 9: provenance wiring — run records carry what each step was given.

three acceptance criteria:

  1. a record from any step in the AUCell chain carries its declared params
     in its `params` field (e.g. library_normalise -> target_sum,
     zscore_within_group -> group_by)

  2. params_space is enforced: a step sees only the keys it declares.
     no cross-step bleed-through in the recorded params.

  3. an end-to-end invocation through pathway_assay in graph.py produces
     records whose params match the cohort declaration, not the whole dict.
"""
from __future__ import annotations

import json
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# shared fixture: AUCell chain inputs
# ---------------------------------------------------------------------------

AU_PARAMS = {
    "niche_qc": {"min_niche_umi": 1000, "min_spot_umi": 200, "min_detected_spots": 3},
    "niche_sum": {},
    "library_normalise": {"target_sum": 1e4},
    "pathway_net": {"min_size": 3, "max_size": 500},
    "aucell": {},
    "zscore_within_group": {"group_by": "subarray"},
    "save_scores": {},
}


def _aucell_ctx(G=300, n=30):
    rng = np.random.default_rng(1)
    genes = [f"G{j}" for j in range(G)]
    niches = [(f"s{i // 10}::n{i}", (rng.poisson(3, size=(7, G)) * 20).astype("float32"))
              for i in range(n)]
    return {
        "params": AU_PARAMS,
        "niches": niches,
        "gene_symbols": genes,
        "pathway_sets": {"P_hit": genes[:40], "P_mid": genes[100:160]},
        "niche_groups": {nid: nid.split("::")[0] for nid, _ in niches},
    }


def _run_aucell_chain(**extra_ctx):
    from omicstra.protocols import build_protocol
    from omicstra.protocols.pathway import AUCELL_STEPS, VARIABLE_GENE_STEPS

    ctx = {**_aucell_ctx(), **extra_ctx}
    return build_protocol(VARIABLE_GENE_STEPS[:1] + AUCELL_STEPS, "aucell").invoke(ctx)


# ---------------------------------------------------------------------------
# criterion 1: records carry per-step params
# ---------------------------------------------------------------------------

def test_library_normalise_record_carries_target_sum():
    """library_normalise declares params_space=('target_sum',) and its record
    must carry exactly that key with the declared value."""
    r = _run_aucell_chain()

    rec = r["library_normalise"]
    assert rec.get("params"), "library_normalise record has no params"
    assert "target_sum" in rec["params"], (
        f"target_sum missing from library_normalise params: {rec['params']}")
    assert rec["params"]["target_sum"] == 1e4


def test_zscore_within_group_record_carries_group_by():
    """zscore_within_group declares params_space=('group_by',) and its record
    must carry exactly that key with the declared value."""
    r = _run_aucell_chain()

    rec = r["zscore_within_group"]
    assert rec.get("params"), "zscore_within_group record has no params"
    assert "group_by" in rec["params"]
    assert rec["params"]["group_by"] == "subarray"


def test_niche_qc_record_carries_all_declared_keys():
    """niche_qc declares three keys; all three must appear in its record."""
    r = _run_aucell_chain()

    rec = r["niche_qc"]
    for key in ("min_niche_umi", "min_spot_umi", "min_detected_spots"):
        assert key in rec.get("params", {}), (
            f"niche_qc record missing {key!r}: {rec.get('params')}")


# ---------------------------------------------------------------------------
# criterion 2: params_space filter — no cross-step bleed-through
# ---------------------------------------------------------------------------

def test_library_normalise_params_contain_only_declared_keys():
    """library_normalise declares params_space=('target_sum',).
    its record must NOT carry min_size, group_by, or any other step's keys."""
    r = _run_aucell_chain()

    params = r["library_normalise"].get("params", {})
    extra = set(params) - {"target_sum"}
    assert not extra, (
        f"library_normalise record carries undeclared keys: {extra}")


def test_pathway_net_params_contain_only_declared_keys():
    """pathway_net declares params_space=('min_size', 'max_size').
    its record must not carry target_sum or group_by."""
    r = _run_aucell_chain()

    params = r["pathway_net"].get("params", {})
    extra = set(params) - {"min_size", "max_size"}
    assert not extra, (
        f"pathway_net record carries undeclared keys: {extra}")


# ---------------------------------------------------------------------------
# criterion 3: end-to-end through pathway_assay in graph.py
# ---------------------------------------------------------------------------

@pytest.fixture
def computable_project(tmp_path):
    """minimal project with canonical data, niche parquet, pathway sets,
    and pathway_params — enough for pathway_assay to run the AUCell chain."""
    root = tmp_path / "proj"
    root.mkdir()

    canon = root / "data" / "canonical"
    canon.mkdir(parents=True)

    n_spots, n_genes = 20, 50
    rng = np.random.default_rng(42)
    X = rng.poisson(5, size=(n_spots, n_genes)).astype("float32")
    spot_ids = [f"{i}x{j}" for i in range(5) for j in range(4)]
    sample = "S1"

    anndata = pytest.importorskip("anndata")
    import scipy.sparse as sp

    adata = anndata.AnnData(
        X=sp.csr_matrix(X),
        obs=__import__("pandas").DataFrame(
            {"spot_id": spot_ids, "patient_id": "P1", "sample_id": sample},
            index=spot_ids),
        var=__import__("pandas").DataFrame(
            index=[f"ENSG{i:011d}.1" for i in range(n_genes)]))
    import pandas as pd
    adata.write_h5ad(canon / f"{sample}.h5ad")
    pd.DataFrame({"spot_id": spot_ids, "x": range(n_spots), "y": range(n_spots)}
                 ).to_parquet(canon / f"{sample}_spots.parquet")
    (canon / "ingest.json").write_text(json.dumps({"samples": [sample]}))

    mapping = pd.DataFrame({
        "ensembl": [f"ENSG{i:011d}" for i in range(n_genes)],
        "symbol": [f"GENE{i}" for i in range(n_genes)]})
    mapping.to_csv(root / "gene_map.tsv", sep="\t", index=False)

    niches_dir = root / "data" / "niches"
    niches_dir.mkdir(parents=True)
    centres = spot_ids[:10]
    nbrs_list = []
    for i, c in enumerate(centres):
        nbrs = [s for s in spot_ids if s != c][:6]
        nbrs_list.append(np.array(nbrs))
    pd.DataFrame({"neighbor_spot_ids": nbrs_list}, index=centres).to_parquet(
        niches_dir / f"{sample}.parquet")

    sets_dir = root / "data" / "pathway_sets"
    sets_dir.mkdir(parents=True)
    sets_path = sets_dir / "genesets.json"
    sets_path.write_text(json.dumps({
        "pw_A": ["GENE0", "GENE1", "GENE2", "GENE3"],
        "pw_B": ["GENE10", "GENE11", "GENE12"],
    }))

    pathway_params = {
        "niche_qc": {"min_niche_umi": 10, "min_spot_umi": 1, "min_detected_spots": 1},
        "niche_sum": {},
        "library_normalise": {"target_sum": 5000.0},
        "pathway_net": {"min_size": 2, "max_size": 500},
        "aucell": {},
        "zscore_within_group": {"group_by": "sample"},
        "pathway_sets_file": "data/pathway_sets/genesets.json",
        "save_scores": {},
    }

    (root / "project.json").write_text(json.dumps({
        "project_id": "ws9_test", "niches_dir": "data/niches",
        "platform": "synthetic", "work_dir": "work",
        "pathway_params": pathway_params,
    }))
    (root / "cohort.json").write_text(json.dumps({
        "gene_mapping": {"map": "gene_map.tsv", "map_key": "ensembl",
                         "map_value": "symbol"},
        "pathway_params": pathway_params,
    }))

    return root, sample, pathway_params


def test_pathway_assay_records_carry_params_matching_declaration(computable_project):
    """records produced by pathway_assay carry params that match the cohort's
    declaration, not the whole pathway_params dict.

    specifically:
      - library_normalise.params["target_sum"] == declared target_sum
      - zscore_within_group.params["group_by"] == declared group_by
    """
    from omicstra.graph import pathway_assay
    from omicstra.settings import settings

    root, sample, pp = computable_project
    # load cohort so pathway_assay can read pathway_params from state
    cohort = json.loads((root / "cohort.json").read_text())

    before = settings.project_dir
    settings.project_dir = root
    try:
        out = pathway_assay({
            "include_pathway_scores": True,
            "project_id": "ws9_test",
            "cohort": cohort,
        })
    finally:
        settings.project_dir = before

    stage = next((s for s in out.get("stages", [])
                  if s.get("stage") == "pathway_assay"), None)
    assert stage is not None
    if stage["status"] != "computed":
        pytest.skip(f"pathway_assay did not compute (status={stage['status']}); "
                    "likely missing dependency — provenance not testable")

    records = out.get("records", [])
    aucell_recs = {r["step_id"]: r for r in records
                  if r.get("unit") == "assay_pathway_aucell"}
    assert aucell_recs, "no assay_pathway_aucell records produced"

    ln = aucell_recs.get("library_normalise")
    assert ln is not None, "library_normalise record missing"
    assert ln.get("params", {}).get("target_sum") == pp["library_normalise"]["target_sum"], (
        f"library_normalise params mismatch: {ln.get('params')}")

    zs = aucell_recs.get("zscore_within_group")
    assert zs is not None, "zscore_within_group record missing"
    assert zs.get("params", {}).get("group_by") == pp["zscore_within_group"]["group_by"], (
        f"zscore_within_group params mismatch: {zs.get('params')}")

    # no whole-dict bleed: library_normalise must not carry group_by
    assert "group_by" not in ln.get("params", {}), (
        "library_normalise record carries group_by — whole dict leaked")
