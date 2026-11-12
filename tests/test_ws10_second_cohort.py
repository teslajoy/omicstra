"""WS 10: second cohort generalizability demo.

three acceptance criteria:

  1. alternate cohort declaration loads without errors in both the alignment
     and evaluation adapters — using non-conventional column names, different
     platform (visium), different encoder dims (uni2 1536d).

  2. question.validate() accepts a request against the alternate cohort's
     declared task families.

  3. missing declarations surface as explicit, named refusals — not silent
     wrong answers that inherit the seed cohort's values.
"""
from __future__ import annotations

import json
import pytest


# ---------------------------------------------------------------------------
# shared fixture: alternate cohort with every name differing from tnbc-92
# ---------------------------------------------------------------------------

@pytest.fixture
def alt_cohort(tmp_path):
    """minimal visium cohort fixture with non-conventional naming.

    seed cohort (tnbc-92)       alternate (visium-alt)
    -------------------------   ------------------------
    platform:  original_st      visium
    encoder:   virchow2 1280d   uni2 1536d
    h2_label:  mc_megacluster   leiden_cluster
    headline:  z_he             z_he_visium
    supervision: mc_weights_niche  cell_type_weights
    sample_col: subarray        slide_id
    niche_col:  virchow2_niche  uni2_embeddings
    """
    root = tmp_path / "projects" / "visium-alt"
    root.mkdir(parents=True)

    (root / "project.json").write_text(json.dumps({
        "project_id": "visium-alt",
        "platform": "visium",
        "encoders": [{"name": "uni2", "dim": 1536, "role": "primary"}],
        "h2_label": "leiden_cluster",
        "headline_view": "z_he_visium",
        "supervision": "cell_type_weights",
        "niches_dir": "data/niches",
        "runs_dir": "runs",
        "seed": 42,
        "contrastive_runs": ["R1_v1"],
        "classical_runs": ["B1_v1"],
    }))

    (root / "source_mapping.yaml").write_text(
        "he_niche_column: uni2_embeddings\n"
        "he_tokens_column: uni2_tile_tokens\n"
        "sample_column: slide_id\n"
    )

    canon = root / "data" / "canonical"
    canon.mkdir(parents=True)
    (canon / "ingest.json").write_text(json.dumps({"samples": ["slide_001"]}))

    import pandas as pd
    pd.DataFrame({
        "spot_id": [f"bc_{i}" for i in range(10)],
        "x": range(10),
        "y": range(10),
    }).to_parquet(canon / "slide_001_spots.parquet")

    (root / "cohort.json").write_text(json.dumps({
        "evaluate_params": {
            "paired_identifiability": {},
            "held_out_comparison": {},
            "hypothesis_verdict": {},
        }
    }))

    return root, tmp_path / "projects"


# ---------------------------------------------------------------------------
# criterion 1: adapters load alternate declarations without errors
# ---------------------------------------------------------------------------

def test_canonical_loads_alternate_cohort(alt_cohort):
    """list_samples() reads ingest.json and returns alternate sample ids."""
    from omicstra.adapters.canonical import list_samples
    root, _ = alt_cohort
    samples = list_samples(root)
    assert len(samples) == 1
    assert samples[0].sample_id == "slide_001"
    assert samples[0].spots is not None and samples[0].spots.exists()


def test_source_mapping_loads_non_conventional_names(alt_cohort):
    """SourceMapping.from_project_dir() reads uni2/slide_id naming without fallback."""
    from omicstra.adapters.source_mapping import SourceMapping
    root, _ = alt_cohort
    sm = SourceMapping.from_project_dir(root)
    assert sm.he_niche_column == "uni2_embeddings"
    assert sm.he_tokens_column == "uni2_tile_tokens"
    assert sm.sample_column == "slide_id"


def test_evaluation_declared_loads_alternate_fields(alt_cohort):
    """declared() returns leiden_cluster / z_he_visium — no seed-cohort defaults."""
    from omicstra.adapters.evaluation import declared
    root, _ = alt_cohort
    d = declared(root)
    assert d["h2_label"] == "leiden_cluster"
    assert d["headline_view"] == "z_he_visium"
    assert d["supervision"] == "cell_type_weights"
    assert d["seed"] == 42


def test_project_config_loads_alternate_encoder(alt_cohort):
    """ProjectConfig.load() resolves uni2 1536d, platform=visium."""
    from omicstra.contracts.project import ProjectConfig
    root, projects_dir = alt_cohort
    cfg = ProjectConfig.load("visium-alt", projects_dir=projects_dir)
    assert cfg.platform == "visium"
    assert cfg.h2_label == "leiden_cluster"
    primary = next((e for e in cfg.encoders if e.role == "primary"), None)
    assert primary is not None
    assert primary.name == "uni2"
    assert primary.dim == 1536


# ---------------------------------------------------------------------------
# criterion 2: validate() accepts a request against the alternate cohort
# ---------------------------------------------------------------------------

def test_question_validate_accepts_alternate_cohort(alt_cohort):
    """validate() with alternate project_id succeeds for a declared task family."""
    from omicstra.question import validate
    from omicstra.settings import settings

    root, projects_dir = alt_cohort
    old_pdir = settings.project_dir
    old_pdirs = settings.projects_dir
    settings.project_dir = None
    settings.projects_dir = projects_dir
    try:
        req = validate({
            "question": "does cross-modal retrieval work on this cohort?",
            "task_id": "cross_modal_retrieval",
            "run": "R1_v1",
            "project_id": "visium-alt",
        })
    finally:
        settings.project_dir = old_pdir
        settings.projects_dir = old_pdirs

    assert req.project_id == "visium-alt"
    assert req.task_id == "cross_modal_retrieval"
    assert req.hypothesis == "H1"
    assert req.run == "R1_v1"


# ---------------------------------------------------------------------------
# criterion 3: missing declarations surface explicitly, not silently
# ---------------------------------------------------------------------------

def test_missing_h2_label_named_in_error(tmp_path):
    """declared() refuses and names h2_label — does not inherit mc_megacluster."""
    from omicstra.adapters.evaluation import declared, DeclarationMissing
    (tmp_path / "project.json").write_text(json.dumps({
        "project_id": "incomplete-cohort",
        "platform": "visium",
        "niches_dir": "data/niches",
        "runs_dir": "runs",
        "headline_view": "z_he",
        "supervision": "cell_type_weights",
        "seed": 42,
        # h2_label intentionally absent
    }))
    with pytest.raises(DeclarationMissing) as exc:
        declared(tmp_path)
    assert "h2_label" in str(exc.value)


def test_missing_source_mapping_field_named_in_error(tmp_path):
    """SourceMappingMissing names the absent field — does not silently default."""
    from omicstra.adapters.source_mapping import SourceMapping, SourceMappingMissing
    (tmp_path / "source_mapping.yaml").write_text(
        "he_niche_column: uni2_embeddings\n"
        "sample_column: slide_id\n"
        # he_tokens_column intentionally absent
    )
    with pytest.raises(SourceMappingMissing) as exc:
        SourceMapping.from_project_dir(tmp_path)
    assert "he_tokens_column" in str(exc.value)
