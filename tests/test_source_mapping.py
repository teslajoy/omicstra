"""source mapping: declared column-name bindings per cohort.

tests three concerns:
  1. SourceMapping.load_from_yaml - happy path and refusals
  2. AlignmentContract.from_project_dir with source_mapping.yaml - uses declared names
  3. AlignmentContract.from_project_dir without yaml - convention fallback unchanged
  4. alternate fixture (uni2/1536d, donor_id, slide_id) loads through the same path
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from omicstra.adapters.source_mapping import SourceMapping, SourceMappingMissing
from omicstra.adapters.alignment_data import AlignmentContract, AlignmentDeclarationMissing

FIXTURES = Path(__file__).parent / "fixtures" / "schema"


# ---------------------------------------------------------------------------
# 1. SourceMapping.load_from_yaml
# ---------------------------------------------------------------------------

class TestSourceMappingLoadFromYaml:
    def test_happy_path(self, tmp_path):
        p = tmp_path / "source_mapping.yaml"
        p.write_text(
            "he_niche_column: custom_niche\n"
            "he_tokens_column: custom_tokens\n"
            "sample_column: slide_id\n"
        )
        sm = SourceMapping.load_from_yaml(p)
        assert sm.he_niche_column == "custom_niche"
        assert sm.he_tokens_column == "custom_tokens"
        assert sm.sample_column == "slide_id"

    def test_missing_file(self, tmp_path):
        with pytest.raises(SourceMappingMissing, match="not found"):
            SourceMapping.load_from_yaml(tmp_path / "nonexistent.yaml")

    def test_missing_field_named(self, tmp_path):
        p = tmp_path / "source_mapping.yaml"
        p.write_text("he_niche_column: x\nhe_tokens_column: y\n")
        with pytest.raises(SourceMappingMissing, match="sample_column"):
            SourceMapping.load_from_yaml(p)

    def test_all_missing_fields_named(self, tmp_path):
        p = tmp_path / "source_mapping.yaml"
        p.write_text("{}\n")
        with pytest.raises(SourceMappingMissing) as exc_info:
            SourceMapping.load_from_yaml(p)
        msg = str(exc_info.value)
        assert "he_niche_column" in msg
        assert "he_tokens_column" in msg
        assert "sample_column" in msg

    def test_from_project_dir_reads_yaml(self, tmp_path):
        (tmp_path / "source_mapping.yaml").write_text(
            "he_niche_column: a\nhe_tokens_column: b\nsample_column: c\n"
        )
        sm = SourceMapping.from_project_dir(tmp_path)
        assert sm.he_niche_column == "a"

    def test_from_project_dir_missing_yaml_refused(self, tmp_path):
        with pytest.raises(SourceMappingMissing, match="not found"):
            SourceMapping.from_project_dir(tmp_path)


# ---------------------------------------------------------------------------
# 2. AlignmentContract.from_project_dir WITH source_mapping.yaml
# ---------------------------------------------------------------------------

def _write_base_declarations(proj: Path, encoder_name: str, dim: int,
                              subject_col: str = "patient_id") -> None:
    (proj / "project.json").write_text(json.dumps({
        "encoders": [{"name": encoder_name, "dim": dim, "role": "morphology"}],
        "k_neighbors": 6,
        "supervision": "mc_weights_niche",
    }))
    (proj / "cohort.json").write_text(json.dumps({
        "subject_id_column": subject_col,
    }))


class TestAlignmentContractWithSourceMapping:
    def test_yaml_overrides_convention(self, tmp_path):
        """when source_mapping.yaml is present, its names win over convention."""
        proj = tmp_path / "proj"
        proj.mkdir()
        _write_base_declarations(proj, "virchow2", 1280)
        (proj / "source_mapping.yaml").write_text(
            "he_niche_column: virchow2_embeddings\n"
            "he_tokens_column: virchow2_patch_tokens\n"
            "sample_column: section_id\n"
        )
        c = AlignmentContract.from_project_dir(proj)
        assert c.he_niche_column == "virchow2_embeddings"
        assert c.he_tokens_column == "virchow2_patch_tokens"
        assert c.sample_column == "section_id"
        assert c.he_dim == 1280

    def test_seed_cohort_yaml_produces_same_columns_as_convention(self, tmp_path):
        """tnbc-92 source_mapping.yaml declares the same names convention would derive."""
        proj = tmp_path / "proj"
        proj.mkdir()
        _write_base_declarations(proj, "virchow2", 1280)
        (proj / "source_mapping.yaml").write_text(
            "he_niche_column: virchow2_niche\n"
            "he_tokens_column: virchow2_cell_tokens\n"
            "sample_column: subarray\n"
        )
        c = AlignmentContract.from_project_dir(proj)
        assert c.he_niche_column == "virchow2_niche"
        assert c.he_tokens_column == "virchow2_cell_tokens"
        assert c.sample_column == "subarray"

    def test_alternate_fixture_yaml(self, tmp_path):
        """alternate cohort (uni2/1536d, donor_id, slide_id) loads through same function."""
        proj = tmp_path / "proj"
        proj.mkdir()
        _write_base_declarations(proj, "uni2", 1536, subject_col="donor_id")
        (proj / "source_mapping.yaml").write_text(
            (FIXTURES / "alternate_source_mapping.yaml").read_text()
        )
        c = AlignmentContract.from_project_dir(proj)
        assert c.he_niche_column == "uni2_embeddings"
        assert c.he_tokens_column == "uni2_tile_tokens"
        assert c.sample_column == "slide_id"
        assert c.he_dim == 1536
        assert c.subject_column == "donor_id"

    def test_yaml_missing_field_refused_with_field_name(self, tmp_path):
        """a yaml missing sample_column is refused with that field named."""
        proj = tmp_path / "proj"
        proj.mkdir()
        _write_base_declarations(proj, "virchow2", 1280)
        (proj / "source_mapping.yaml").write_text(
            "he_niche_column: x\nhe_tokens_column: y\n"
        )
        with pytest.raises(SourceMappingMissing, match="sample_column"):
            AlignmentContract.from_project_dir(proj)


# ---------------------------------------------------------------------------
# 3. AlignmentContract.from_project_dir WITHOUT yaml - convention unchanged
# ---------------------------------------------------------------------------

class TestAlignmentContractConventionFallback:
    def test_no_yaml_falls_back_to_convention(self, tmp_path):
        """without source_mapping.yaml the naming convention is still the fallback."""
        proj = tmp_path / "proj"
        proj.mkdir()
        _write_base_declarations(proj, "virchow2", 1280)
        c = AlignmentContract.from_project_dir(proj)
        assert c.he_niche_column == "virchow2_niche"
        assert c.he_tokens_column == "virchow2_cell_tokens"
        assert c.sample_column == "subarray"

    def test_no_yaml_alt_encoder_convention(self, tmp_path):
        proj = tmp_path / "proj"
        proj.mkdir()
        _write_base_declarations(proj, "uni2", 1536, subject_col="donor_id")
        c = AlignmentContract.from_project_dir(proj)
        assert c.he_niche_column == "uni2_niche"
        assert c.he_tokens_column == "uni2_cell_tokens"