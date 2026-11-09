"""analytical bindings: declaration-driven encoder selection and pathway targets.

tests two concerns:
  1. graphs/encode.py _resolve_encoder — declared primary encoder, refusal when absent
  2. adapters/evaluation.py pathway_cosines — pathway_targets from project.json,
     refusal when absent but pathway_node_embeddings is present
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. encoder resolution
# ---------------------------------------------------------------------------

class TestResolveEncoder:
    def test_explicit_encoder_wins(self, monkeypatch):
        """state['encoder'] always wins over any declaration."""
        from omicstra.graphs.encode import _resolve_encoder
        assert _resolve_encoder({"encoder": "uni2"}) == "uni2"

    def test_primary_role_resolved_from_declaration(self, tmp_path, monkeypatch):
        """when state has no encoder, reads declared primary encoder."""
        from omicstra.graphs.encode import _resolve_encoder

        (tmp_path / "project.json").write_text(json.dumps({
            "encoders": [
                {"name": "novae", "dim": 64, "role": "st"},
                {"name": "uni2", "dim": 1536, "role": "primary"},
            ]
        }))
        import omicstra.settings as _s
        monkeypatch.setattr(_s.settings, "project_dir", tmp_path)
        assert _resolve_encoder({}) == "uni2"

    def test_morphology_role_resolved(self, tmp_path, monkeypatch):
        """role 'morphology' is equivalent to 'primary' for encoder resolution."""
        from omicstra.graphs.encode import _resolve_encoder

        (tmp_path / "project.json").write_text(json.dumps({
            "encoders": [{"name": "virchow2", "dim": 1280, "role": "morphology"}]
        }))
        import omicstra.settings as _s
        monkeypatch.setattr(_s.settings, "project_dir", tmp_path)
        assert _resolve_encoder({}) == "virchow2"

    def test_no_primary_encoder_refused(self, tmp_path, monkeypatch):
        """when no encoder with role primary/morphology is declared, ValueError."""
        from omicstra.graphs.encode import _resolve_encoder

        (tmp_path / "project.json").write_text(json.dumps({
            "encoders": [{"name": "novae", "dim": 64, "role": "st"}]
        }))
        import omicstra.settings as _s
        monkeypatch.setattr(_s.settings, "project_dir", tmp_path)
        with pytest.raises(ValueError, match="no encoder passed"):
            _resolve_encoder({})

    def test_alternate_cohort_different_primary(self, tmp_path, monkeypatch):
        """alternate cohort with uni2 as primary uses uni2, no code change."""
        from omicstra.graphs.encode import _resolve_encoder

        (tmp_path / "project.json").write_text(json.dumps({
            "project_id": "alt",
            "encoders": [
                {"name": "uni2", "dim": 1536, "role": "primary"},
                {"name": "novae", "dim": 64, "role": "st"},
            ]
        }))
        import omicstra.settings as _s
        monkeypatch.setattr(_s.settings, "project_dir", tmp_path)
        assert _resolve_encoder({"project_id": "alt"}) == "uni2"


# ---------------------------------------------------------------------------
# 2. pathway targets
# ---------------------------------------------------------------------------

def _write_project_with_targets(proj_dir: Path, targets: list | None) -> None:
    """write a minimal project.json; include pathway_targets when given."""
    data: dict = {"pathway_node_embeddings": "data/embeddings/gpath2vec.pkl"}
    if targets is not None:
        data["pathway_targets"] = targets
    (proj_dir / "project.json").write_text(json.dumps(data))


class TestPathwayTargets:
    def test_absent_pathway_node_embeddings_returns_none(self, tmp_path):
        """when pathway_node_embeddings is not declared, returns None (no analysis)."""
        from omicstra.adapters.evaluation import pathway_cosines

        (tmp_path / "project.json").write_text(json.dumps({}))
        result = pathway_cosines(tmp_path, None)
        assert result is None

    def test_missing_pathway_targets_refused(self, tmp_path):
        """pathway_node_embeddings present but pathway_targets absent -> ValueError."""
        from omicstra.adapters.evaluation import pathway_cosines

        (tmp_path / "project.json").write_text(json.dumps({
            "pathway_node_embeddings": "data/embeddings/gpath2vec.pkl"
        }))
        with pytest.raises(ValueError, match="pathway_targets"):
            pathway_cosines(tmp_path, None)

    def test_pathway_targets_declaration_read(self, tmp_path):
        """declared pathway_targets are parsed without error (file absent returns None)."""
        from omicstra.adapters.evaluation import pathway_cosines

        (tmp_path / "project.json").write_text(json.dumps({
            "project_id": "test-alt",
            "platform": "original_st",
            "pathway_node_embeddings": "data/embeddings/gpath2vec.pkl",
            "pathway_targets": [
                {"name": "Immune_System", "reactome_id": "R-HSA-168256"},
            ]
        }))
        # emb_path will not exist -> returns None after declaring targets
        result = pathway_cosines(tmp_path, None)
        assert result is None

    def test_seed_cohort_project_json_has_pathway_targets(self):
        """tnbc-92 project.json now declares the 5 pathway targets."""
        root = Path(__file__).resolve().parents[1] / "projects" / "tnbc-92"
        raw = json.loads((root / "project.json").read_text())
        targets = raw.get("pathway_targets")
        assert targets is not None, "pathway_targets missing from tnbc-92/project.json"
        assert len(targets) == 5
        names = {t["name"] for t in targets}
        assert "TGF-beta_Signaling" in names
        assert "Immune_System" in names
        ids = {t["reactome_id"] for t in targets}
        assert "R-HSA-168256" in ids  # Immune_System
        assert "R-HSA-170834" in ids  # TGF-beta
