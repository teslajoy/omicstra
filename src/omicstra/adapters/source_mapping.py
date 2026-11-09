"""declared bindings from schema entities to source column names.

a source mapping is a yaml file, one per cohort, that makes the mapping
from schema roles to actual column names explicit. without it, the alignment
contract falls back to naming conventions ({encoder_name}_niche, etc.);
with it, any column naming scheme works without code changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class SourceMappingMissing(RuntimeError):
    """a required field in source_mapping.yaml is absent or invalid."""


REQUIRED_FIELDS = ("he_niche_column", "he_tokens_column", "sample_column")


@dataclass(frozen=True)
class SourceMapping:
    """declared column-name bindings for one cohort.

    maps schema roles to the actual column names present in the cohort's
    niche-join parquets. three fields cover the naming conventions that
    AlignmentContract.from_project_dir() previously derived by convention:

      he_niche_column   - pooled H&E niche embedding  (was: {encoder}_niche)
      he_tokens_column  - per-cell H&E token block    (was: {encoder}_cell_tokens)
      sample_column     - sample/subarray identifier  (was: hardcoded "subarray")
    """
    he_niche_column: str
    he_tokens_column: str
    sample_column: str

    @classmethod
    def load_from_yaml(cls, path: Path | str) -> SourceMapping:
        """load from a source_mapping.yaml file. refuses with every missing field named."""
        import yaml  # pyyaml; dev dep, same as schema tests

        p = Path(path)
        if not p.exists():
            raise SourceMappingMissing(f"source_mapping.yaml not found at {p}")

        try:
            raw = yaml.safe_load(p.read_text()) or {}
        except Exception as exc:
            raise SourceMappingMissing(f"cannot parse {p}: {exc}") from exc

        if not isinstance(raw, dict):
            raise SourceMappingMissing(f"{p}: expected a mapping, got {type(raw).__name__}")

        problems = [f for f in REQUIRED_FIELDS if not raw.get(f)]
        if problems:
            raise SourceMappingMissing(
                f"{p}: missing required field(s): {', '.join(problems)}"
            )

        return cls(
            he_niche_column=raw["he_niche_column"],
            he_tokens_column=raw["he_tokens_column"],
            sample_column=raw["sample_column"],
        )

    @classmethod
    def from_project_dir(cls, project_dir: Path | str) -> SourceMapping:
        """load from the source_mapping.yaml in a cohort's project directory."""
        return cls.load_from_yaml(Path(project_dir) / "source_mapping.yaml")