"""ProjectConfig - a cohort loaded as data, not hardcoded constants.

tnbc-92 is instance-zero: it loads from projects/tnbc-92/project.json. a new
cohort is a new project.json in its own directory. this is the generalization
surface.

`project_id` is authoritative from the file, never inferred from the directory
name - a cohort directory may be called anything.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .settings import settings


class EncoderSpec(BaseModel):
    name: str
    dim: int
    role: str          # primary | alternative | st | pathway
    raw: bool = False  # True = no stain-norm / no z-score before the FM


class ProjectConfig(BaseModel):
    project_id: str
    platform: str
    k_neighbors: int = 6
    seed: int = 42
    split: str = "patient_stratified_85_15"

    encoders: list[EncoderSpec] = []
    gpath2vec_sha256: str | None = None  # sha-lock; verified on every read

    h2_label: str = "mc_megacluster"       # audit-correct niche-level label
    supervision: str = "mc_weights_niche"  # supervision-only, never in st_features
    headline_view: str = "z_he"            # the confounder-clean view
    hypothesis_metric: str = "auc"         # not R@1 (at floor at this scale)

    contrastive_runs: list[str] = []
    classical_runs: list[str] = []

    niches_dir: str = ""
    runs_dir: str = ""

    # where this cohort was loaded from. not serialised back out.
    project_dir: Path | None = None

    @classmethod
    def from_dir(cls, project_dir: str | Path) -> "ProjectConfig":
        """load a cohort from its own directory - the shipped path."""
        d = Path(project_dir)
        p = d / "project.json"
        if not p.exists():
            raise FileNotFoundError(
                f"no project.json in {d} - run `omicstra init --project-dir {d}` first"
            )
        cfg = cls.model_validate(json.loads(p.read_text()))
        cfg.project_dir = d
        return cfg

    @classmethod
    def load(cls, project_id: str | None = None,
             projects_dir: str | Path | None = None) -> "ProjectConfig":
        """resolve a cohort through the settings boundary."""
        if projects_dir is not None:
            return cls.from_dir(Path(projects_dir) / project_id)
        return cls.from_dir(settings.project_root(project_id))

    # --- path resolution: cohort paths resolve against the PROJECT root ---
    def path(self, value: str | Path) -> Path:
        """resolve a cohort-declared path.

        absolute passes through; relative resolves against this cohort's
        project_dir - NEVER against the package. resolving a cohort's data
        against the package root is how an external cohort ends up pointing
        into the omicstra install.

        the tnbc-92 fixture keeps its data at the repo root, outside its own
        project root, so its project.json declares `../../data/...`. explicit
        and visible, rather than a silent fallback.
        """
        p = Path(value)
        if p.is_absolute():
            return p
        base = self.project_dir or Path(".")
        return (base / p).resolve()

    # --- conventional sub-paths inside a project root --------------------
    @property
    def steps_dir(self) -> Path:
        return (self.project_dir or Path(".")) / "steps"

    @property
    def eda_summary_path(self) -> Path:
        return (self.project_dir or Path(".")) / "eda_summary.json"

    @property
    def niches_path(self) -> Path:
        return self.path(self.niches_dir)

    @property
    def runs_path(self) -> Path:
        return self.path(self.runs_dir)
