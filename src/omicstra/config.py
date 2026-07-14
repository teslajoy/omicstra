"""ProjectConfig - a cohort loaded as data, not hardcoded constants.

tnbc-92 is instance-zero: it loads from projects/tnbc-92/project.json. a new
cohort is a new project.json. this is the generalization surface.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


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

    @classmethod
    def load(cls, project_id: str, projects_dir: str | Path) -> "ProjectConfig":
        p = Path(projects_dir) / project_id / "project.json"
        return cls.model_validate(json.loads(p.read_text()))