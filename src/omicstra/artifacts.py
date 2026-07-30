"""artifact store - owns the runs/{project}/ schema. one place to change for v4.

the read path resolves cached refs here (a -resume-style cache hit); the compute
path writes here. keeping this the single schema-owner means a new run layout is
a one-file change, not a sweep.
"""
from __future__ import annotations

import json

from .config import ProjectConfig


class ArtifactStore:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        # cohort paths resolve against the PROJECT root, never the package
        self.runs_dir = cfg.runs_path

    def hypothesis_summary(self, hypothesis: str) -> dict:
        # runs/{project}_v3/eval/{H1|H2|H3}/summary.json
        p = self.runs_dir / "eval" / hypothesis / "summary.json"
        return json.loads(p.read_text())

    def run_config(self, run_id: str) -> dict:
        return json.loads((self.runs_dir / run_id / "run_config.json").read_text())

    def verify_gpath2vec_sha(self, manifest: dict) -> bool:
        got = manifest.get("sources", {}).get("gpath2vec_sha256")
        return got == self.cfg.gpath2vec_sha256