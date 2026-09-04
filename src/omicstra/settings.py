"""runtime settings - all paths injected, nothing bound to __file__ location.

this kills the scripts/ `Path(__file__).parents[1]` portability blocker: every
stage / tool / agent reads paths through Settings, so the code runs the same
under src/, a subprocess, a temporal activity, or a nextflow process.

two roots, deliberately separate:

  PACKAGE root   this repo / the wheel. code + the generalizable contract.
                 cohort-free - nothing here may name a cohort.
  PROJECT root   one cohort. `project_dir` is BOTH the security boundary and
                 the cohort selector. lives anywhere on disk, usually its own
                 git repo, never inside the package.

tnbc-92 is the fixture and happens to sit at ./projects/tnbc-92, but it is
reached through the same `project_root()` call as any external cohort - one
code path, different default.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _package_repo_root() -> Path:
    """repo root derived from the package, not the working directory.

    package data (the contract) must be findable no matter where the process
    was launched - resolving it against cwd breaks the moment the server runs
    from a project directory. this is the ONE place __file__ is consulted;
    every cohort path still comes through injected settings.

    OPEN: `configs/` sits at repo root, so it is not inside the wheel. shipping
    a non-editable install needs it moved under src/omicstra/ or declared as
    package data. tracked as a packaging item, not a blocker for dev installs.
    """
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMICSTRA_", env_file=".env", extra="ignore"
    )

    # where the PACKAGE lives. override per machine via env or constructor.
    repo_root: Path = _package_repo_root()
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    knowledge_dir: Path = Path("knowledge")
    configs_dir: Path = Path("configs")

    # the cohort root. when set, it IS the boundary - project_id is not
    # consulted and nothing outside this directory is addressable.
    project_dir: Path | None = None

    # dev convenience only: an in-repo registry of cohorts. used when
    # project_dir is unset, which is the fixture path, not the shipped path.
    projects_dir: Path = Path("projects")

    compute_backend: str = "local"  # local | cloud | slurm

    def resolve(self, p: Path) -> Path:
        # relative paths resolve under repo_root; absolute passthrough
        return p if p.is_absolute() else (self.repo_root / p)

    def default_project_id(self) -> str | None:
        reg = self.resolve(self.projects_dir) / "registry.json"
        if not reg.exists():
            return None
        return json.loads(reg.read_text()).get("default")

    def _bound_project_id(self, root: Path) -> str:
        """the cohort a project_dir actually holds.

        project.json is authoritative; the directory name is the fallback, since
        `init` defaults the id to the directory name anyway.
        """
        cfg = root / "project.json"
        if cfg.exists():
            try:
                return json.loads(cfg.read_text()).get("project_id") or root.name
            except (json.JSONDecodeError, OSError):
                pass
        return root.name

    def project_root(self, project_id: str | None = None) -> Path:
        """the one cohort directory this call may read.

        project_dir wins outright - a tool call cannot escape the boundary by
        passing a different project_id. falls back to the in-repo registry.
        """
        if self.project_dir is not None:
            root = self.resolve(self.project_dir)
            # the boundary REFUSES; it does not substitute. winning outright
            # while staying silent turns a security boundary into a data mixup -
            # ask about cohort B, receive cohort A's evidence, with nothing said.
            # that is precisely the "inheriting another cohort's winner" failure
            # the package/project split exists to stop.
            if project_id and project_id != self._bound_project_id(root):
                raise ValueError(
                    f"project_id={project_id!r} does not match the bound cohort "
                    f"{self._bound_project_id(root)!r} (project_dir={root}). "
                    "the boundary refuses rather than silently serving another "
                    "cohort's data. unset OMICSTRA_PROJECT_DIR, or pass the "
                    "matching project_id."
                )
            return root
        pid = project_id or self.default_project_id()
        if not pid:
            raise ValueError(
                "no cohort selected: set OMICSTRA_PROJECT_DIR (or --project-dir), "
                "pass a project_id, or provide projects/registry.json with a default"
            )
        return self.resolve(self.projects_dir) / pid


settings = Settings()