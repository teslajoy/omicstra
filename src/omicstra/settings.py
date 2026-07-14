"""runtime settings - all paths injected, nothing bound to __file__ location.

this kills the scripts/ `Path(__file__).parents[1]` portability blocker: every
stage / tool / agent reads paths through Settings, so the code runs the same
under src/, a subprocess, a temporal activity, or a nextflow process.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMICSTRA_", env_file=".env", extra="ignore"
    )

    # repo-relative by default; override per machine via env or constructor
    repo_root: Path = Path.cwd()
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    knowledge_dir: Path = Path("knowledge")
    projects_dir: Path = Path("projects")
    compute_backend: str = "local"  # local | cloud | slurm

    def resolve(self, p: Path) -> Path:
        # relative paths resolve under repo_root; absolute passthrough
        return p if p.is_absolute() else (self.repo_root / p)


settings = Settings()