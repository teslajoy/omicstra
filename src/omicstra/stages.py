"""the six deterministic pipeline stages - THE SEAM.

these typed signatures are frozen. what runs INSIDE a stage is a swappable body:
  - phase 2: a direct python call to the ported script
  - phase 3 (optional): `nextflow run stage.nf`
the temporal workflow, narration contract, and artifact store never change when
the body swaps. keep these signatures stable.
"""
from __future__ import annotations

from pydantic import BaseModel

from .config import ProjectConfig


# --- typed artifact refs: paths + provenance, not the arrays themselves ---
class QCReport(BaseModel):
    verdict: str  # proceed | proceed_with_caution | stop
    eda_summary_ref: str


class HEArtifacts(BaseModel):
    virchow2_niche_ref: str
    cell_tokens_ref: str
    n_niches: int
    sha256: str


class STArtifacts(BaseModel):
    novae_ref: str
    gpath2vec_ref: str
    st_features_ref: str
    mc_weights_ref: str  # separate slot - supervision only, never in st_features
    sha256: str


class NicheJoin(BaseModel):
    manifest_ref: str
    sha256: str
    n_niches: int


class AlignGrid(BaseModel):
    run_refs: dict[str, str]  # run_id -> embeddings_test.parquet


class EvalResult(BaseModel):
    metrics_ref: str
    guard_report_ref: str
    routing_table_ref: str
    winner_ref: str | None = None


def run_qc(cfg: ProjectConfig) -> QCReport:
    raise NotImplementedError("phase 2: wrap the EDA gate")


def run_embed_he(cfg: ProjectConfig) -> HEArtifacts:
    raise NotImplementedError("phase 2: wrap extract_virchow2_niche (gpu)")


def run_embed_st(cfg: ProjectConfig) -> STArtifacts:
    raise NotImplementedError("phase 2: wrap scale_embeddings - novae + gpath2vec (gpu)")


def run_niche_join(cfg: ProjectConfig, he: HEArtifacts, st: STArtifacts) -> NicheJoin:
    raise NotImplementedError("phase 2: wrap build_niche_join")


def run_align_grid(cfg: ProjectConfig, nj: NicheJoin) -> AlignGrid:
    raise NotImplementedError("phase 2: wrap align / align_classical / align_b4")


def run_eval(cfg: ProjectConfig, grid: AlignGrid) -> EvalResult:
    raise NotImplementedError("phase 2: wrap eval + guard library")