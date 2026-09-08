"""alignment, evaluation, and the two encode stages.

a PORT, not a rewrite. the bodies are `scripts/align.py`, `align_classical.py`,
`align_b4.py`, `eval.py` and `eval_alignment_biology.py` - the code that
produced the published grid. this file gives them a typed signature, a record
and a place in the protocol layer. it changes no arithmetic.

the acceptance test is the numbers. `run_align(..., compute=True)` into a
scratch runs-root must reproduce the recorded grid within tolerance. if a
number moves, the port changed something.

two modes, and the file says which it is doing
-----------------------------------------------
    compute=False   DEFAULT. resolve what a previous run produced, refuse when
                    absent. a host serving a published evidence pack can route,
                    gate and describe without ever training anything.
    compute=True    invoke the script. this is the port, and the only way the
                    acceptance test can execute.

every record carries `resolves_only`, true or false, so a reader of the ledger
can tell which stages computed and which read.

the four components, honestly
-----------------------------
    run_align      compute OR resolve
    run_eval       compute OR resolve
    run_embed_he   resolve ONLY
    run_embed_st   resolve ONLY

the encode stages never run an encoder. running a foundation model is a backend
question - GPUs, hours, shards, a scheduler - deliberately outside this layer.
the interface is satisfied and the execution is not claimed, which is what
"encoder-pluggable" asks for and is the honest version of it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from omicstra.config import ProjectConfig  # merges into settings - deferred item 2
from omicstra.records import TransformRecord
from omicstra.settings import settings


# --- typed artifact refs: paths + provenance, never the arrays --------------
class EmbeddingArtifacts(BaseModel):
    modality: str
    encoder: str
    dim: int | None = None
    cache_ref: str | None = None
    n_files: int | None = None   # files in the cache, NOT units of observation
    resolved: bool = False
    why_not: str | None = None


class NicheJoin(BaseModel):
    manifest_ref: str
    niches_dir: str
    sha256: str | None = None
    n_niches: int = 0
    n_subarrays: int = 0
    n_dropped_no_coverage: int = 0


class AlignGrid(BaseModel):
    runs_root: str
    run_refs: dict[str, str]          # run_id -> embeddings_test.parquet
    computed: list[str] = []          # which were trained by THIS call


class EvalResult(BaseModel):
    metrics_ref: str
    evidence_ref: str
    n_tasks: int
    computed: bool = False


class ComputeUnavailable(RuntimeError):
    """the stage needs compute this call is not authorised to do.

    NOT a failure. a host that ships a prebuilt evidence pack is a valid
    deployment - it routes, gates and describes. it simply does not train, and
    saying so beats a traceback from a missing artifact.
    """


def _cohort(project_id: str | None = None) -> dict:
    p = settings.project_root(project_id) / "cohort.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _project_rel(raw: str | Path, project_id: str | None) -> Path:
    """resolve a path a COHORT declared.

    project.json writes these relative to the project directory ("../../data/..."),
    not the package root. resolving against repo_root leaves `../..` in the path
    and produces a wrong answer that looks like a missing artifact.
    """
    q = Path(raw)
    return q if q.is_absolute() else (settings.project_root(project_id) / q).resolve()


def _rec(step_id: str, status: str, resolves_only: bool, **params) -> TransformRecord:
    return TransformRecord(step_id=step_id, status=status, actor="deterministic",
                           params={"resolves_only": resolves_only, **params})


def _run_script(script: str, args: list[str]) -> subprocess.CompletedProcess:
    """the port boundary. the script IS the method; this only invokes it.

    a subprocess rather than an import, deliberately: the scripts set global
    state (seed, device, matplotlib backend) and own their argument parsing.
    importing them would fork the method into two callable paths that could
    drift - and the acceptance test is precisely that the numbers do not move.
    """
    root = settings.resolve(Path("scripts"))
    cp = subprocess.run([sys.executable, str(root / script), *args],
                        capture_output=True, text=True)
    if cp.returncode:
        raise RuntimeError(f"{script} failed ({cp.returncode}):\n{cp.stderr[-2000:]}")
    return cp


# --- the two RESOLVE stages -------------------------------------------------
def _resolve_cache(modality: str, encoder: str, dim: int | None,
                   cache_dir: Path) -> EmbeddingArtifacts:
    if not cache_dir.is_dir():
        return EmbeddingArtifacts(
            modality=modality, encoder=encoder, dim=dim, resolved=False,
            cache_ref=str(cache_dir),
            why_not=f"no cache at {cache_dir.name}. computing it needs a backend "
                    "with a GPU; this stage resolves, it does not encode.")
    files = sorted(cache_dir.glob("*.npy")) or sorted(cache_dir.glob("*.parquet"))
    return EmbeddingArtifacts(
        modality=modality, encoder=encoder, dim=dim, cache_ref=str(cache_dir),
        n_files=len(files), resolved=bool(files),
        why_not=None if files else "cache directory is empty")


def _embed(role: str, step_id: str, cfg: ProjectConfig, subdir: str,
           project_id: str | None) -> tuple[EmbeddingArtifacts, TransformRecord]:
    enc = next((e for e in cfg.encoders if e.role == role), None)
    name = enc.name if enc else "unknown"
    art = _resolve_cache(step_id.split("_", 1)[1], name, enc.dim if enc else None,
                         settings.resolve(settings.data_dir) / "embeddings" / subdir.format(name=name))
    return art, _rec(step_id, "pass" if art.resolved else "not_applicable",
                     resolves_only=True, encoder=name,
                     backend=_cohort(project_id).get("compute_backend") or "none",
                     why_not=art.why_not)


def run_embed_he(cfg: ProjectConfig, project_id: str | None = None):
    """resolve cached H&E embeddings. does NOT run the encoder."""
    return _embed("primary", "embed_he", cfg, "{name}_niche", project_id)


def run_embed_st(cfg: ProjectConfig, project_id: str | None = None):
    """resolve cached molecular embeddings. does NOT run the encoder."""
    return _embed("st", "embed_st", cfg, "{name}", project_id)


# --- the waist --------------------------------------------------------------
def run_niche_join(cfg: ProjectConfig, project_id: str | None = None
                   ) -> tuple[NicheJoin, TransformRecord]:
    """resolve the cached niche join - the table every downstream stage reads.

    counts come from the manifest the join itself wrote, never recomputed here:
    the funnel is a property of that build, and two sources for one number is
    how they disagree.
    """
    # cfg.niches_dir is written relative to the PROJECT directory ("../../data/..."),
    # not the package root. resolving it against repo_root yields a path with
    # `../..` still in it that happens not to exist - a wrong answer that looks
    # like a missing artifact.
    d = _project_rel(cfg.niches_dir, project_id) if cfg.niches_dir else None
    if d is None or not d.is_dir():
        raise ComputeUnavailable(
            f"no niche join at {d}. it is built by scripts/build_niche_join.py, "
            "which needs both modalities' caches.")
    man = d / "manifest.json"
    m = json.loads(man.read_text()) if man.is_file() else {}
    nj = NicheJoin(
        manifest_ref=str(man), niches_dir=str(d),
        sha256=(m.get("sources") or {}).get("gpath2vec_sha256"),
        n_niches=m.get("n_niches_total_post_intersection", 0),
        n_subarrays=m.get("n_subarrays_enumerated", len(sorted(d.glob("*.parquet")))),
        n_dropped_no_coverage=m.get("n_niches_dropped_no_gpath2vec_coverage", 0))
    return nj, _rec("niche_join", "pass", resolves_only=True,
                    n_niches=nj.n_niches, n_dropped=nj.n_dropped_no_coverage)


# --- the two COMPUTE-OR-RESOLVE stages --------------------------------------
def run_align(cfg: ProjectConfig, nj: NicheJoin, run_ids: list[str] | None = None,
              project_id: str | None = None, *, compute: bool = False,
              runs_root: Path | None = None) -> tuple[AlignGrid, TransformRecord]:
    """resolve the grid, or train the runs that are missing.

    contrastive runs are driven by their config; classical baselines are driven
    against a reference split, because a baseline must see exactly the split its
    leader saw or the comparison is between two different held-out sets.
    """
    root = Path(runs_root) if runs_root else _project_rel(cfg.runs_dir, project_id)
    ids = run_ids or (list(cfg.contrastive_runs) + list(cfg.classical_runs))
    missing = [r for r in ids if not (root / r / "embeddings_test.parquet").is_file()]

    if missing and not compute:
        raise ComputeUnavailable(
            f"{len(missing)} run(s) not trained under {root}: {missing[:4]}. "
            "pass compute=True to train them, or point at a runs root that has "
            "them. this stage resolves by default.")

    computed: list[str] = []
    for rid in missing:
        base = rid.split("_")[0]
        cfg_path = settings.resolve(Path("configs/v3")) / f"{rid}.json"
        if cfg_path.is_file():                                  # contrastive
            _run_script("align.py", ["--config", str(cfg_path),
                                     "--niches-dir", nj.niches_dir,
                                     "--runs-root", str(root)])
        else:
            # a baseline must see EXACTLY the split its leader saw, or the
            # comparison is between two different held-out sets.
            ref = root / (cfg.contrastive_runs[0] if cfg.contrastive_runs else "R1_v3") / "split.json"
            common = ["--reference-split", str(ref), "--run-id", rid,
                      "--niches-dir", nj.niches_dir, "--runs-root", str(root)]
            if base == "B4":
                # random-init has its own script and takes no --baseline.
                # routing it through align_classical would produce a CCA number
                # wearing B4's name - a fabricated control.
                _run_script("align_b4.py", common)
            elif base in _BASELINE_OF:
                _run_script("align_classical.py", ["--baseline", _BASELINE_OF[base], *common])
            else:
                raise ValueError(
                    f"no script known for baseline {rid!r}. add it to _BASELINE_OF "
                    "or give it a config - guessing produces a number under the "
                    "wrong name.")
        computed.append(rid)

    refs = {r: str(root / r / "embeddings_test.parquet")
            for r in ids if (root / r / "embeddings_test.parquet").is_file()}
    return (AlignGrid(runs_root=str(root), run_refs=refs, computed=computed),
            _rec("align", "pass", resolves_only=not computed,
                 n_runs=len(refs), computed=computed))


_BASELINE_OF = {"B1": "cca", "B2": "procrustes", "B3": "unaligned"}

# H1 scores the whole grid in one call; H2/H3 take ONE run at a time.
_EVAL_SCRIPT = {"H1": "eval.py",
                "H2": "eval_alignment_biology.py",
                "H3": "eval_alignment_biology.py"}


def run_eval(cfg: ProjectConfig, grid: AlignGrid, project_id: str | None = None,
             *, compute: bool = False, hypotheses: tuple[str, ...] = ("H1",),
             evidence_out: Path | None = None
             ) -> tuple[EvalResult, TransformRecord]:
    """score the grid, and resolve the cohort's evidence pack.

    these are two different things and the boundary is deliberate.

    SCORING is mechanical: eval.py writes `{runs_root}/eval/{H}/summary.json`,
    eval_alignment_biology.py writes its own per-run outputs. compute=True runs
    them, and a rerun must reproduce those numbers.

    the EVIDENCE PACK is curated. no script writes routing_evidence.json,
    because deciding which metric answers which task family, what the floors
    are, and which caveat belongs on which candidate is editorial work - the
    file's own notes argue with the report in places. so this stage RESOLVES
    the pack and never generates it. generating it would be inventing the
    judgements it records.

    that is the same boundary the router draws elsewhere: mechanical things are
    computed, judgements are declared and attributed.
    """
    # evidence_out defaults to the project root. the acceptance test passes a
    # scratch path so a compute run cannot overwrite the very file it is
    # supposed to be diffed against.
    ev = Path(evidence_out) if evidence_out else (
        settings.project_root(project_id) / "routing_evidence.json")
    computed = False

    if compute:
        for h in hypotheses:
            script = _EVAL_SCRIPT.get(h)
            if script is None:
                raise ValueError(f"no eval script for hypothesis {h!r}")
            if script == "eval.py":                      # scores the whole grid
                _run_script(script, ["--hypothesis", h, "--runs", *grid.run_refs,
                                     "--runs-dir", grid.runs_root])
            else:                                        # one run at a time
                for rid in grid.run_refs:
                    _run_script(script, ["--run-id", rid, "--runs-dir", grid.runs_root])
        computed = True

    summaries = [str(p) for p in
                 sorted(Path(grid.runs_root).glob("eval/*/summary.json"))]

    if not ev.is_file():
        if computed:
            # compute WAS available and did run. calling this "unavailable"
            # would make the ledger say "not authorised" when the truth is
            # "broke", which is the more expensive thing to misreport.
            # scoring succeeded; the pack is simply not curated yet. that is a
            # real state, not a breakage - say which it is.
            raise ComputeUnavailable(
                f"scored {list(hypotheses)} into {grid.runs_root}/eval/ "
                f"({len(summaries)} summary file(s)), but no curated evidence "
                f"pack at {ev}. the pack is written by a person from those "
                "summaries; scoring does not produce it.")
        raise ComputeUnavailable(
            "no routing_evidence.json for this cohort. without it the cohort is "
            "not routable - which the router already reports, rather than "
            "inheriting another cohort's winner.")
    d = json.loads(ev.read_text())
    return (EvalResult(metrics_ref=str(Path(grid.runs_root) / "eval"),
                       evidence_ref=str(ev), n_tasks=len(d.get("tasks", {})),
                       computed=computed),
            _rec("eval", "pass", resolves_only=not computed,
                 n_tasks=len(d.get("tasks", {})), n_summaries=len(summaries),
                 evidence_is_curated=True))


def run_qc(cfg: ProjectConfig, project_id: str | None = None):
    """DEPRECATED. graphs/eda.py owns the admissibility gate.

    kept as a signature so nothing importing it breaks silently. two answers to
    one question is worse than one answer in the wrong place.
    """
    raise NotImplementedError(
        "the admissibility gate is graphs/eda.py - use `omicstra eda`.")