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

from omicstra.contracts.project import ProjectConfig  # merges into settings - deferred item 2
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
def run_niche_join(cfg: ProjectConfig, project_id: str | None = None, *,
                   compute: bool = False, out_dir: Path | None = None,
                   subarrays: list[str] | None = None
                   ) -> tuple[NicheJoin, TransformRecord]:
    """resolve the cached niche join - or build it, into a directory of its own.

    counts come from the manifest the join itself wrote, never recomputed here:
    the funnel is a property of that build, and two sources for one number is
    how they disagree.

    compute=True runs `build_niche_join.py`. the cohort's declared join is an
    oracle - every port is diffed against it - so a build targets `out_dir` and
    is refused into any declared read-only path. the pathway table it joins is
    sha-locked: a build that silently used a different gpath2vec would produce a
    table wearing the published name.
    """
    # cfg.niches_dir is written relative to the PROJECT directory ("../../data/..."),
    # not the package root. resolving it against repo_root yields a path with
    # `../..` still in it that happens not to exist - a wrong answer that looks
    # like a missing artifact.
    d = _project_rel(cfg.niches_dir, project_id) if cfg.niches_dir else None
    if compute:
        if out_dir is None:
            raise ComputeUnavailable(
                "building the join needs an out_dir. the cohort's declared join is "
                "what the ports are diffed against, so a build never writes into it.")
        d = Path(out_dir).resolve()
        _refuse_protected(cfg, d, "build the niche join into")
        args = ["--out-dir", str(d)]
        pw = cfg.pathway_cluster_embeddings
        if pw:
            args += ["--gpath2vec-parquet", str(_project_rel(pw, project_id))]
        if subarrays:
            # a SUBSET build, for exercising the path rather than producing a
            # cohort. the manifest records how many subarrays it covers, so a
            # partial table cannot be mistaken for the cohort's join by anything
            # that reads the counts - which is every stage downstream.
            args += ["--subarrays", *subarrays]
        _run_script("build_niche_join.py", args)
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
    declared_sha = cfg.gpath2vec_sha256
    if declared_sha and nj.sha256 and nj.sha256 != declared_sha:
        raise ComputeUnavailable(
            f"the join at {d} was built from gpath2vec {nj.sha256[:12]}, and this cohort "
            f"declares {declared_sha[:12]}. a table built from a different pathway build "
            "would carry the published name over different numbers.")
    return nj, _rec("niche_join", "pass", resolves_only=not compute,
                    n_niches=nj.n_niches, n_dropped=nj.n_dropped_no_coverage,
                    out_dir=str(d))


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
    if missing:
        _refuse_protected(cfg, root, "train into")

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

# the stages that SCORE each hypothesis, in order, as (script, scope).
#
# the map follows the artifacts the evidence pack CITES, not the rollups. eval.py
# writes `eval/{H}/summary.json` for every hypothesis and stays first, but for H2
# and H3 that rollup is diagnostic: no routed number is read from it.
#
#   H2-C  subject_identity_suppression cites eval/H2/metrics_h2_ci.json -> h2c.
#         eval_alignment_biology.py writes each run's eval/biology.parquet (the
#         TIME and patient z), then eval_h2_bootstrap_ci.py derives the ratio and
#         its interval from those.
#   H3    pathway_transfer cites eval/H3/pathway_cca_gpath2vec_v3/
#         per_pathway_cca.parquet + perm_nulls.parquet, written by
#         eval_h3_pathway_cca_gpath2vec_v2.py.
#
# NOT mapped: H2-A's point estimates (mc_coherence.parquet) come from a script
# that lives in scripts/_scratch/, and the H3 DAG decomposition is deferred. both
# wait; a compute run for H2 refreshes h2a's intervals but not its point table.
#
# two earlier versions of this map were wrong. the first sent H2 and H3 to the
# biology script alone, which never writes a rollup. the second put the biology
# script under H3, where it scores nothing H3 cites - biology.parquet is H2-C.
_EVAL_STAGES = {
    "H1": (("eval.py", "rollup"),),
    "H2": (("eval.py", "rollup"),
           ("eval_alignment_biology.py", "per_run"),
           ("eval_h2_bootstrap_ci.py", "grid_ci")),
    "H3": (("eval.py", "rollup"),
           ("eval_h3_pathway_cca_gpath2vec_v2.py", "pathway_cca")),
}

# the settings that produced the cited artifacts, with where each is evidenced.
# a script default is not a declaration: eval_alignment_biology.py defaults to
# 100,000 permutations, and the published biology.parquet was built with 10,000.
_AS_BUILT = {
    "eval_alignment_biology.py": {
        "n_permutations": 10000,   # biology.parquet p_floor = 1e-4 on every row
        "n_rho_null": 1000,        # biology_rho.parquet n_permutations
    },
    "eval_h2_bootstrap_ci.py": {
        "n_bootstrap": 200,        # metrics_h2_ci.json n_bootstrap_h2a
    },
    "eval_h3_pathway_cca_gpath2vec_v2.py": {
        "n_perms": 500,            # pathway_cca_gpath2vec_v3/provenance.json
        "n_test_pats": 3,          # provenance.json
        "allow_set_size_drift": True,   # provenance.json set_size_drift_allowed
        "out_subdir": "eval/H3/pathway_cca_gpath2vec_v3",
    },
}


def _stage_args(script: str, scope: str, h: str, cfg: ProjectConfig,
                grid: AlignGrid, project_id: str | None) -> list[list[str]]:
    """one argument list per invocation of a stage. per_run stages get one each."""
    root, runs, seed = grid.runs_root, list(grid.run_refs), str(cfg.seed)
    if scope == "rollup":
        return [["--hypothesis", h, "--runs", *runs, "--runs-dir", root]]

    niches = _project_rel(cfg.niches_dir, project_id) if cfg.niches_dir else None
    if niches is None or not niches.is_dir():
        raise ComputeUnavailable(f"{script} needs the niche join; none at {niches}")
    ab = _AS_BUILT[script]

    if scope == "per_run":
        return [["--run-id", rid, "--runs-dir", root, "--niches-dir", str(niches),
                 "--n-permutations", str(ab["n_permutations"]),
                 "--n-rho-null", str(ab["n_rho_null"]), "--seed", seed]
                for rid in runs]
    if scope == "grid_ci":
        return [["--runs-dir", root, "--niches-dir", str(niches), "--runs", ",".join(runs),
                 "--n-bootstrap", str(ab["n_bootstrap"]), "--seed", seed]]
    if scope == "pathway_cca":
        pkl = _pathway_node_embeddings(cfg, project_id)
        args = ["--embeddings-pkl", str(pkl), "--runs-dir", root, "--runs", ",".join(runs),
                "--out-dir", str(Path(root) / ab["out_subdir"]),
                "--n-perms", str(ab["n_perms"]), "--n-test-pats", str(ab["n_test_pats"])]
        if ab["allow_set_size_drift"]:
            args.append("--allow-set-size-drift")
        return [args]
    raise ValueError(f"unknown eval stage scope {scope!r} for {script}")


def _pathway_node_embeddings(cfg: ProjectConfig, project_id: str | None) -> Path:
    """the declared gpath2vec node-embedding pickle, sha-locked when a sha is declared.

    the H3 script takes it as a required argument with no default, to prevent a
    legacy build being scored by accident. the same reasoning applies one level
    up: the path is read from the cohort's declaration, never guessed.
    """
    if not cfg.pathway_node_embeddings:
        raise ComputeUnavailable(
            "H3 pathway CCA needs `pathway_node_embeddings` declared in project.json - "
            "the gpath2vec node-embedding pickle the pathway sets are embedded from.")
    p = _project_rel(cfg.pathway_node_embeddings, project_id)
    if not p.is_file():
        raise ComputeUnavailable(f"declared pathway_node_embeddings not found: {p}")
    if cfg.pathway_node_embeddings_sha256:
        import hashlib
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != cfg.pathway_node_embeddings_sha256:
            raise ComputeUnavailable(
                f"{p.name} sha256 {got[:12]} != declared "
                f"{cfg.pathway_node_embeddings_sha256[:12]} - a different build would be "
                "scored under the published name")
    return p


def _refuse_protected(cfg: ProjectConfig, target: Path, verb: str) -> None:
    from omicstra.dispatch import WouldOverwriteOracle, protected_by, read_only_paths

    ro = protected_by(target, read_only_paths(cfg))
    if ro is not None:
        raise WouldOverwriteOracle(
            f"would {verb} {Path(target).resolve()}, inside {ro}, which this cohort "
            "declares read-only. a finished grid is what reruns are diffed against; "
            "writing into it makes the diff compare a rerun against itself and pass. "
            "pass an out_root outside every declared read-only path.")


# what a rerun reads from each run directory. linked, not copied: they are large,
# and no eval stage writes them - every write lands under an eval/ directory,
# which staging creates fresh. `_verify_links_untouched` holds that to the file.
_STAGE_LINK = ("embeddings_test.parquet", "checkpoint.pt")
_STAGE_COPY = ("run_config.json", "split.json", "metrics_h1_raw.json")


def stage_grid(cfg: ProjectConfig, grid: AlignGrid, out_root: Path) -> AlignGrid:
    """a fresh runs root holding a finished grid's inputs and none of its outputs.

    the rerun scores into `out_root`, so every file it writes is new, and the
    finished grid it reads from stays exactly as the evidence pack cites it.
    """
    out = Path(out_root).resolve()
    _refuse_protected(cfg, out, "stage a rerun in")
    src = Path(grid.runs_root)
    refs = {}
    for rid in grid.run_refs:
        s, d = src / rid, out / rid
        d.mkdir(parents=True, exist_ok=True)
        for name in _STAGE_LINK:
            if (s / name).is_file() and not (d / name).exists():
                (d / name).symlink_to((s / name).resolve())
        for name in _STAGE_COPY:
            if (s / name).is_file() and not (d / name).exists():
                (d / name).write_bytes((s / name).read_bytes())
        refs[rid] = str(d / "embeddings_test.parquet")
    return AlignGrid(runs_root=str(out), run_refs=refs, computed=[])


def _link_state(grid: AlignGrid) -> dict[str, tuple[int, int]]:
    root = Path(grid.runs_root)
    return {str(p): (p.resolve().stat().st_mtime_ns, p.resolve().stat().st_size)
            for rid in grid.run_refs for name in _STAGE_LINK
            if (p := root / rid / name).is_symlink()}


def _verify_links_untouched(before: dict[str, tuple[int, int]], grid: AlignGrid) -> None:
    from omicstra.dispatch import WouldOverwriteOracle

    changed = [p for p, st in _link_state(grid).items() if before.get(p) != st]
    if changed:
        raise WouldOverwriteOracle(
            f"a stage wrote through a staged link into the finished grid: {changed[:3]}. "
            "the rerun's results are not trustworthy and the grid needs checking "
            "against its pins.")


def run_eval(cfg: ProjectConfig, grid: AlignGrid, project_id: str | None = None,
             *, compute: bool = False, hypotheses: tuple[str, ...] = ("H1",),
             evidence_out: Path | None = None, out_root: Path | None = None
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
        # scoring writes into the runs root. a finished grid is declared
        # read-only, so a rerun is staged into out_root and scores there.
        if out_root is not None:
            grid = stage_grid(cfg, grid, out_root)
        _refuse_protected(cfg, Path(grid.runs_root), "score into")
        before = _link_state(grid)
        for h in hypotheses:
            stages = _EVAL_STAGES.get(h)
            if stages is None:
                raise ValueError(f"no eval stages for hypothesis {h!r}")
            # every argument list is built before any script runs, so an
            # undeclared input refuses the hypothesis rather than half-scoring it
            plan = [(script, args) for script, scope in stages
                    for args in _stage_args(script, scope, h, cfg, grid, project_id)]
            for script, args in plan:
                _run_script(script, args)
        _verify_links_untouched(before, grid)
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
                 runs_root=grid.runs_root, evidence_is_curated=True))


def run_qc(cfg: ProjectConfig, project_id: str | None = None):
    """DEPRECATED. graphs/eda.py owns the admissibility gate.

    kept as a signature so nothing importing it breaks silently. two answers to
    one question is worse than one answer in the wrong place.
    """
    raise NotImplementedError(
        "the admissibility gate is graphs/eda.py - use `omicstra eda`.")