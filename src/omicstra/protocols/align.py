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

from omicstra.contracts.project import (
    ProjectConfig,  # merges into settings - deferred item 2
)
from omicstra.protocols import Ctx, Step, build_protocol
from omicstra.records import DiagnosticRecord, TransformRecord
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
        _build_join(cfg, project_id, d, subarrays)
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




def _build_join(cfg: ProjectConfig, project_id: str | None, out: Path,
                subarrays: list[str] | None) -> None:
    """build the join by running JOIN_STEPS, once per sample.

    this is what `compute=True` used to shell out to `scripts/build_niche_join.py`
    for. the script stays as the historical record of what produced the published
    parquets - and `script_map/build_niche_join.md` maps all 489 lines of it -
    but it is no longer the method: it names this cohort 27 times, and a second
    cohort could not call it.

    `dispatch` owns the loop: one shard per sample, idempotent on the output
    file, three attempts, and a total failure refused rather than reported as a
    bad cohort. the chain owns the funnel, and the manifest is written from what
    the chain recorded rather than from a second count.
    """
    import json as _j

    from omicstra.adapters.join import JoinDeclarationMissing
    from omicstra.adapters.join import arrays as join_arrays
    from omicstra.adapters.join import samples as join_samples
    from omicstra.dispatch import Shard, atomic_write, run_shards
    from omicstra.protocols import build_protocol
    from omicstra.protocols.join import JOIN_STEPS

    root = settings.project_root(project_id)
    params = (_cohort(project_id) or {}).get("join_params")
    if not params:
        raise ComputeUnavailable(
            "this cohort declares no `join_params`. every step of the join - the "
            "niche width, which assay is the anchor, which denominators must be "
            "uniform, the standardisation group - is the cohort's to declare, "
            "and a default here would be the seed cohort's answer.")
    try:
        ids = subarrays or join_samples(root)
    except JoinDeclarationMissing as e:
        raise ComputeUnavailable(str(e)) from e

    chain = build_protocol(JOIN_STEPS, "niche_join")
    records: dict[str, dict] = {}
    shapes: dict[str, dict] = {}

    def body(shard: Shard, heartbeat) -> None:
        import pandas as pd

        ctx = join_arrays(root, shard.id)
        # the data columns are a property of the frames the reader produced, so
        # they are derived here rather than declared a second time - two sources
        # for one column list is how they come to disagree.
        p = {k: dict(v) for k, v in params.items() if k != "note"}
        cols = {a: list(f.columns) for a, f in ctx["assays"].items()}
        for step, key in (("pool_continuous", "pooled_assays"),
                          ("carry_per_unit", "per_unit_assays"),
                          ("stack_unit_tokens", "token_assays")):
            if step in p:
                p[step]["data_columns"] = {a: cols[a] for a in p[step][key]
                                           if a in cols}
        heartbeat(shard.id)
        recs = chain.invoke({**ctx, "params": p})
        failed = {k: v["result"] for k, v in recs.items()
                  if v["status"] in ("fail", "error")}
        if failed:
            raise ComputeUnavailable(f"{shard.id}: {failed}")

        c = dict(ctx)
        for st in JOIN_STEPS:
            if st.id in p:
                o = st.fn({**c, "params": p[st.id]})
                if isinstance(o, tuple):
                    c.update(o[1])
        table = c.get("table") or []
        if not table:
            raise ComputeUnavailable(f"{shard.id}: the join produced no rows")
        # a token block is (k+1, dim) and parquet stores 1-D values, so it is
        # FLATTENED on the way to disk - which is what the published build did.
        # the flatten belongs here and not in the step: the step's ordered
        # (k+1, dim) shape is load-bearing, since a model attending over those
        # positions reads a different unit if the order changes. the shape goes
        # in the manifest so a reader can restore it rather than guess.
        import numpy as np

        token_shapes: dict[str, list[int]] = {}
        rows = []
        for r in table:
            out_row = {}
            for k, v in r.items():
                arr = np.asarray(v) if isinstance(v, np.ndarray) else v
                if isinstance(arr, np.ndarray) and arr.ndim > 1:
                    token_shapes[k] = list(arr.shape)
                    out_row[k] = arr.ravel()
                else:
                    out_row[k] = v
            rows.append(out_row)
        shapes[shard.id] = token_shapes
        df = pd.DataFrame(rows).set_index("unit_id")
        atomic_write(shard.output, lambda path: df.to_parquet(path))
        records[shard.id] = recs

    shards = [Shard(id=s, output=out / f"{s}.parquet") for s in ids]
    report = run_shards(shards, body, project_id=project_id)

    # the manifest is written from what the CHAIN recorded. the funnel is a
    # property of the build, and a manifest that recounted it would be a second
    # source for one number.
    per_sample = []
    for sid, recs in sorted(records.items()):
        f = (recs.get("join_funnel") or {}).get("observed") or {}
        anc = (recs.get("anchor_coverage") or {}).get("observed") or {}
        per_sample.append({"subarray": sid, "status": "ok",
                           "n_niches": anc.get("n_out"),
                           "n_niches_pre_intersection": anc.get("n_in"),
                           "n_intersection_dropped_no_gpath2vec": anc.get("n_dropped"),
                           "stages": f.get("stages")})
    (out / "manifest.json").write_text(_j.dumps({
        "run_type": "niche_join_table",
        "built_by": "omicstra.protocols.join.JOIN_STEPS",
        "n_subarrays_enumerated": len(ids),
        "n_ok": len(report.done) + len(report.skipped),
        "n_failed": len(report.failed),
        "n_niches_total_post_intersection": sum(
            r["n_niches"] or 0 for r in per_sample),
        "n_niches_dropped_no_gpath2vec_coverage": sum(
            r["n_intersection_dropped_no_gpath2vec"] or 0 for r in per_sample),
        "sources": {"declared_by": "project.json#join_inputs"},
        # the on-disk form of any column the chain produced as (k+1, dim).
        # without it a reader cannot tell 8,960 numbers from 7 x 1,280.
        "flattened_shapes": next(iter(shapes.values()), {}),
        "per_subarray": per_sample,
    }, indent=2) + "\n")

# --- contrastive alignment: five LCEL steps ----------------------------------
#
# the chain is load_data | split | train | embed_test | evaluate_and_save.
# each step calls the extracted functions in adapters/alignment_data.py and
# training/alignment.py. the functions are unchanged; the steps give them a
# record, timing, and an order assertion.

def _step_load_data(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    from omicstra.adapters.alignment_data import discover_subarrays, load_subarray

    p = ctx["params"]
    niches_dir = Path(p["niches_dir"])
    st_features = tuple(p["st_features"])
    subset_n = p.get("subset_n")

    paths = discover_subarrays(niches_dir)
    if not paths:
        return DiagnosticRecord(
            step_id="load_data", status="fail",
            result=f"no parquets under {niches_dir}",
            decision="cannot train without data"), {}

    if subset_n:
        paths = paths[:subset_n]

    subarrays = [load_subarray(path, st_features) for path in paths]
    n_spots = sum(len(s["spot_id"]) for s in subarrays)
    n_dropped = sum(s["n_dropped_nan"] for s in subarrays)

    return DiagnosticRecord(
        step_id="load_data", status="pass",
        method="load niche-join parquets, materialise feature tensors",
        observed={"n_subarrays": len(subarrays), "n_spots_total": n_spots,
                  "n_dropped_nan": n_dropped, "st_features": list(st_features)},
        result=f"{len(subarrays)} subarrays, {n_spots} spots",
        decision="data loaded",
    ), {"subarrays": subarrays}


def _step_split(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    from omicstra.adapters.alignment_data import split_by_patient

    p = ctx["params"]
    subarrays = ctx["subarrays"]
    val_frac = p["val_frac"]
    test_frac = p["test_frac"]
    seed = p["seed"]
    run_dir = Path(ctx.get("run_dir") or p.get("run_dir", "."))

    train_p, val_p, test_p = split_by_patient(subarrays, val_frac, test_frac, seed)
    train_subs = [s for s in subarrays if s["patient_id"] in set(train_p)]
    val_subs = [s for s in subarrays if s["patient_id"] in set(val_p)]
    test_subs = [s for s in subarrays if s["patient_id"] in set(test_p)]

    split_info = {"train_patients": train_p, "val_patients": val_p,
                  "test_patients": test_p, "seed": seed}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "split.json").write_text(json.dumps(split_info, indent=2))

    return DiagnosticRecord(
        step_id="split", status="pass",
        method="three-way patient split, stratified by archetype when counts allow",
        observed={"n_train_patients": len(train_p), "n_val_patients": len(val_p),
                  "n_test_patients": len(test_p),
                  "n_train_subs": len(train_subs), "n_val_subs": len(val_subs),
                  "n_test_subs": len(test_subs), "seed": seed},
        result=f"{len(train_p)}/{len(val_p)}/{len(test_p)} patients",
        decision="split written",
    ), {"split": {"train": train_subs, "val": val_subs, "test": test_subs,
                  "train_patients": train_p, "val_patients": val_p,
                  "test_patients": test_p}}


def _step_train(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    from omicstra.training.alignment import train_model

    p = ctx["params"]
    split = ctx["split"]
    train_subs, val_subs = split["train"], split["val"]
    run_dir = Path(ctx.get("run_dir") or p["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    he_dim = p.get("he_dim") or train_subs[0]["x_he_niche"].shape[1]
    st_dim = p.get("st_dim") or train_subs[0]["x_st"].shape[1]

    model = train_model(
        he_dim=he_dim, st_dim=st_dim,
        shared_dim=p.get("shared_dim", 512),
        dropout=p.get("dropout", 0.3),
        fusion=p.get("fusion", "late"),
        anisotropic=p.get("anisotropic", False),
        he_tile_dim=p.get("he_tile_dim", 1280),
        loss_name=p["loss"],
        lr=p.get("lr", 5e-4),
        weight_decay=p.get("weight_decay", 1e-3),
        epochs=p.get("epochs", 50),
        patience=p.get("patience", 10),
        seed=p["seed"],
        device=p.get("device", "cpu"),
        train_subs=train_subs,
        val_subs=val_subs,
        run_dir=run_dir,
        tau=p.get("tau", 0.07),
        tau_target=p.get("tau_target", 0.1),
        lam=p.get("barlow_lambda", 5e-3),
    )

    import csv
    log_path = run_dir / "training_log.csv"
    best_cos, epochs_run = 0.0, 0
    if log_path.is_file():
        with log_path.open() as f:
            rows = list(csv.DictReader(f))
            if rows:
                epochs_run = len(rows)
                best_cos = max(float(r["val_cos_sim"]) for r in rows)

    return DiagnosticRecord(
        step_id="train", status="pass",
        method=f"{p.get('fusion', 'late')} fusion, {p['loss']} loss",
        observed={"best_val_cos": best_cos, "epochs_run": epochs_run,
                  "checkpoint": str(run_dir / "checkpoint.pt"),
                  "he_dim": he_dim, "st_dim": st_dim},
        result=f"trained {epochs_run} epochs, best val cos {best_cos:.4f}",
        decision="checkpoint saved",
    ), {"model": model}


def _step_embed_test(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    from omicstra.training.alignment import embed_test

    p = ctx["params"]
    model = ctx["model"]
    test_subs = ctx["split"]["test"]
    device = p.get("device", "cpu")

    emb_df, Z_he, Z_st, X_he, X_st = embed_test(model, test_subs, device=device)

    return DiagnosticRecord(
        step_id="embed_test", status="pass",
        method="forward pass on held-out test subarrays",
        observed={"n_test_spots": len(emb_df),
                  "n_test_subarrays": len(test_subs),
                  "z_dim": Z_he.shape[1] if Z_he.shape[0] else 0},
        result=f"{len(emb_df)} spots embedded",
        decision="test embeddings computed",
    ), {"embeddings": {"emb_df": emb_df, "Z_he": Z_he, "Z_st": Z_st,
                       "X_he": X_he, "X_st": X_st}}


def _step_evaluate_and_save(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    from omicstra.adapters.alignment_data import save_test_artifacts

    p = ctx["params"]
    emb = ctx["embeddings"]
    run_dir = Path(ctx.get("run_dir") or p["run_dir"])

    metrics = save_test_artifacts(
        emb["emb_df"], emb["Z_he"], emb["Z_st"], emb["X_he"], emb["X_st"],
        run_dir=run_dir,
        ks=tuple(p.get("ks", (1, 5, 10))),
        negative_ratio=p.get("negative_ratio", 10),
        negative_seed=p.get("negative_seed", 0),
    )

    return DiagnosticRecord(
        step_id="evaluate_and_save", status="pass",
        method="cross-modal retrieval + CKA on test set",
        observed=metrics,
        result=f"R@1={metrics.get('recall_at_1', 'n/a')}, "
               f"MRR={metrics.get('mrr', 'n/a')}, "
               f"CKA={metrics.get('cka_after', 'n/a'):.4f}"
               if isinstance(metrics.get("cka_after"), float) else
               f"R@1={metrics.get('recall_at_1', 'n/a')}",
        decision="metrics and embeddings written",
    ), {"metrics": metrics}


ALIGN_TRAIN_STEPS: list[Step] = [
    Step(id="load_data", fn=_step_load_data,
         produces=frozenset({"subarrays"}),
         params_space=("niches_dir", "st_features", "subset_n")),
    Step(id="split", fn=_step_split,
         produces=frozenset({"split"}),
         requires=frozenset({"subarrays"}),
         params_space=("val_frac", "test_frac", "seed", "run_dir")),
    Step(id="train", fn=_step_train,
         produces=frozenset({"model"}),
         requires=frozenset({"split"}),
         params_space=("fusion", "he_dim", "st_dim", "shared_dim", "dropout",
                       "he_tile_dim", "anisotropic", "loss", "lr",
                       "weight_decay", "epochs", "patience", "seed",
                       "device", "tau", "tau_target", "barlow_lambda",
                       "run_dir")),
    Step(id="embed_test", fn=_step_embed_test,
         produces=frozenset({"embeddings"}),
         requires=frozenset({"model", "split"}),
         params_space=("device",)),
    Step(id="evaluate_and_save", fn=_step_evaluate_and_save,
         produces=frozenset({"metrics"}),
         requires=frozenset({"embeddings"}),
         params_space=("run_dir", "ks", "negative_ratio", "negative_seed")),
]


def _train_contrastive_chain(run_cfg: dict, niches_dir: str, run_dir: Path) -> dict:
    """invoke the five-step LCEL chain for one contrastive run.

    returns the records dict from build_protocol - one entry per step id,
    each with status, duration_s, params, observed.
    """
    chain = build_protocol(ALIGN_TRAIN_STEPS, "align_train")

    ctx_params = {
        "load_data": {
            "niches_dir": niches_dir,
            "st_features": tuple(run_cfg.get("st_features", ("novae_niche", "gpath2vec_niche"))),
            "subset_n": run_cfg.get("subset_n_subarrays"),
        },
        "split": {
            "val_frac": run_cfg.get("val_frac", 0.15),
            "test_frac": run_cfg.get("test_frac", 0.15),
            "seed": run_cfg["seed"],
            "run_dir": str(run_dir),
        },
        "train": {
            "fusion": run_cfg.get("fusion", "late"),
            "he_dim": None,
            "st_dim": None,
            "shared_dim": run_cfg.get("shared_dim", 512),
            "dropout": run_cfg.get("dropout", 0.3),
            "he_tile_dim": 1280,
            "anisotropic": run_cfg.get("anisotropic", False),
            "loss": run_cfg["loss"],
            "lr": run_cfg.get("lr", 5e-4),
            "weight_decay": run_cfg.get("weight_decay", 1e-3),
            "epochs": run_cfg.get("epochs", 50),
            "patience": run_cfg.get("patience", 10),
            "seed": run_cfg["seed"],
            "device": run_cfg.get("device", "cpu"),
            "tau": run_cfg.get("tau", 0.07),
            "tau_target": run_cfg.get("tau_target", 0.1),
            "barlow_lambda": run_cfg.get("barlow_lambda", 5e-3),
            "run_dir": str(run_dir),
        },
        "embed_test": {
            "device": run_cfg.get("device", "cpu"),
        },
        "evaluate_and_save": {
            "run_dir": str(run_dir),
            "ks": [1, 5, 10],
            "negative_ratio": 10,
            "negative_seed": 0,
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(
        {**run_cfg, **{"method": "package", "params": ctx_params}}, indent=2))

    return chain.invoke({"params": ctx_params, "run_dir": run_dir})


def _step_apply_reference_split(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """read an existing split.json and partition subarrays by patient.

    unlike _step_split (which generates a fresh split), this step reads the
    split from a reference contrastive run so that baselines see exactly the
    same held-out set.
    """
    from omicstra.adapters.alignment_data import build_matrices

    p = ctx["params"]
    subarrays = ctx["subarrays"]
    ref_split_path = Path(p["reference_split"])
    run_dir = Path(ctx.get("run_dir") or p.get("run_dir", "."))

    subs = [s for s in subarrays
            if s["patient_id"] is not None and s["archetype"] is not None]

    split = json.loads(ref_split_path.read_text())
    train_p = set(split["train_patients"])
    test_p = set(split["test_patients"])

    train_subs = [s for s in subs if s["patient_id"] in train_p]
    test_subs = [s for s in subs if s["patient_id"] in test_p]

    X_he_tr, X_st_tr, _ = build_matrices(train_subs)
    X_he_te, X_st_te, meta_te = build_matrices(test_subs)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "split.json").write_text(json.dumps(split, indent=2))

    return DiagnosticRecord(
        step_id="apply_reference_split", status="pass",
        method="apply reference patient split from contrastive run",
        observed={"n_train_subs": len(train_subs),
                  "n_test_subs": len(test_subs),
                  "train_niches": X_he_tr.shape[0],
                  "test_niches": X_he_te.shape[0],
                  "n_dropped_labels": len(subarrays) - len(subs)},
        result=f"{len(train_subs)} train, {len(test_subs)} test subarrays",
        decision="reference split applied",
    ), {"matrices": {"X_he_tr": X_he_tr, "X_st_tr": X_st_tr,
                     "X_he_te": X_he_te, "X_st_te": X_st_te,
                     "meta_te": meta_te}}


def _step_fit_project(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """fit the classical baseline on train matrices, project test to shared space."""
    from omicstra.models.alignment_classical import BASELINES

    import numpy as np
    import pandas as pd

    p = ctx["params"]
    m = ctx["matrices"]
    method = p["method"]
    shared_dim = p.get("shared_dim", 512)

    runner = BASELINES[method]
    kwargs: dict = {}
    if "reg" in p:
        kwargs["reg"] = p["reg"]
    z_he, z_st, k_used = runner(
        m["X_he_tr"], m["X_st_tr"], m["X_he_te"], m["X_st_te"],
        shared_dim, **kwargs)

    meta_te = m["meta_te"]
    emb_rows = []
    for i, row in meta_te.reset_index(drop=True).iterrows():
        emb_rows.append({
            "spot_id": row["spot_id"],
            "subarray": row["subarray"],
            "patient_id": row["patient_id"],
            "archetype": row["archetype"],
            "compartment": row["compartment"],
            "z_he": z_he[i].tolist(),
            "z_st": z_st[i].tolist(),
        })
    emb_df = pd.DataFrame(emb_rows).set_index("spot_id")

    return DiagnosticRecord(
        step_id="fit_project", status="pass",
        method=f"{method} baseline, shared_dim={shared_dim}",
        observed={"k_used": int(k_used), "n_test_spots": len(emb_df)},
        result=f"{method}: k_used={k_used}, {len(emb_df)} test spots",
        decision="baseline fitted and test projected",
    ), {"embeddings": {"emb_df": emb_df, "Z_he": z_he, "Z_st": z_st,
                       "X_he": m["X_he_te"], "X_st": m["X_st_te"]}}


CLASSICAL_BASELINE_STEPS: list[Step] = [
    Step(id="load_data", fn=_step_load_data,
         produces=frozenset({"subarrays"}),
         params_space=("niches_dir", "st_features", "subset_n")),
    Step(id="apply_reference_split", fn=_step_apply_reference_split,
         produces=frozenset({"matrices"}),
         requires=frozenset({"subarrays"}),
         params_space=("reference_split", "run_dir")),
    Step(id="fit_project", fn=_step_fit_project,
         produces=frozenset({"embeddings"}),
         requires=frozenset({"matrices"}),
         params_space=("method", "shared_dim", "reg")),
    Step(id="evaluate_and_save", fn=_step_evaluate_and_save,
         produces=frozenset({"metrics"}),
         requires=frozenset({"embeddings"}),
         params_space=("run_dir", "ks", "negative_ratio", "negative_seed")),
]


def _train_classical_chain(
    run_cfg: dict, niches_dir: str, ref_split_path: Path, run_dir: Path,
) -> dict:
    """invoke the four-step chain for one classical baseline.

    returns the records dict from build_protocol - one entry per step id,
    each with status, duration_s, params, observed.
    """
    chain = build_protocol(CLASSICAL_BASELINE_STEPS, "classical_baseline")

    method = run_cfg["method"]
    fit_params: dict = {
        "method": method,
        "shared_dim": run_cfg.get("shared_dim", 512),
    }
    if "reg" in run_cfg:
        fit_params["reg"] = run_cfg["reg"]

    ctx_params = {
        "load_data": {
            "niches_dir": niches_dir,
            "st_features": tuple(run_cfg.get(
                "st_features", ("novae_niche", "gpath2vec_niche"))),
        },
        "apply_reference_split": {
            "reference_split": str(ref_split_path),
            "run_dir": str(run_dir),
        },
        "fit_project": fit_params,
        "evaluate_and_save": {
            "run_dir": str(run_dir),
            "ks": [1, 5, 10],
            "negative_ratio": 10,
            "negative_seed": 0,
        },
    }

    gpath_sha = None
    manifest_path = Path(niches_dir) / "manifest.json"
    if manifest_path.is_file():
        _man = json.loads(manifest_path.read_text())
        gpath_sha = _man.get("sources", {}).get("gpath2vec_sha256")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(
        {**run_cfg, "reference_split": str(ref_split_path),
         "gpath2vec_parquet_sha256": gpath_sha,
         "backend": "package", "params": ctx_params}, indent=2))

    records = chain.invoke({"params": ctx_params, "run_dir": run_dir})

    metrics_path = run_dir / "metrics_h1_raw.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        k_used = records.get("fit_project", {}).get("observed", {}).get("k_used")
        if k_used is not None:
            metrics["shared_dim_used"] = k_used
        metrics["baseline"] = method
        metrics_path.write_text(json.dumps(metrics, indent=2))

    return records


# --- the two COMPUTE-OR-RESOLVE stages --------------------------------------
def run_align(cfg: ProjectConfig, nj: NicheJoin, run_ids: list[str] | None = None,
              project_id: str | None = None, *, compute: bool = False,
              runs_root: Path | None = None,
              use_package: bool = True) -> tuple[AlignGrid, TransformRecord]:
    """resolve the grid, or train the runs that are missing.

    dispatch is configuration-driven: each run's config declares its method
    (contrastive, cca, procrustes, unaligned, random_init). the method
    determines the compute path; the run ID is an experiment identifier only.

    use_package=True routes contrastive runs through the five-step LCEL chain
    and classical baselines through the four-step classical chain. both are
    composed via build_protocol. use_package=False shells out to scripts/.
    random_init always dispatches to script.
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
    chain_records: dict[str, dict] = {}
    for rid in missing:
        run_cfg = _resolve_run_config(rid)
        method = run_cfg["method"]

        if method == "contrastive":
            if use_package:
                recs = _train_contrastive_chain(
                    run_cfg, nj.niches_dir, root / rid)
                chain_records[rid] = recs
            else:
                cfg_path = settings.resolve(Path("configs/v3")) / f"{rid}.json"
                _run_script("align.py", ["--config", str(cfg_path),
                                         "--niches-dir", nj.niches_dir,
                                         "--runs-root", str(root)])

        elif method in _CLASSICAL_METHODS:
            ref_run = run_cfg.get("reference_run",
                                  cfg.contrastive_runs[0] if cfg.contrastive_runs else "R1_v3")
            ref = root / ref_run / "split.json"
            if use_package:
                recs = _train_classical_chain(
                    run_cfg, nj.niches_dir, ref, root / rid)
                chain_records[rid] = recs
            else:
                common = ["--reference-split", str(ref), "--run-id", rid,
                          "--niches-dir", nj.niches_dir, "--runs-root", str(root)]
                _run_script("align_classical.py",
                            ["--baseline", method, *common])

        elif method == "random_init":
            ref_run = run_cfg.get("reference_run",
                                  cfg.contrastive_runs[0] if cfg.contrastive_runs else "R1_v3")
            ref = root / ref_run / "split.json"
            common = ["--reference-split", str(ref), "--run-id", rid,
                      "--niches-dir", nj.niches_dir, "--runs-root", str(root)]
            _run_script("align_b4.py", common)

        else:
            raise ValueError(
                f"unknown method {method!r} in config for {rid!r}. "
                "expected: contrastive, cca, procrustes, unaligned, "
                "or random_init.")

        computed.append(rid)

    refs = {r: str(root / r / "embeddings_test.parquet")
            for r in ids if (root / r / "embeddings_test.parquet").is_file()}
    rec = _rec("align", "pass", resolves_only=not computed,
               n_runs=len(refs), computed=computed,
               method="package" if use_package else "script")
    if chain_records:
        rec.params["chain_records"] = chain_records
    return AlignGrid(runs_root=str(root), run_refs=refs, computed=computed), rec


_CLASSICAL_METHODS = frozenset({"cca", "procrustes", "unaligned"})

# legacy fallback: resolve method from run ID prefix when no config exists.
# new runs should have a config with an explicit "method" field.
_LEGACY_METHOD = {"B1": "cca", "B2": "procrustes", "B3": "unaligned", "B4": "random_init"}


def _resolve_run_config(rid: str) -> dict:
    """load a run's config and ensure it declares a method.

    backward compatibility:
        - config with "method" -> used as-is
        - config with "loss" but no "method" -> "contrastive" (legacy R configs)
        - no config file -> legacy map from run ID prefix
    """
    cfg_path = settings.resolve(Path("configs/v3")) / f"{rid}.json"
    if cfg_path.is_file():
        run_cfg = json.loads(cfg_path.read_text())
        if "method" not in run_cfg:
            if "loss" in run_cfg:
                run_cfg["method"] = "contrastive"
            else:
                raise ValueError(
                    f"config for {rid} has no 'method' and no 'loss' field - "
                    "cannot infer the alignment method. add an explicit "
                    "'method' key (contrastive, cca, procrustes, unaligned, "
                    "or random_init).")
        return run_cfg

    prefix = rid.split("_")[0]
    if prefix in _LEGACY_METHOD:
        return {"run_id": rid, "method": _LEGACY_METHOD[prefix]}

    raise ValueError(
        f"no config for {rid!r} and no legacy mapping for prefix {prefix!r}. "
        "create configs/v3/{rid}.json with an explicit 'method' field.")

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

# --- the align gates: admissibility before a grid ---------------------------
class EdaGateMissing(ComputeUnavailable):
    """no eda_summary.json, so there is no verdict to accept.

    a refusal and not a gate. a gate is a question with a closed option set and
    someone to ask; "the gate has never been run" offers neither, and asking a
    person to accept a verdict nobody computed is how a cohort gets a pass it
    never earned.
    """


def _declarations(project_id: str | None) -> dict:
    """project.json over cohort.json, merged.

    both are the cohort's, and which field lives in which is a layout decision
    rather than a semantic one - the grid is in project.json, the gate answers in
    cohort.json. a caller asking "what did this cohort declare" should not have
    to know that, so the pointer in the contract names the file and this reads
    both.
    """
    import json as _j

    out: dict = {}
    # cohort.json FIRST so project.json wins a collision, which is what the
    # sentence above says. merged the other way round, cohort.json won silently.
    for name in ("cohort.json", "project.json"):
        f = settings.project_root(project_id) / name
        if not f.is_file():
            continue
        try:
            out |= _j.loads(f.read_text())
        except ValueError as e:
            # a declaration that will not parse is not an absent declaration.
            # swallowing it meant a typo in cohort.json reopened every gate as
            # though nothing had been declared.
            raise ComputeUnavailable(
                f"{name} does not parse: {e}. a malformed declaration is not an "
                f"absent one - fix it rather than running as if undeclared.") from e
    return out


def preflight_align(cohort: dict | None = None, project_id: str | None = None) -> list:
    """the gates that fire before a grid, resolved as far as the cohort allows.

    the first is this repo's opening constraint, which was a convention until
    now: no alignment, retrieval or evaluation until the EDA gate is satisfied.
    `eda_admissibility` is that sentence as a gate.

    deterministic and side-effect free, like `encode.preflight`: it reads the
    cohort's recorded summary through `run_gate` and measures nothing itself.
    """
    from omicstra.eda import run_gate
    from omicstra.protocols.encode import GateRequest, _resolve_option, contract

    cohort = _declarations(project_id) if cohort is None else {
        **_declarations(project_id), **cohort}
    try:
        r = run_gate(project_id)
    except FileNotFoundError as e:
        # FileNotFoundError only. catching KeyError here reported any lookup bug
        # inside run_gate as "the gate was never run", which is a different and
        # much more comfortable statement than the truth.
        raise EdaGateMissing(
            f"no EDA verdict for this cohort, so there is nothing to accept and no "
            f"option list to offer: {e}. run the gate first - the constraint is that "
            f"no alignment, retrieval or evaluation happens until it is satisfied, and "
            f"an unrun gate is not a satisfied one.") from e                     # raises if there is no summary
    gates = {g["id"]: g for g in contract()["gates"] if g.get("stage") == "align"}
    out = []
    for gid in ("eda_admissibility", "verdict_disagreement", "grid_scope"):
        g, answer, observed = gates[gid], None, {}

        if gid == "eda_admissibility":
            observed = {"verdict": r.verdict, "declared_verdict": r.declared_verdict,
                        "n_violations": len(r.violations),
                        "n_cautions": len(r.cautions),
                        "n_escalations": len(r.escalations),
                        "cautions": list(r.cautions)[:6]}
            if r.verdict == "proceed":
                answer = "accept_cautions"          # nothing to accept; it passed
            elif r.declared_verdict == "proceed":
                # `graphs/eda.py` already interrupts on a cautionary verdict, so
                # asking again here would have a person accept the same cautions
                # twice, in two records. a person wrote `proceed` into the
                # committed summary and version control attests the file - that
                # IS the recorded acceptance, the same rule the actor vocabulary
                # uses. `verdict_disagreement` below still fires, because the
                # acceptance being on record does not make the two verdicts agree.
                answer = "accept_cautions"
        elif gid == "verdict_disagreement":
            observed = {"computed": r.verdict, "declared": r.declared_verdict,
                        "agrees": r.agrees_with_declared}
            if r.agrees_with_declared:
                answer = "computed"                # they are the same value
        else:
            grid = list(cohort.get("contrastive_runs") or []) + \
                   list(cohort.get("classical_runs") or [])
            observed = {"declared_grid": grid, "n_arms": len(grid)}
            if grid:
                answer = "declared_grid"

        src = None
        if answer is not None:
            src = ("the verdict itself" if gid != "eda_admissibility" or
                   r.verdict == "proceed" else "eda_summary.json#verdict")
        declared = _declared_align(cohort, g.get("pre_answerable_by"))
        if declared is not None:
            answer = _resolve_option(declared, tuple(g["options"]), gid,
                                     g.get("answer_kind", "choice"))
            src = g.get("pre_answerable_by")

        out.append(GateRequest(
            id=gid, when=g["when"], question=g["question"],
            options=tuple(g["options"]), forecloses=g["forecloses"],
            pre_answerable_by=g.get("pre_answerable_by"),
            answer=answer, source=src, observed=observed or None))
    return out


def _declared_align(cohort: dict, pointer: str | None):
    """the cohort's own answer, if it declared one. `cohort.json#field`."""
    if not pointer or "#" not in pointer:
        return None
    return cohort.get(pointer.split("#", 1)[1]) or None
