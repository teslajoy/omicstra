"""modality-specialised agents - one per representation.

each agent owns exactly one encoder and answers one question: what does MY
modality contribute to THIS biological question, on this cohort? it reports the
encoder's provenance, the modality-only baseline where the cohort declares one,
and the guard it is responsible for. it does not align, evaluate, or route -
those belong to the judge (CLAUDE.md agent constraints).

read path: every embedding is already computed and cached. an agent resolves
references and reads recorded evidence; none of them touch a GPU.

the emission is a DiagnosticRecord, so a modality report is the same scientific
unit as an EDA step: method, scope, observed, criterion, result, decision.
"""
from __future__ import annotations

from omicstra.contracts.project import ProjectConfig
from omicstra.records import DiagnosticRecord
from omicstra.routing import _blocks, load_routing_evidence


def _encoder(cfg: ProjectConfig, name: str) -> dict:
    e = next((e for e in cfg.encoders if e.name == name), None)
    return e.model_dump() if e else {}


def _modality_baseline(task: dict, bar_id: str) -> dict | None:
    """the modality-alone value this cohort recorded for the task, if any.

    a floor or reference named for a raw modality IS that modality's solo
    performance - the agent reads its own baseline out of the shared evidence
    rather than keeping a second copy that could drift.
    """
    for blk in _blocks(task):
        for key in ("floor", "reference"):
            bar = blk.get(key)
            if bar and bar.get("id") == bar_id:
                # spread first, then set - the evidence file may itself carry a
                # `role` note, and it must not be shadowed by, or shadow, ours
                return {**bar, "metric": blk.get("metric", ""), "declared_as": key}
    return None


def _separable(a: dict, b: dict) -> bool | None:
    """do two candidates' intervals separate them? None when either has no CI.

    the same rule the router applies. an ablation that sits inside the leader's
    interval has not been shown to matter, however far apart the point estimates
    look - and on a task the router already called a tie, claiming otherwise
    would contradict the route from inside its own evidence.
    """
    ca, cb = a.get("ci"), b.get("ci")
    if not ca or not cb:
        return None
    return not (ca[0] <= cb[1] and cb[0] <= ca[1])


def _report(step_id: str, method: str, scope: str, observed: dict, criterion: str,
            result: str, decision: str, caveats: list[str]) -> DiagnosticRecord:
    return DiagnosticRecord(step_id=step_id, method=method, scope=scope,
                            observed=observed, criterion=criterion, result=result,
                            decision=decision, caveats=caveats, actor="deterministic")


def _n_test(ev: dict) -> str:
    return str((ev.get("eval_scope") or {}).get("test_niches", "?"))


# --- H&E -------------------------------------------------------------------
def he_agent(task_id: str, project_id: str | None = None) -> DiagnosticRecord:
    """morphology. owns Virchow2 and the raw-H&E baseline."""
    cfg, ev = ProjectConfig.load(project_id), load_routing_evidence(project_id)
    enc = _encoder(cfg, "virchow2")
    task = (ev.get("tasks") or {}).get(task_id, {})
    base = _modality_baseline(task, "raw_he")

    obs = {"encoder": enc.get("name"), "dim": enc.get("dim"),
           "stain_normalised": False,
           "tissue_fit": "validated - MSK breast cases in training",
           "modality_alone": base}
    if base:
        res = (f"raw H&E alone scores {base['value']} on this task, declared as the "
               f"{base['declared_as']}; alignment is measured against that")
        dec = ("morphology carries a standalone signal here - any aligned method "
               "must be reported against it")
    else:
        res = "this cohort declares no H&E-only baseline for this task"
        dec = "cannot say what morphology contributes alone - report the aligned result only"
    return _report(
        f"he_agent::{task_id}",
        "Virchow2 niche embedding, frozen foundation model, mean-pooled over the niche's tiles",
        f"{_n_test(ev)} held-out niches, z_he view", obs,
        "a modality-only baseline is reported wherever the cohort declares one",
        res, dec,
        ["Virchow2 is stain-invariant by construction - no Reinhard/Macenko and no "
         "z-score before the foundation model (program.md hard constraint 2)"])


# --- spatial transcriptomics ------------------------------------------------
def st_agent(task_id: str, project_id: str | None = None) -> DiagnosticRecord:
    """molecular expression. owns Novae and the raw-ST baseline."""
    cfg, ev = ProjectConfig.load(project_id), load_routing_evidence(project_id)
    enc = _encoder(cfg, "novae")
    task = (ev.get("tasks") or {}).get(task_id, {})
    base = _modality_baseline(task, "raw_st")

    obs = {"encoder": enc.get("name"), "dim": enc.get("dim"),
           "normalisation": "z-scored per subarray AFTER niche aggregation",
           "tissue_fit": "validated_with_constraints - trained on subcellular "
                         "image-based ST, applied to 100um spots",
           "modality_alone": base}
    if base:
        res = (f"raw ST alone scores {base['value']} on this task, declared as the "
               f"{base['declared_as']}")
        dec = ("expression alone is the weaker modality here - the lift over it is "
               "the alignment claim")
    else:
        res = "this cohort declares no ST-only baseline for this task"
        dec = "cannot say what expression contributes alone on this task"
    return _report(
        f"st_agent::{task_id}",
        "Novae GNN niche latent, frozen, mean-pooled over the niche then z-scored per subarray",
        f"{_n_test(ev)} held-out niches, z_st view", obs,
        "a modality-only baseline is reported wherever the cohort declares one",
        res, dec,
        ["mc_weights are supervision only and never enter the ST feature vector "
         "(program.md hard constraint 3)",
         "Novae zero-shot is undefined below its prototype count - subarrays under "
         "512 spots are excluded"])


# --- pathway ----------------------------------------------------------------
def pathway_agent(task_id: str, project_id: str | None = None) -> DiagnosticRecord:
    """pathway context. owns gpath2vec, its sha-lock, and the ablation.

    the only agent whose encoder reads knowledge from outside the measurement:
    Virchow2 and Novae encode the assay, gpath2vec additionally reads a curated
    Reactome hierarchy. that is why it needs an ablation to be load-bearing.
    """
    cfg, ev = ProjectConfig.load(project_id), load_routing_evidence(project_id)
    enc = _encoder(cfg, "gpath2vec")
    task = (ev.get("tasks") or {}).get(task_id, {})

    # an ablation is only interpretable against a MATCHED control - a run
    # differing in the ablated component and nothing else. comparing it to the
    # grid leader attributes to one change an effect that two changes produced.
    spec = (ev.get("ablations") or {}).get("gpath2vec", {})
    abl_id = spec.get("run", "R6_v3")
    ctl_id = spec.get("matched_control")
    abl = ctl = None
    metric, sep = "", None
    for blk in _blocks(task):
        by_id = {c["id"]: c for c in blk.get("candidates", [])}
        if abl_id in by_id and ctl_id in by_id:
            abl, ctl = by_id[abl_id], by_id[ctl_id]
            sep = _separable(abl, ctl)
            metric = blk.get("metric", "")
            break

    obs = {"encoder": enc.get("name"), "dim": enc.get("dim"),
           "sha256": cfg.gpath2vec_sha256,
           "reads_external_knowledge": "Reactome hierarchy, TF-restricted Fisher enrichment",
           "ablation_run": abl_id, "matched_control": ctl_id,
           "differs_only_in": spec.get("differs_only_in", ""),
           "ablation": abl, "control": ctl, "intervals_separate": sep}
    if abl is None or ctl is None:
        res = ("this cohort does not score the ablation and its matched control on "
               "this task")
        dec = "cannot say whether the pathway anchor is load-bearing here"
    else:
        d = ctl["value"] - abl["value"]
        res = (f"matched ablation on {metric[:52]}: removing gpath2vec moves "
               f"{ctl['value']} to {abl['value']}, a difference of {d:+g}")
        if sep is False:
            dec = ("NOT shown load-bearing here - the ablation's interval overlaps its "
                   "matched control's, so removing the pathway anchor is not measurably "
                   "worse on this task")
        elif sep is None:
            dec = (f"the matched control is {d:+g} above the ablation, but neither "
                   "carries an interval - report the gap, do not call it separation"
                   if d else
                   "the ablation matches its control - the anchor is not load-bearing here")
        else:
            dec = ("load-bearing for this task - the ablation's interval sits clear of "
                   "its matched control's")
    return _report(
        f"pathway_agent::{task_id}",
        "gpath2vec niche embedding - metapath2vec over a niche-pathway graph built "
        "from TF-restricted Fisher enrichment against Reactome",
        f"{_n_test(ev)} held-out niches", obs,
        "the pathway anchor must be shown load-bearing by ablation, not assumed",
        res, dec,
        ["sha-locked to the v3 build - a silent swap invalidates every H3 result",
         "TF restriction under-represents non-TF gene families (collagen, "
         "immunoglobulin); this is why TLS signal degrades under late fusion",
         "z_st and z_mean are circular when gpath2vec is on the ST input - z_he is "
         "the only clean view for H3",
         (spec.get("untested_run") or "")])


MODALITY_AGENTS = {"he_agent": he_agent, "st_agent": st_agent,
                   "pathway_agent": pathway_agent}
