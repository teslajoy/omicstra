"""omicstra MCP server - read path, zero compute.

first component: the EDA gate. a cohort is admissible only if its eda_summary.json
satisfies configs/eda_contract.json, and this is the tool that says so.

no data access, no R, no GPU, no network. reads two committed JSON files.

the cohort is selected at launch via OMICSTRA_PROJECT_DIR (or the in-repo
registry default). that directory is the security boundary: a tool call passing
a different project_id cannot reach outside it.

note: mcp 2.0 renamed FastMCP -> MCPServer (mcp.server.fastmcp no longer exists).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import LATEST_PROTOCOL_VERSION

from omicstra.contracts.project import ProjectConfig
from omicstra.eda import load_contract, load_summary, run_gate
from omicstra.settings import settings

srv = MCPServer("omicstra")


# --- tools: model-invoked, do work ----------------------------------------
@srv.tool(description=(
    "Run the EDA admissibility gate for a cohort. Returns a verdict "
    "(proceed | proceed_with_caution | stop), per-check results, blocking "
    "violations, cautions, and advisories for checks the contract expects but "
    "the cohort summary cannot answer. Every check carries an authority: "
    "universal criteria are applied silently, cohort_calibrated criteria were "
    "derived on some cohort and escalate rather than being inherited by a new "
    "one, and advisory checks are reported but never change a verdict. Reads "
    "committed artifacts only."))
def check_eda_gate(project_id: str | None = None) -> dict:
    r = run_gate(project_id)
    return {
        "project_id": r.project_id,
        "verdict": r.verdict,
        "declared_verdict": r.declared_verdict,
        "agrees_with_declared": r.agrees_with_declared,
        "summary": {
            "passed": sum(c.status == "pass" for c in r.checks),
            "failed": sum(c.status == "fail" for c in r.checks),
            "not_run": sum(c.status == "not_run" for c in r.checks),
            "not_applicable": sum(c.status == "not_applicable" for c in r.checks),
            "escalated": sum(c.status == "escalate" for c in r.checks),
        },
        "checks": [c.model_dump() for c in r.checks],
        "violations": r.violations,
        "cautions": r.cautions,
        "advisories": r.advisories,
        "escalations": r.escalations,
    }


@srv.tool(description=(
    "Describe a cohort's data structure: platform, encoders and their "
    "dimensions, atomic unit, split protocol, and the supervision/label roles "
    "that must never enter a feature vector."))
def describe_data_structure(project_id: str | None = None) -> dict:
    cfg = ProjectConfig.load(project_id)
    s = load_summary(project_id)
    return {
        "project_id": cfg.project_id,
        "project_dir": str(cfg.project_dir),
        "platform": cfg.platform,
        "atomic_unit": f"niche = centre spot + {cfg.k_neighbors} neighbours",
        "split": cfg.split,
        "seed": cfg.seed,
        "encoders": [e.model_dump() for e in cfg.encoders],
        "supervision_only": cfg.supervision,
        "h2_label": cfg.h2_label,
        "headline_view": cfg.headline_view,
        "counts_are_raw": s.get("counts_are_raw"),
        "encoder_input_decision": s.get("encoder_input_decision"),
        "gpath2vec_sha256": cfg.gpath2vec_sha256,
    }


# --- routing: which method can answer which biological question -----------
@srv.tool(description=(
    "List the biological question types this cohort's evaluation can route. Call "
    "this FIRST when asked what the system can answer, or before routing, to pick "
    "the task_id. Each family states the level of biological abstraction it lives "
    "at - measurement, structure, cohort or program - because the same words can "
    "sit at two levels and they route to different methods. If a question could "
    "belong to two families, ask the user which they mean rather than guessing; "
    "this server will not guess for you."))
def list_task_families(project_id: str | None = None) -> dict:
    from omicstra.routing import list_task_families as _list
    return _list(project_id=project_id)


@srv.tool(description=(
    "Route one biological question to the method this cohort's evidence supports, "
    "and say why. You choose the task_id from list_task_families; everything after "
    "that is resolved by rule from recorded measurements - no judgement of yours "
    "enters it.\n\n"
    "Returns a selection record whose `resolution` field is one of five, and they "
    "are not interchangeable: `recommend` - one method leads on every declared "
    "axis; `tie` - the measurement does not order the candidates, so the system "
    "declines to pick; `escalate` - the rules cannot resolve it at all, because "
    "the declared axes disagree or this cohort has no evidence; `refuse` - the "
    "evidence contraindicates the method the user named; `override_ack` - the "
    "user was refused and chose to proceed, recorded as dissent.\n\n"
    "A tie is a statement about the measurement; an escalation is a statement "
    "about the rules. Report them differently.\n\n"
    "WHEN REPORTING THIS TO THE USER: state the scope sentence from `caveats` "
    "BEFORE the recommendation, and reproduce `why` in full including any "
    "shortfall it opens with. The `why` field sometimes leads with a comparison "
    "against the raw un-aligned modality - that sentence is the finding, not a "
    "hedge, and must not be moved to the end or summarised away. Report a tie as "
    "a tie; do not pick a winner the evidence declined to pick.\n\n"
    "Set proposed_method when the user names a method themselves. Set "
    "override_refusal only when the user has seen a refusal and asked to proceed "
    "anyway - that is recorded as dissent against the evidence."))
def route(task_id: str, question: str = "", proposed_method: str | None = None,
          override_refusal: bool = False, project_id: str | None = None) -> dict:
    from omicstra.routing import resolve
    r = resolve(task_id, question=question, proposed_method=proposed_method,
                override=override_refusal, project_id=project_id)
    d = r.model_dump()
    d["resolution"] = ("override_ack" if (r.contraindicated and r.actor == "human")
                       else "refuse" if r.contraindicated
                       else "escalate" if r.escalated
                       else "tie" if r.tie else "recommend")
    d["headline"] = r.headline()
    return d


@srv.tool(description=(
    "Summarise the decision ledger for this cohort: how many routing decisions "
    "were taken, how they resolved, and what each rested on. Use when asked what "
    "the system has decided, how reproducible it is, or how much was rule-resolved "
    "versus escalated to a human. Note the boundary it reports: mapping a question "
    "onto a task_id happens in the calling model and is NOT counted here - only "
    "task_id to route is."))
def decision_record(project_id: str | None = None) -> dict:
    from omicstra.routing import decision_record as _rec
    return _rec(project_id=project_id)


@srv.tool(description=(
    "Show the evidence behind a routing decision as a chart. Use when the user "
    "asks to see the evidence, the numbers, the intervals, or why a task tied or "
    "escalated. Renders the candidates for one task_id with their confidence "
    "intervals, the floor and reference drawn as lines, and any contraindicated "
    "method greyed out and labelled.\n\n"
    "The chart is drawn from the same evidence file the route resolves against, "
    "so it cannot disagree with what `route` returned. A task with two axes gets "
    "two panels - that is what makes an escalation visible: you can see the axes "
    "name different leaders."))
def plot_evidence(task_id: str, project_id: str | None = None) -> list:
    import base64

    from mcp.types import ImageContent, TextContent

    from omicstra.figures import evidence_plot
    try:
        path, caption = evidence_plot(task_id, project_id=project_id)
    except KeyError as e:
        return [TextContent(type="text", text=str(e))]
    return [ImageContent(type="image", mimeType="image/png",
                         data=base64.b64encode(path.read_bytes()).decode()),
            TextContent(type="text", text=caption)]


# --- compute-lite: run a real EDA step on cached data ---------------------
@srv.tool(description=(
    "Run a real exploratory-data-analysis step on this cohort's actual measurements "
    "and return the scientific finding: the method used, the scope it ran on, what "
    "was observed, the criterion it was judged against, the result, and the decision "
    "that follows. Steps: count_statistics (depth, detection, and whether values are "
    "genuinely integer counts), marker_expression (are expected markers present and "
    "unexpected ones absent), spatial_autocorrelation (is expression spatially "
    "structured - this routes the molecular encoder), batch_structure (does technical "
    "origin explain more variation than biology). This computes; it does not read a "
    "stored verdict."))
def run_eda_step(step: str, n_perm: int = 199) -> dict:
    import anndata as ad

    from omicstra import measures as st
    from omicstra.eda import load_summary

    cache = settings.resolve(settings.data_dir) / "embeddings/_eda_cache/cohort_sample.h5ad"
    if not cache.exists():
        return {"error": f"no cached cohort at {cache}. build it with the adapter first."}
    a = ad.read_h5ad(cache)
    s = load_summary()

    if step == "count_statistics":
        r = st.count_statistics(a, sample_key="sample")
    elif step == "marker_expression":
        r = st.marker_expression(
            a,
            positive={g: v["expected"] for g, v in s.get("positive_markers", {}).items()},
            negative={g: v["expected"] for g, v in s.get("negative_markers", {}).items()})
    elif step == "spatial_autocorrelation":
        genes = list(s.get("morans_i_summary", {}).get("results", {}))
        r = st.spatial_autocorrelation(a, markers=genes, sample_key="sample",
                                       n_perm=int(n_perm))
    elif step == "batch_structure":
        meta = {smp: {"slide": smp.split("_")[0]} for smp in a.obs["sample"].unique()}
        r = st.batch_structure(a, sample_key="sample", sample_meta=meta,
                               technical=["slide"], biological=[])
    else:
        return {"error": f"unknown step {step!r}",
                "available": ["count_statistics", "marker_expression",
                              "spatial_autocorrelation", "batch_structure"]}

    d = r.model_dump()
    d["headline"] = r.headline()

    # the cache is a convenience subset, not the cohort. a subset must not be
    # allowed to state a cohort-level decision - that is how a routing choice
    # ends up resting on unrepresentative data.
    n_samples = int(a.obs["sample"].nunique())
    n_subjects = len({s.split("_")[0] for s in a.obs["sample"].unique()})
    total = s.get("samples_with_counts") or s.get("samples")
    d["scope_warning"] = (
        f"computed on a cached subset: {n_samples} samples from {n_subjects} "
        f"subject group(s)" + (f", of {total} in the cohort" if total else "")
        + ". Not representative. Treat `decision` as illustrative of the method, "
          "not as a cohort-level finding."
    )
    d["decision_authority"] = "subset - illustrative only"
    return d


# --- a picture of the caveat ----------------------------------------------
@srv.tool(description=(
    "Plot Moran's I spatial autocorrelation for this cohort's marker genes, comparing "
    "the single subarray the recorded EDA measured against the median across every "
    "sample in the cohort. Use this when asked how well a spatial-signal result "
    "generalises, whether a check rested on one sample, or to see the evidence behind "
    "the spatial_autocorrelation gate check. Returns a chart plus the counts above "
    "threshold for each scope."))
def plot_spatial_autocorrelation(project_id: str | None = None) -> list:
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mcp.types import ImageContent, TextContent

    s = load_summary(project_id)
    single = (s.get("morans_i_summary") or {})
    recorded = single.get("results") or {}
    cache = (settings.resolve(settings.data_dir)
             / "embeddings/_eda_cache/spatial_autocorrelation_cohort.json")
    if not recorded:
        return [TextContent(type="text", text="no morans_i_summary recorded for this cohort")]
    cohort = json.loads(cache.read_text())["observed"] if cache.exists() else {}

    thr = 0.3
    genes = sorted(recorded, key=lambda g: -(cohort.get(g, {}).get("median_I")
                                             if cohort.get(g) else recorded[g]["I"]))
    sns.set_theme(style="ticks", palette="Set2", context="notebook")
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(genes) + 1.5))
    y = range(len(genes))

    for i, g in enumerate(genes):
        c = cohort.get(g)
        if c:
            ax.plot([c["q25"], c["q75"]], [i, i], lw=7, color="#6FA3B8", alpha=.45,
                    solid_capstyle="butt", zorder=1)
            ax.plot(c["median_I"], i, "o", ms=8, color="#4A5D7E", zorder=3,
                    label="cohort median (281 samples)" if i == 0 else None)
        ax.plot(recorded[g]["I"], i, "D", ms=7, color="#C97B84", zorder=4,
                label=f"{single.get('subarray', 'single subarray')} only (n=1)" if i == 0 else None)

    ax.axvline(thr, color="#B5544F", ls="--", lw=1.4, zorder=2)
    ax.text(thr, len(genes) - .3, f"  threshold {thr}", color="#B5544F", fontsize=9, va="top")
    ax.set_yticks(list(y)); ax.set_yticklabels(genes)
    ax.set_xlabel("Moran's I"); ax.invert_yaxis()
    ax.set_title("the check that gates the molecular encoder\n"
                 "recorded on one subarray, recomputed across the cohort", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    sns.despine(ax=ax)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); plt.close(fig)

    # the criterion is "top 5 markers" ranked WITHIN a scope, so each scope ranks
    # by its own values. ranking both by the cohort would silently restate the
    # recorded result on a different marker set.
    top_single = sorted(recorded, key=lambda g: -recorded[g]["I"])[:5]
    n_single = sum(1 for g in top_single if recorded[g]["I"] > thr)
    msg = (f"Top 5 markers above I > {thr}: "
           f"{n_single} of 5 on {single.get('subarray', 'the recorded subarray')} alone")
    if cohort:
        top_cohort = sorted(cohort, key=lambda g: -cohort[g]["median_I"])[:5]
        n_cohort = sum(1 for g in top_cohort if cohort[g]["median_I"] > thr)
        msg += f", {n_cohort} of 5 across all {cohort[top_cohort[0]]['n_samples']} samples."
    else:
        n_cohort = None
        msg += ". No cohort-wide recomputation is cached."
    if cohort and n_cohort is not None and n_cohort < n_single:
        msg += (" The recorded value rests on one sample and does not hold cohort-wide - "
                "the criterion behind it is calibrated, not universal.")
    return [ImageContent(type="image", data=base64.b64encode(buf.getvalue()).decode(),
                         mimeType="image/png"),
            TextContent(type="text", text=msg)]


# --- resources: URI-addressed data, not model-invoked ---------------------
@srv.resource("omicstra://eda-contract", mime_type="application/json",
              description="the generalizable EDA gate contract - thresholds and "
                          "verdict logic, no cohort measurements")
def eda_contract() -> str:
    return json.dumps(load_contract(), indent=2)


@srv.resource("omicstra://routing-contract", mime_type="application/json",
              description="the generalizable routing contract - task taxonomy, "
                          "outcome vocabulary and decision rules, no cohort numbers")
def routing_contract() -> str:
    from omicstra.routing import load_routing_contract
    return json.dumps(load_routing_contract(), indent=2)


@srv.resource("omicstra://project/{project_id}/routing-evidence",
              mime_type="application/json",
              description="the selected cohort's measured routing evidence - the "
                          "numbers a route resolves against")
def routing_evidence(project_id: str) -> str:
    from omicstra.routing import load_routing_evidence
    return json.dumps(load_routing_evidence(project_id), indent=2)


@srv.resource("omicstra://project/{project_id}/eda-summary",
              mime_type="application/json",
              description="the selected cohort's measured EDA summary")
def eda_summary(project_id: str) -> str:
    # project_id is advisory; the boundary is settings.project_dir
    return json.dumps(load_summary(project_id), indent=2)


# --- transport --------------------------------------------------------------
# stdio and http are the SAME server. the transport is a deployment choice, not
# an architectural one - no tool, contract or record changes between them.
#
# on the protocol revision: this SDK's LATEST_PROTOCOL_VERSION is 2026-07-28,
# which is NOT backward compatible with 2025-03-26 - a client on one cannot
# talk to a server on the other. the SDK negotiates and will fall back, so the
# default here is permissive. set REQUIRE_PROTOCOL_2026=1 to refuse anything
# older, which is what a deployment behind an authorization server wants: the
# july revision is where mcp servers became oauth 2.1 resource servers, and
# accepting an older client silently means accepting one that cannot present
# an audience-bound token.
REQUIRE_PROTOCOL_2026 = os.environ.get("REQUIRE_PROTOCOL_2026", "0") == "1"


def negotiated_protocol_version(meta: dict | None) -> str | None:
    """the version the client declared, read off `_meta`.

    the july revision removed the initialize handshake: version and capabilities
    travel in `_meta` on every request instead. so this is per-request, not
    per-session, and it belongs on the record - a run is only reproducible if
    you know which protocol produced it.
    """
    from mcp.types import PROTOCOL_VERSION_META_KEY
    return (meta or {}).get(PROTOCOL_VERSION_META_KEY)


def request_protocol_version() -> str | None:
    """the version of the CURRENT request, validated.

    called from the tool path rather than a handshake, because the july
    revision removed the handshake. every tool that stamps a record calls this,
    so strict mode refuses at the door AND the version lands on the record.
    """
    try:
        from mcp.server import context
        ctx = context.request_ctx.get()
        meta = getattr(ctx, "meta", None)
        meta = meta if isinstance(meta, dict) else (meta.model_dump() if meta else None)
    except Exception:
        meta = None                       # stdio, or outside a request
    return check_protocol(meta)


def check_protocol(meta: dict | None) -> str | None:
    v = negotiated_protocol_version(meta)
    if REQUIRE_PROTOCOL_2026 and v != LATEST_PROTOCOL_VERSION:
        raise ValueError(
            f"REQUIRE_PROTOCOL_2026=1 and this request declares {v!r}. the "
            f"{LATEST_PROTOCOL_VERSION} revision is required: older clients "
            "cannot present audience-bound tokens.")
    return v


def serve(mode: str = "stdio", mount: str = "/mcp", host: str = "127.0.0.1",
          port: int = 8000) -> None:
    """the one entry point. `mode` is the only thing that differs.

    http is STATELESS by design - any instance handles any request, so n
    servers sit behind one address with no session affinity. omicstra's own
    handle is `run_id`, minted by a tool and passed back as an ordinary
    parameter, so nothing needs a sticky session to resume.
    """
    if mode == "stdio":
        srv.run()
        return
    if mode != "http":
        raise ValueError(f"mode must be stdio or http, not {mode!r}")
    import uvicorn
    app = srv.streamable_http_app(streamable_http_path=mount, stateless_http=True,
                                  host=host)
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    serve(mode=os.environ.get("OMICSTRA_TRANSPORT", "stdio"),
          mount=os.environ.get("OMICSTRA_HTTP_MOUNT", "/mcp"),
          host=os.environ.get("OMICSTRA_HTTP_HOST", "127.0.0.1"),
          port=int(os.environ.get("OMICSTRA_HTTP_PORT", "8000")))


if __name__ == "__main__":
    main()

def _unit_counts(project_id: str | None) -> tuple[dict[str, int], dict]:
    """per-sample unit counts, and an honest statement of what they cover.

    the inventory is the declared source, and it records only the samples it was
    run over - on a partial run that is a handful. the canonical ingest record
    covers every converted sample, so it is preferred when it reaches further,
    and the returned coverage says which was used and how many units it saw.

    this matters because the platform_floor gate decides from these. reading two
    samples where the cohort has 280 makes the gate silent rather than wrong,
    and a silent gate is indistinguishable from a passing one.
    """
    from omicstra.graph import unit_counts_from_records

    root = settings.project_root(project_id)
    inv_counts: dict[str, int] = {}
    inv = root / "inventory.json"
    if inv.is_file():
        rec = json.loads(inv.read_text())
        inv_counts = unit_counts_from_records(
            [rec[k] for k in rec if isinstance(rec[k], dict)])

    ing_counts: dict[str, int] = {}
    ing = root / "data" / "canonical" / "ingest.json"
    if ing.is_file():
        src = (json.loads(ing.read_text()) or {}).get("sources", {})
        ing_counts = {k: int(v["n_spots"]) for k, v in src.items()
                      if isinstance(v, dict) and "n_spots" in v}

    if len(ing_counts) > len(inv_counts):
        return ing_counts, {"source": "data/canonical/ingest.json",
                            "n_samples": len(ing_counts),
                            "note": ("preferred over inventory.json, which covers "
                                     f"{len(inv_counts)} sample(s) - it was run over "
                                     "fewer than the cohort holds")}
    return inv_counts, {"source": "inventory.json", "n_samples": len(inv_counts),
                        "note": "the declared source" if inv_counts else
                                "no counts available - unit-floor gates cannot decide"}


def _bound_id(project_id: str | None) -> str:
    """the cohort's declared id. authoritative from project.json, never inferred
    from the directory name - a scaffolded cohort may be anywhere."""
    try:
        return ProjectConfig.load(project_id).project_id
    except (OSError, ValueError, KeyError):
        # a scaffolded cohort may have no readable project.json yet; falling back
        # to the caller's id is better than failing a read-only tool.
        return project_id or ""


# --- the compute path, made visible ----------------------------------------
#
# these two READ. there is deliberately no tool that runs an extraction, and the
# reason is the protocol rather than caution: MCP is request/response, and a full
# H&E extraction is ~5.8 hours on a laptop. a blocking tool for that would time
# out on every real cohort, so submission belongs to the dispatcher and the
# entry point belongs to the CLI. what a client needs from here is to SEE the
# compute path - what it would ask, what it would do, and what already exists -
# because until now a client could not tell that one existed at all.

@srv.tool(description=(
    "Report the preflight gates on the compute path for one encoder: which are "
    "resolved from this cohort's declarations, which are still open, and what "
    "accepting each one forecloses. A cohort that declares every answer runs "
    "unattended; an open gate means a person has to decide before accelerated "
    "compute starts, and the system does not pick a default. Reads declarations "
    "and the inventory record only - it measures nothing and starts nothing."))
def check_compute_gates(encoder: str | None = None,
                        project_id: str | None = None) -> dict:
    from omicstra.graph import _load_cohort, _primary_encoder
    from omicstra.protocols.encode import gate_record, preflight

    enc = encoder or _primary_encoder(project_id) or "virchow2"
    cohort = _load_cohort(project_id)

    counts, coverage = _unit_counts(project_id)
    reqs = preflight(cohort, enc, unit_counts=counts or None)
    rec = gate_record(reqs, enc)
    return {
        "project_id": _bound_id(project_id),
        "encoder": enc,
        "unattended": not any(r.open for r in reqs),
        "n_gates": len(reqs),
        "n_open": sum(r.open for r in reqs),
        # a gate decides from these counts, so how many units they cover is part
        # of the answer. a floor gate that stays silent because it only saw two
        # samples is worse than one that fires wrongly - it looks like a pass.
        "unit_counts_cover": coverage,
        "gates": [{"id": r.id, "question": r.question, "options": list(r.options),
                   "answer": r.answer, "answered_from": r.source,
                   "pre_answerable_by": r.pre_answerable_by,
                   "forecloses": r.forecloses, "open": r.open,
                   "observed": r.observed} for r in reqs],
        "verdict": rec.decision,
        "note": ("an open gate is a decision, not an error. the declaration named in "
                 "pre_answerable_by answers it in advance and is recorded with "
                 "actor=human either way."),
    }


@srv.tool(description=(
    "Describe what a compute run would do on this cohort without running it: "
    "how many units are declared, how many shards the work splits into, which "
    "outputs already exist and would therefore be skipped, and what is missing. "
    "Resume is decided by the output file, so an interrupted run restarts from "
    "what landed. Reads the filesystem only."))
def describe_compute_plan(encoder: str | None = None,
                          project_id: str | None = None) -> dict:
    from omicstra.adapters.canonical import CanonicalMissing, list_samples
    from omicstra.graph import _primary_encoder

    enc = encoder or _primary_encoder(project_id) or "virchow2"
    root = settings.project_root(project_id)
    try:
        samples = list_samples(root)
    except (CanonicalMissing, FileNotFoundError) as e:
        return {"project_id": _bound_id(project_id), "encoder": enc, "runnable": False,
                "why_not": str(e),
                "note": ("the cohort has not been converted to the package's input "
                         "contract - .h5ad plus a coordinates table - so there is "
                         "nothing to plan over yet.")}

    with_image = [s for s in samples if s.image]

    # the ORACLE is read, never written. it is what the ports are diffed
    # against, so a plan that named it as an output directory - which this did -
    # invites a run that contaminates the thing the verification depends on.
    oracle = settings.resolve(Path("data/embeddings")) / f"{enc}_niche"
    out_dir = root / "runs" / "encode" / enc
    in_oracle = {q.stem for q in oracle.glob("*.npy")} if oracle.is_dir() else set()
    done = {q.stem for q in out_dir.glob("*.npy")} if out_dir.is_dir() else set()
    todo = [s.sample_id for s in with_image if s.sample_id not in done]

    # how long, how much disk, how much memory - from measurements, or a refusal.
    # the units are the ENCODER's units (tiles here, not samples), taken from the
    # inventory record rather than re-counted.
    from omicstra.protocols.encode import estimate

    n_units, coverage = _unit_counts(project_id)
    todo_units = sum(n for sid, n in (n_units or {}).items() if sid in set(todo))
    cost = estimate(enc, todo_units) if todo_units else {
        "estimated": False, "why_not": "nothing left to run", "verdict": "unknown"}
    cost["units_from"] = coverage

    return {
        "project_id": _bound_id(project_id),
        "encoder": enc,
        "runnable": bool(with_image),
        "cost": cost,
        "n_samples": len(samples),
        "n_with_image": len(with_image),
        "n_shards": len(with_image),
        "already_done": len(done & {s.sample_id for s in with_image}),
        "remaining": len(todo),
        "remaining_examples": sorted(todo)[:5],
        "output_dir": str(out_dir),
        "oracle": {
            "path": str(oracle),
            "n_present": len(in_oracle),
            "role": "read-only. the artifacts the published grid was built from",
            "note": ("a run writes to output_dir and diffs against this. the "
                     "dispatcher refuses a shard whose output lands inside a path "
                     "the cohort declares read-only, so a plan cannot become a "
                     "contamination by accident."),
        },
        "unit_of_retry": "one shard per sample; three attempts, then recorded failed "
                         "and the run continues",
        "resume": "decided by the output file, not a ledger - the file is what the next "
                  "stage reads, so its existence is the only honest evidence",
        "note": ("samples without an image cannot enter the morphology arm and are "
                 "excluded from the shard list rather than failing inside it."),
    }
