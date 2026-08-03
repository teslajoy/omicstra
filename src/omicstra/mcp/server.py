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

from mcp.server import MCPServer

from ..config import ProjectConfig
from ..eda import load_contract, load_summary, run_gate
from ..settings import settings

srv = MCPServer("omicstra")


# --- tools: model-invoked, do work ----------------------------------------
@srv.tool(description=(
    "Run the EDA admissibility gate for a cohort. Returns a verdict "
    "(proceed | proceed_with_caution | stop), per-check results, blocking "
    "violations, cautions, and advisories for checks the contract expects but "
    "the cohort summary cannot answer. Reads committed artifacts only."))
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
        },
        "checks": [c.model_dump() for c in r.checks],
        "violations": r.violations,
        "cautions": r.cautions,
        "advisories": r.advisories,
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
    from .. import eda_steps as st
    from ..eda import load_summary

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


# --- resources: URI-addressed data, not model-invoked ---------------------
@srv.resource("omicstra://eda-contract", mime_type="application/json",
              description="the generalizable EDA gate contract - thresholds and "
                          "verdict logic, no cohort measurements")
def eda_contract() -> str:
    return json.dumps(load_contract(), indent=2)


@srv.resource("omicstra://project/{project_id}/eda-summary",
              mime_type="application/json",
              description="the selected cohort's measured EDA summary")
def eda_summary(project_id: str) -> str:
    # project_id is advisory; the boundary is settings.project_dir
    return json.dumps(load_summary(project_id), indent=2)


def main() -> None:
    srv.run()


if __name__ == "__main__":
    main()