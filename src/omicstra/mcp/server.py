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