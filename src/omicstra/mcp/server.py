"""omicstra MCP server - the interface. read tools first (zero compute).

read-path tools (phase 1) resolve through ArtifactStore + the routing table.
compute-path tools (phase 2+) trigger the durable pipeline. this is a stub -
wire the mcp sdk in phase 1.
"""
from __future__ import annotations

# read-path tools (phase 1):
#   route_question(nl, project_id)          -> {hypothesis, runs, view, guards[]}
#   get_hypothesis_result(project_id, h, ...) -> metrics + 95% CIs
#   get_guards(project_id, hypothesis)      -> construct-validity guards
# compute-path tools (phase 2+):
#   run_pipeline(project_id, params)        -> job handle (temporal-backed)


def main() -> None:
    raise NotImplementedError("phase 1: wire mcp sdk + read tools over ArtifactStore")


if __name__ == "__main__":
    main()