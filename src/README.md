# src/omicstra - the pip-installable MCP deliverable

the reference pipeline lives in `scripts/` (gitignored QA workspace). this package
is where it gets ported behind stable contracts so the tool generalizes past tnbc-92.

layout:

```
settings.py    injected paths (kills the scripts/ parents[1] blocker)
config.py      ProjectConfig - a cohort as data (tnbc-92 = instance-zero)
stages.py      the six frozen pipeline-stage signatures (THE SEAM)
guards.py      the construct-validity guard library (the generalizable IP)
artifacts.py   owns the runs/{project}/ schema (read-path cache resolution)
mcp/server.py  the MCP interface (read tools first, zero compute)
agents/        langgraph nodes (orchestrator, he, st, alignment, eval)
```

build order: P0 this scaffold -> P1 agent graph + read path -> P2 temporal over
the scripts -> P3 (optional) nextflow -> P4 karpathy loop.
see `docs/mcp_agent_design.md` for the node spec and routing table.

these files are stubs (`NotImplementedError`) - they define contracts, not behavior.
the seam: temporal orchestrates the `stages.py` signatures; each stage body swaps
from a python call (P2) to `nextflow run stage.nf` (P3) without touching the graph.