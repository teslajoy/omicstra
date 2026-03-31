# CLAUDE.md

this file provides guidance to Claude Code when working with code in this repository.

## project overview

omicstra is a multi-agent MCP server for cross-modal reasoning in spatial biology. it aligns embeddings from H&E histopathology (UNI2/Virchow2) and spatial transcriptomics (Novae GNN) into a shared 512d space, then exposes cross-modal retrieval and evaluation through the Model Context Protocol.

seed dataset: 92 TNBC patients from Wang et al. 2024 (co-registered Visium ST + H&E WSI, Zenodo doi:10.5281/zenodo.8135721).

---

## critical constraints

- do not propose alignment, contrastive learning, retrieval evaluation, or agent routing until EDA.md is satisfied
- `projects/{project_id}/eda_summary.json` must exist and pass before any pipeline work
- Novae GNN training data does not strongly overlap with TNBC Visium — embedding quality on this tissue is empirical, not an assumption
- UMAP inspection of Novae embeddings is part of EDA, not optional
- if Novae underperforms, fall back to PCA on HVGs as ST baseline before blaming alignment
- see MODELS.md for tissue-model compatibility matrix

---

## three-layer architecture

```
layer 0: EDA (gate)
  is data usable? quality? signal? modalities alignable?
  -> proceed / proceed with caution / stop

layer 1: pipeline (execution)
  embed -> align -> eval
  fixed protocol, deterministic with seed

layer 2: karpathy loop (outer optimization)
  editable asset: alignment_config.json
  scalar metric: Recall@K or MRR (fixed, never changed mid-loop)
  proposes config -> runs pipeline -> scores -> keeps or discards
  output: git log of validated decisions + winner.json
```

`program.md` is the most important file — defines search space, constraints, stopping criteria. the human writes program.md, not code.

---

## embedding dimensions

- UNI2: 1536d (Virchow2 as swap: 1280d)
- Novae GNN: 256d
- shared space: 512d

---

## alignment strategies (experimental variable)

- late interaction — tiles aggregated per spot (mean/max), then independent MLPs -> shared 512d (InfoNCE)
- early interaction — tile-level cross-attention between H&E tokens and ST spot before projection (~4M params, local K tiles)
- comparison: does tile-level morphological detail lost at aggregation matter for cross-modal alignment quality?

early vs late interaction is an experimental variable, not an assumption.

---

## hypotheses

- H1: aligned embeddings retrieve better cross-modal matches than unaligned baselines (Recall@K, MRR, CKA)
- H2: manifold alignment preserves biological structure better than either modality alone (ARI, silhouette, UMAP)
- H3: shared space encodes interpretable biological pathway signals (CCA, spatial maps)

---

## agent constraints

- `he_agent`: processes H&E only — cannot perform alignment or eval
- `st_agent`: processes ST only — cannot perform alignment or eval
- `alignment_agent`: consumes embeddings only — cannot modify raw data
- `karpathy_loop`: proposes config changes from search space only — cannot modify eval metrics or baselines
- `eval_agent`: computes metrics only — cannot propose configs
- `orchestrator`: routes between agents only — cannot execute modality-specific steps

---

## eval baselines (required before winner.json)

every run must beat all three on Recall@1 before writing winner.json:

- random retrieval
- spatial-NN (Visium coordinate nearest neighbor)
- unaligned concat (L2-norm UNI2 + Novae, no projection)

a run that does not beat all three baselines is not a winner.

---

## fast iteration

use `config/debug.json` for loop development: 5 patients, 500 tiles, 5 epochs.
debug runs write to `runs/{project_id}/debug_{run_id}/` — never candidates for winner.json.
do not compare debug metrics to full run metrics.

---

## stack

| layer | tool |
|---|---|
| orchestration | LangGraph StateGraph |
| routing | Claude Sonnet 4.6 |
| synthesis | Claude Opus 4.6 |
| tracing | LangSmith (observability only) |
| H&E encoder | UNI2 1536d (Virchow2 1280d as swap) |
| ST encoder | Novae GNN 256d |
| pathway embeddings | gpath2vec |
| alignment | contrastive MLP + cross-attention bridge |
| vector retrieval | Qdrant |
| interface | Model Context Protocol (MCP) |
| compute | local / cloud GPU / SLURM |

---

## key directories

- `src/agents/` — orchestrator, he_agent, st_agent, alignment_agent, eval_agent, karpathy_loop
- `src/tools/` — embed_he, embed_st, align, retrieve, evaluate
- `src/models/encoders/` — foundation model wrappers
- `src/models/alignment.py` — MLP + cross-attention bridge
- `src/eval/` — h1, h2, h3 hypothesis evaluation + failure_modes.py
- `src/mcp/` — MCP protocol server
- `projects/` — registry, per-project config, program.md, eda_summary.json (committed)
- `config/` — qc_params.json (K1 editable), alignment_config.json (K2 editable), debug.json
- `knowledge/` — shared domain assets: pathways, annotations (committed)
- `runs/` — execution traces per run (git-ignored)
- `data/` — inputs, embeddings, outputs (git-ignored except READMEs)
- `notebooks/final/` — best-run notebooks from LangSmith traces (committed)

---

## key docs

- `README.md` — what + why
- `CLAUDE.md` — dev guidance (this file)
- `EDA.md` — gate checks before pipeline work
- `PLAN_RULES.md` — constraints on when to proceed
- `MODELS.md` — training provenance + tissue compatibility per model
- `MODALITY_TEMPLATE.md` — how to add a new modality
- `projects/{project_id}/program.md` — search space, constraints, stopping criteria

---

## reproducibility

every run writes full config to `runs/{project_id}/{run_id}/`. notebooks in `notebooks/final/` are Opus-generated from LangSmith traces, rerunnable top-to-bottom with fixed seed 42.

---

## token budget

~$850 for LLM API (~85% of $1,000 ResearchHub grant).
real per-experiment cost: ~$1–2 (Opus synthesis input context is 20–50k tokens per trace).
gives ~400–850 full experiments. track burn rate from experiment 1.

---

## environment

- python 3.14
- virtual environment: `venv/`
- environment config: `.env` (see `.env.example`)
- `COMPUTE_BACKEND=local|cloud|slurm`

---

## team

- Nasim Sanati — AI/ML systems, MCP server, agent orchestration, embedding pipelines, main analysis pipeline
- Cameron Watson — biological validation, spatial transcriptomics, TME interpretation
- Dr. Allison Creason — scientific oversight

OHSU Knight Cancer Institute, Creason Lab. funded by ResearchHub Foundation (doi:10.55277/researchhub.6ou1w3h3).