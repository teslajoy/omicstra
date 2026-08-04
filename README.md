# omic<span style="color:#5DCAA5">stra</span>

a multi-agent MCP server for cross-modal reasoning in spatial biology. modality-specialized components are orchestrated with LangGraph; foundation model encoders are pluggable; integration strategy is a configurable experimental dimension, not a fixed pipeline choice.

the seed implementation evaluates 92 TNBC patients from [Wang et al. 2024](https://www.nature.com/articles/s41467-024-54145-w) using H&E morphology (Virchow2 primary, UNI2 swap) and spatial transcriptomics (Novae GNN).

`active development` `ResearchHub Foundation grant` `OHSU Knight Cancer Institute` `MIT`

---

## hypotheses

| | | metrics |
|---|---|---|
| H1 | aligned embeddings retrieve better cross-modal matches than any unaligned baseline | Recall@K - MRR - CKA |
| H2 | manifold alignment preserves biological structure better than either modality alone | ARI - silhouette - UMAP |
| H3 | shared space encodes interpretable biological pathway signals | CCA - spatial maps |

early vs late fusion is an experimental variable, not an assumption. 

---

## seed cohort results (v3, 10-run grid)

live report: **[teslajoy.github.io/omicstra/reports/internal/tnbc-92/](https://teslajoy.github.io/omicstra/reports/internal/tnbc-92/)**

evaluated on **35,594 niches across 14 held-out TNBC patients** (zero patient leakage across train / val / test). 6 contrastive runs (R1-R6) + 4 classical baselines (B1-B4):

- **H1 (cross-modal retrieval).** R4 wins. AUC 0.859 (95% CI 0.855-0.862). R4 is the only run using cross-attention - the other 5 contrastive runs use late fusion (mean-pooled H&E + independent MLPs).
- **H2 (biology preservation).** Split outcome. Late-fusion contrastive (R1, R6) wins K-means MC ARI; R4 still encodes MC biology at linear-probe parity. KMeans-ARI is geometry-biased - load-bearing diagnostic is the linear probe.
- **H3 (pathway transfer).** R4 wins again. 15/15 cells BH-FDR < 0.05 cross-patient (Welch z 3.8-5.7). At DAG resolution, R4 stays best (78% sig nodes) on clinically interpretable hits: BTLA checkpoint, IL signaling, IFN regulation, ECM collagen biology.

**R4 = the only non-late-fusion run** (cross-attention between the niche's 7 Virchow2 tile tokens and the 576-d ST vector). H1 + H3 both reward keeping local morphological heterogeneity un-pooled. See report for the full proposal-alignment matrix and 95% CIs.

---

## architecture

three layers:

```
layer 0: EDA (gate)
  is data usable? quality? signal? modalities alignable?
  -> proceed / proceed with caution / stop

layer 1: pipeline (execution)
  embed -> align -> eval
  fixed protocol, deterministic with seed

layer 2: karpathy loop (outer optimization)
  editable asset: alignment_config.json
  scalar metric: Recall@K or MRR (fixed)
  proposes config -> runs pipeline -> scores -> keeps or discards
  output: git log of validated decisions + winner.json
```

program.md defines search space, constraints, stopping criteria.

embedding dimensions: Virchow2 1280d (primary) / UNI2 1536d (alternative) - Novae GNN 64d (novae_latent) -> shared 512d

alignment strategies (experimental variable):
```
  late interaction  - tiles aggregated per spot (mean/max),
                      then independent MLPs -> shared 512d (InfoNCE)
  early interaction - tile-level cross-attention between H&E tokens
                      and ST spot before projection (~4M params, local K tiles)
  comparison: does tile-level morphological detail lost at aggregation
              matter for cross-modal alignment quality?
```

---

## initial repo structure-design

```
omicstra/
│
├── CLAUDE.md                        # north star for Claude Code
├── EDA.md                           # gate - data quality checks before pipeline
├── PLAN_RULES.md                    # constraints - no alignment until EDA passes
├── MODELS.md                        # training provenance + tissue compatibility per model
├── MODALITY_TEMPLATE.md             # how to add a new modality
├── .mcp.json                        # MCP server entry
├── .env.example                     # COMPUTE_BACKEND=local|cloud|slurm
│
├── .claude/
│   ├── skills/                      # he_agent - st_agent - alignment
│   │                                # karpathy_loop - eval - notebook_gen
│   └── hooks/                       # post_edit - block_data_writes - run_tests
│
├── docs/                            # -> teslajoy.github.io/omicstra
│   ├── index.html
│   ├── timeline.html
│   └── images/
│
├── knowledge/                       # shared domain assets - committed
│   ├── pathways/
│   └── annotations/
│
├── projects/                        # committed
│   ├── registry.json
│   └── {project_id}/
│       ├── project.json
│       ├── program.md               # karpathy loop - search space + constraints + metric
│       └── eda_summary.md           # EDA gate output
│
├── src/omicstra/                    # the pip-installable package - src layout
│   ├── agents/                      # orchestrator - he_agent - st_agent - alignment - eval
│   ├── adapters/                    # a cohort's files -> AnnData
│   ├── mcp/                         # server.py
│   ├── eda.py  eda_steps.py  eda_graph.py    # the gate - steps - subgraph
│   ├── records.py  guards.py        # the record contract - construct-validity guards
│   └── settings.py  config.py  stages.py  artifacts.py  cli.py
│
├── config/
│   ├── qc_params.json               # K1 editable asset
│   └── alignment_config.json        # K2 editable asset
│
├── compute/
│   └── slurm/                       # optional - uni2_array - novae_array - alignment_train
│
├── runs/            [ignored]
│   └── {project_id}/{run_id}/
│       ├── run_config.json
│       ├── winner.json              # best validated run record
│       ├── qc_summary.json
│       ├── embedding_summary.json
│       ├── alignment_summary.json
│       ├── evaluation_summary.json
│       └── synthesis.md
│
├── data/            [ignored except READMEs]
│   ├── inputs/{project_id}/
│   ├── embeddings/{project_id}/
│   └── outputs/{project_id}/
│
└── notebooks/
    ├── exploration/ [ignored]
    └── final/       [git]           # best run generated in trace_notebook 
```

---

## extending to a new modality

`MODALITY_TEMPLATE.md` is the contract a new modality agent (single-cell, protein, electron microscopy, etc.) must satisfy: a frozen foundation-model encoder, a niche-aggregation step, a clean ST/H&E-equivalent feature parquet shape, and the construct-validity guards the eval agent enforces. each modality adds one row to the alignment grid; the orchestrator's routing table and the karpathy-loop search space pick it up automatically.

---

## seed cohort

[Wang et al. 2024](https://www.nature.com/articles/s41467-024-54145-w) - 92 TNBC patients, co-registered ST + H&E WSI. platform: original Spatial Transcriptomics (Stahl et al. 2016, KTH/Spatial Transcriptomics AB, acquired by 10x Genomics 2018) - 1934 spots/array, 100um diameter, 200um center-to-center. not 10x Visium.

| resource | url |
|---|---|
| processed data | [doi:10.5281/zenodo.14204217](https://doi.org/10.5281/zenodo.14204217) (mc-labels release also at [doi:10.5281/zenodo.8135721](https://doi.org/10.5281/zenodo.8135721)) |
| original analysis code | [BCTL-Bordet/ST](https://github.com/BCTL-Bordet/ST) |

---

## quick start

---

## stack

| layer | tool |
|---|---|
| orchestration | LangGraph StateGraph |
| routing | Claude Sonnet 4.6 |
| synthesis | Claude Opus 4.6 |
| tracing | LangSmith - observability only |
| H&E foundation model | Virchow2 (primary, 1280-d) - UNI2 as swap (1536-d) |
| ST foundation model | Novae GNN |
| pathway | gpath2vec + biological pathway embeddings |
| alignment | contrastive MLP + cross-attention bridge |
| compute | laptop / cloud GPU / SLURM (ARC HPC at OHSU) |
| vector retrieval | Qdrant |
| graph store | Neo4j |
| interface | Model Context Protocol |

---

## reproducibility

every run writes `runs/{project_id}/{run_id}/` - config, QC, embeddings, alignment, evaluation, synthesis. `notebooks/final/` contains the best run in notebook which is Opus-generated from the LangSmith trace, reruns can be done top-to-bottom with fixed seed.

---

## deliverables

- [x] v3 10-run alignment grid on TNBC-92 ([live report](https://teslajoy.github.io/omicstra/reports/internal/tnbc-92/))
- [ ] pip-installable MCP server (MIT)
- [ ] second-modality plug-in via `MODALITY_TEMPLATE.md` (target: CODA or CyCIF)

---

## team

**Nasim Sanati, M.S.** - AI/ML systems - MCP server - agent orchestration - embedding pipelines  
**Cameron Watson, M.S.** - biological validation - spatial transcriptomics - TME interpretation  
**Dr. Allison Creason** - scientific oversight  

OHSU Knight Cancer Institute - Department of Biomedical Engineering - Creason Lab

---

## citation

```bibtex
@software{sanati2026omicstra,
  author  = {Sanati, Nasim and Watson, Cameron},
  title   = {omicstra: cross-modal biomedical embedding alignment + reasoning via MCP orchestration},
  year    = {2026},
  url     = {https://github.com/teslajoy/omicstra},
  license = {MIT}
}
```

*funded by ResearchHub Foundation ([doi:10.55277/researchhub.6ou1w3h3](https://doi.org/10.55277/researchhub.6ou1w3h3))*  
*nonprofit recipient: OHSU Foundation - EIN 23-7083114*  
*seed cohort: Wang et al. 2024, Nat. Commun. 15:10232*
