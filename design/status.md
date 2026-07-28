# status - 2026-07-28

snapshot before the first MCP component lands. covers: environment verification, the `scripts/`
audit, and which notebooks carry forward vs which are retained as evidence only.

scaffolding drafted by Claude per memory `feedback_writing_voice` - voice pass before this becomes
a shared doc.

---

## 1. environment - verified, PASS

`venv_test/` on python 3.14.6 (homebrew python@3.14). **66 packages, zero source builds.** all 11
native C extensions ship cp314 arm64 wheels: `pydantic_core`, `rpds-py`, `cffi`, `orjson`,
`ormsgpack`, `xxhash`, `zstandard`, `uuid_utils`, `jiter`, `PyYAML`, `charset-normalizer`.

| package | version |
|---|---|
| mcp | **2.0.0** |
| langgraph | 1.2.9 |
| langchain-core | 1.5.1 |
| langchain-anthropic | 1.5.2 |
| anthropic | 0.120.0 |
| pydantic / pydantic-settings | 2.13.4 / 2.14.2 |

11/11 runtime checks pass: StateGraph + conditional edges, determinism across 3 invocations,
mermaid render, checkpointer + state history, `interrupt()` HITL pause/resume, MCP
`list_tools`/`call_tool` round-trip, `pip install -e .`, `ProjectConfig.load("tnbc-92")`.

**closes backlog item I1.** the 3.14 wheel risk is not real, so design decisions D1 (temporal now
vs later) and D2 (LangGraph vs LCEL) are not invalidated by it.

### two findings that change how code gets written

| # | finding |
|---|---|
| 1 | **mcp 2.0 renamed `FastMCP` -> `MCPServer`.** `mcp.server.fastmcp` no longer exists. same shape (`.tool()`, `run_stdio_async`) plus a `structured_output` option. also snake_case result fields (`resource_templates`, not `resourceTemplates`). every design doc written before today says FastMCP |
| 2 | **`from __future__ import annotations` forces langgraph state schemas to be module-level.** PEP 563 stores annotations as strings; `get_type_hints()` resolves them against module globals, which cannot see an enclosing function's locals -> a function-local `TypedDict` raises `NameError`. verified identical on 3.11 / 3.13 / 3.14. **not a 3.14 issue and not a langgraph bug** - a 3.15 bump will not fix it; dropping the future import would. every file in `src/omicstra/` has that import, so: declare state schemas at module scope |

`temporalio[langgraph]` not tested - not a dependency, and durability has no role in ASK.

---

## 2. scripts/ audit

62 python files (35 root + 27 `_scratch/`) + 1 R file. **1.3 MB total.**

### safety - clean

| check | result |
|---|---|
| secrets / API keys / credentials | **0** |
| absolute paths | 1, in `_scratch/mc_pathway_embeddings.py` -> `/Users/sanati/BForePC/gpath2vec` |
| PHI | none - all inputs are the public Zenodo release |
| `Path(__file__).resolve().parents[1]` coupling | 32 of 62 - real, but it is the port's problem; `settings.py` exists to replace it. not fixed now |

### the core chain - 12 scripts behind the published results

| script | ln | referenced by | produces |
|---|---:|---|---|
| `eval.py` | 416 | nb 2 · doc 15 · scr 11 | H1/H2 metrics |
| `align.py` | 672 | nb 2 · doc 13 · scr 9 | R1-R6; `LateFusion` / `CrossAttnFusion` |
| `build_niche_join.py` | 490 | nb 1 · doc 10 · scr 1 | the waist + `manifest.json` |
| `align_classical.py` | 261 | doc 7 · scr 4 | B1/B2/B3 |
| `eval_h3_pathway_cca.py` | 329 | nb 1 · doc 6 · scr 3 | H3 AUCell arm |
| `eval_h1_bootstrap_ci.py` | 201 | nb 1 · doc 4 · scr 1 | AUC / CKA CIs |
| `align_b4.py` | 180 | doc 5 | B4 random-init |
| `eval_h3_pathway_cca_gpath2vec_v2.py` | 675 | nb 1 · doc 3 · scr 1 | H3 headline arm |
| `eval_alignment_biology.py` | 446 | doc 3 · scr 1 | bio/patient ratio |
| `niche_aucell.py` | 444 | nb 3 · scr 1 | AUCell per niche |
| `scale_embeddings.py` | 419 | doc 2 | Novae full cohort |
| `extract_biological_signals.py` | 416 | doc 2 | mc / tls / morphology TSVs |

### unreferenced but load-bearing

these produced live artifacts and are mentioned by **no** notebook, doc, or script:

| script | ln | produced | consumed by |
|---|---:|---|---|
| **`extract_virchow2_niche.py`** | 294 | `data/embeddings/virchow2_niche/` (1.4 GB) | **9 of 10 runs - the primary H&E feature** |
| `extract_mc_labels.py` | 196 | `biological_signals/mc_labels.tsv` | H2 audit-correct label |
| `extract_mc_weights.py` | 183 | `biological_signals/mc_weights.tsv` | R2 supervision |
| `rank1a.py` | 234 | `runs/tnbc-92/rank1a/` | the Virchow2 go/no-go probe |

`notebook_flows.md` E11 records "Virchow2 extraction has no notebook". it is broader than that: no
notebook, no doc reference, no script reference. **~900 lines of unreferenced extraction code sit
behind the most load-bearing encoder output in the project.** this - not the 159 GB of regenerable
data - is the real single-copy risk.

### libraries

| tier | packages (count = scripts importing) |
|---|---|
| core / read path | `numpy` 60 · `pandas` 59 · `scipy` 25 · `sklearn` 22 · `click` 17 |
| plotting | `matplotlib` 7 · `seaborn` 6 · `plotly` 3 |
| **heavy -> `[compute]` extra** | `torch` 10 · `anndata` 4 · `timm` 3 · `novae` 2 · `decoupler` 2 · `scanpy` 1 |
| external, not on PyPI | `gpath2vec` 1 |
| intra-script imports | `align` 4 · `align_classical` 1 · `eval_alignment_biology` 1 |

that last row: `align.py` is imported by 4 other scripts. `scripts/` is already a package that was
never declared one. the core/heavy split above is the empirical basis for the `omicstra` vs
`omicstra[compute]` dependency split (backlog I3).

### dead

11 in `scripts/`: `eval_compare_and_plot`, `eval_paired_asym`, `eval_rank_matched_cka`,
`eval_h3_pathway_cca_gpath2vec_vs_aucell_tflow`, `generate_summary_pdf`, `run_alignment_90p`,
`rank2_alignment`, and the four `test_*.py`. 16 of 27 in `_scratch/`, mostly figure generators.

---

## 3. notebooks - what carries forward

18 notebooks. the split is **live** (build on these) vs **evidence** (keep, cite, do not extend).
nothing is being moved or deleted; the trial-and-error notebooks are the record of why the current
design is what it is, and several are load-bearing for decisions that are still binding.

### live - carry forward

| notebook | role | why |
|---|---|---|
| `exploration/eda_data_inventory.ipynb` | **gate 0** | cohort inventory; resolved `encoder_input_decision` |
| `exploration/eda_biological_signal.ipynb` | **gate 0** | markers, Moran's I, batch -> the `proceed` verdict |
| `embeddings/novae_st_embeddings.ipynb` | ST encoder | current copy; sole carrier of `NOVAE_MIN_SPOTS`, `IDS_MAP`, `OUT_FULL` |
| `embeddings/virchow2_cell_vs_niche_interp.ipynb` | encoder gate | Virchow2 primary · stays raw · residualisation dead |
| `embeddings/niche_pathway_analysis_full.ipynb` | pathway EA | canonical of the three; full-cohort QC params |
| `final/05_summary_umaps_v3.ipynb` | **canonical** | source of the published report |

### evidence - retain, do not build on

| notebook | what it establishes | still binding? |
|---|---|---|
| `embeddings/uni2_reinhard.ipynb` | Reinhard changes rho -0.011, patient gap -4.3%, per-patch cos 0.9315 | **yes** - the basis for `program.md` hard constraint 2 (stain norm banned pre-FM) |
| `embeddings/uni2_patch_extraction.ipynb` | UNI2 pilot, 4413 patches, 5 patients | yes - the alternative-encoder baseline |
| `embeddings/pca_hvg_baseline.ipynb` | patient gap 0.671 vs Novae 0.022; rho sign flips positive | yes - why the Novae fallback was never triggered |
| `experiments/alignment_late_interaction.ipynb` | spot-level: within-patient R@1 0.136, **cross-patient 0.002 = random** | **yes** - why the atomic unit moved to niche |
| `final/v2_summary_umaps.ipynb` | v2 gpath2vec dim collapse | yes - published negative result; produced guard G5 |
| `final/05_summary_umaps.ipynb`, `final/proposal_h1h2h3_evaluation.ipynb` | v1 grid | yes - deliberate v1 head-to-head baseline |
| `embeddings/niche_pathway_spatial_maps.ipynb` | Track-A hits on H&E | presentation only, no decision |

### noted, not acted on

| notebook | note |
|---|---|
| `exploration/novae_st_embeddings.ipynb` | 26 of 30 cells identical to the `embeddings/` copy, which is newer (verified 2026-07-28). the exploration copy lacks all three v3 params. superseded - **not moved** |
| `embeddings/niche_aucell_analysis.ipynb` ≈ `niche_pathway_analysis.ipynb` | 15 of 17 cells identical; both pilot-scale (`PILOT_PATIENTS = (1,3,83)`), both superseded by `_full`. **not moved** |
| `final/h3_gpath2vec_v3.ipynb` | reads `runs/tnbc-92`, **not** `runs/tnbc-92_v3` (verified). correct by design - it is the phase-1 preview of the v3 pathway build against v1-trained latents - but the filename misleads. **not renamed** |

---

## 4. changes in this commit

| change | rationale |
|---|---|
| `.gitignore`: `scripts/` -> `scripts/_scratch/` | track the reference pipeline; keep the scratch workspace out. removes the single-copy risk on ~900 lines of unreferenced extraction code |
| `scripts/test_*.py` -> `scripts/diag_*.py` (4 files) | they are one-off diagnostics, not pytest tests; the names would be collected by the future test suite. conclusions already recorded in `eda_summary.json` |
| this file | status snapshot |

deliberately **not** changed: the 32 `parents[1]` call sites, the notebook duplicates, the
`h3_gpath2vec_v3` name, and the one absolute path (now inside gitignored `_scratch/`).

---

## 5. open, in order

1. first MCP component: the data-structure + EDA gate, built on the learned checks in the two EDA notebooks
2. EDA gaps to encode: input-provenance per check · itemised funnel · platform floor · 3-way encoder routing · label-granularity NMI · annotation coverage
3. `docs/reports/tnbc-92/index.html` remains **read-only** - it is the spec and the oracle