# progress

running log of what is built, what it does, and how to exercise it.
`status.md` is the snapshot of the repo; this file is the build log.

---

## what is being built

A **package** that carries the analytical decisions of a spatial-biology
cross-modal study, so a second cohort does not re-derive them.

Not a pipeline for tnbc-92. tnbc-92 is the **fixture** - the one cohort whose
correct answers are already published, which makes it the regression oracle for
every component.

Three things travel in the package, and none of them name a cohort:

| what travels | where | example |
|---|---|---|
| **rules** - thresholds, verdict logic, guards | `configs/*.json`, `guards.py` | Moran's I > 0.3 for >= 3 of top 5 markers |
| **methods** - the step DAG and its contracts | `src/omicstra/` | the ten EDA steps; the six frozen stage signatures |
| **exemplars** - worked analyses to imitate | `notebooks/`, `scripts/` | why stain-norm was banned; why the unit moved to niche |

Cohort-specific facts - measured floors, which run won, file layouts - live in a
**project root** outside the package.

---

## milestones

| milestone | criterion | state |
|---|---|---|
| **ASK** | a client asks a question, gets a routed answer with guards, from committed artifacts - no GPU, no network | **in progress** |
| EXPLAIN | the same call emits a run record, decision ledger, and a notebook that re-derives its own numbers | not started |
| RERUN | the emitted workflow reproduces a headline value from a clean checkout | not started |
| EXTEND | a new cohort passes the gate chain without package code changes | not started |

---

## 2026-07-28 · environment verified

python 3.14.6, 66 packages, **zero source builds**. all 11 native C extensions
ship cp314 arm64 wheels. 11/11 runtime checks pass, including langgraph
conditional edges, checkpointer, `interrupt()` HITL pause/resume, and an MCP
tool round-trip.

closes backlog **I1**. two findings that change how code gets written:

- **mcp 2.0 renamed `FastMCP` -> `MCPServer`**; snake_case result fields. every
  design doc written before this date says FastMCP.
- **`from __future__ import annotations` - not the interpreter - forces langgraph
  state schemas to be module-level.** PEP 563 stringifies; `get_type_hints()`
  resolves against module globals, which cannot see an enclosing function's
  locals. identical on 3.11 / 3.13 / 3.14. a 3.15 bump will not fix it.

## 2026-07-28 · two artifact-correctness fixes  (`76a03ea`)

`configs/v3/R1_v3.json` stray `ref` prefix; `demo_2026_06.md:168` R6 linear probe
`0.235` -> `0.276` (0.235 was B3's value - a row transposition).

## 2026-07-28 · scripts/ tracked  (`ce23dcb`)

36 scripts, 568 KB, previously local-disk only. `scripts/_scratch/` stays ignored.

closed a real single-copy risk: `extract_virchow2_niche.py` (294 ln) is referenced
by no notebook, doc, or script, and produced the primary H&E feature that 9 of 10
runs consume. same for `extract_mc_labels`, `extract_mc_weights`, `rank1a`.

audit: 0 secrets, 1 absolute path (in the ignored `_scratch/`), 32 of 62 files
carry the `Path(__file__).parents[1]` coupling - left alone, that is the port's
problem and `settings.py` exists to replace it.

## 2026-07-29 · first MCP component - the EDA gate

The gate as a **validator**: reads the generalizable contract, checks a cohort's
measured summary against it, emits a verdict. No data, no R, no GPU.

```
configs/eda_contract.json     GENERALIZABLE - thresholds + verdict logic, zero cohort numbers
        x
{project}/eda_summary.json    COHORT-SPECIFIC - measured values
        |
        v
  verdict + per-check results + violations + cautions + advisories
```

On tnbc-92 it returns **`proceed_with_caution` against a declared `proceed`**.
8 checks pass, 0 fail. The disagreement is the two `null` conditional fields and
4 unaccepted risks - both real, both invisible to anyone reading the summary.

Six advisories name the checks the notebooks established that `EDA.md` never
encoded. The sharpest:

> Moran's I computed on 'CN1/C1' alone of 281 samples - this statistic gates the
> molecular encoder branch

### two roots, enforced

| root | holds | selected by |
|---|---|---|
| **package** | code · `configs/eda_contract.json` · exemplars | derived from the package, never cwd |
| **project** | `project.json` · `eda_summary.json` · `steps/` · `data/` · `runs/` | `OMICSTRA_PROJECT_DIR` / `--project-dir` |

`project_dir` is both the security boundary and the cohort selector - a tool call
passing a different `project_id` cannot reach outside it. tnbc-92 is the fixture
and uses the same code path as any external cohort. `project_id` is authoritative
from `project.json`, never inferred from the directory name.

**Cohort-declared paths resolve against the PROJECT root, never the package.**
Found by asking where the data actually lives: `project.json` declared
`niches_dir: "data/embeddings/niches_v3"` and the resolver was resolving it
against the *package*, so an external cohort pointed its data into the omicstra
install - the exact mixing this boundary exists to prevent. Fixed via
`ProjectConfig.path()`.

The tnbc-92 fixture is the one cohort whose data sits **outside** its own project
root - 151 GB of `data/` and `runs/` at the repo root, where `scripts/` and
`notebooks/` have always read it. Its `project.json` therefore declares
`../../data/embeddings/niches_v3`. Explicit and visible, rather than a silent
fallback. A cohort scaffolded by `omicstra init` keeps its data inside its own
root and needs no `../..`.

**Cohort selection lives in `.env` (gitignored), never in `.mcp.json` (committed).**
A committed launch config that names a cohort is the package/project mixing this
boundary exists to prevent.

### surface

| kind | name |
|---|---|
| tool | `check_eda_gate` · `describe_data_structure` |
| resource | `omicstra://eda-contract` · `omicstra://project/{id}/eda-summary` |
| CLI | `omicstra init --project-dir` · `omicstra gate` · `omicstra describe` |

---

## how to test it

### 1. from the shell

```bash
./venv/bin/omicstra gate                  # the fixture, via .env
./venv/bin/omicstra describe
./venv/bin/omicstra gate --project-dir /path/to/other/cohort
```

`gate` exits 0 on proceed / caution, 1 on stop - so it works in CI.

### 2. scaffold a cohort outside the package

```bash
./venv/bin/omicstra init --project-dir ~/cohorts/my-cohort
```

Creates `project.json` + `steps/`. Downloads nothing, creates nothing else.
Then point `OMICSTRA_PROJECT_DIR` at it.

### 3. from an MCP client

`.mcp.json` is already wired. Claude Code picks it up from the repo root; verified
over real stdio - `omicstra up · proto 2025-11-25`. Questions it can answer today:

- *"Does this cohort pass the EDA gate?"*
- *"Why doesn't the gate agree with the recorded verdict?"*
- *"What checks were never run, and what did the contract expect?"*
- *"What is the atomic unit, and which column is supervision-only?"*

It will **not** answer H1/H2/H3 questions yet - the routing table is not wired.
Ask it something outside the gate and it has no tool for it.

---

## finding · 2026-07-29 · H2 is under-reported, not wrong

The report states H2 as *"better than either modality alone"* and displays only
the raw-ST comparison (ARI 0.046 -> 0.246, "5x lift"). It contains **zero**
occurrences of `raw_he`, `raw_st`, `z_joint`, or `0.164`.

All of it was computed. `scripts/_scratch/eval_h2_mc_coherence.py:93` builds
`z_joint = concat([z_he, z_st])`, and `mc_coherence.parquet` carries both raw
baselines and all three views:

| | ARI | silhouette |
|---|---:|---:|
| raw ST alone | 0.046 | -0.072 |
| raw H&E alone | **0.164** | -0.006 |
| aligned, best `z_he` (R5) | **0.246** | ~0 |
| aligned, fused `z_joint` (R6) | 0.205 | |

Two consequences, and they point in opposite directions:

1. **H2 is stronger than published.** Aligned beats raw H&E 1.5x as well as raw
   ST 5.4x, so the "either modality alone" clause is actually satisfied on ARI.
2. **The mechanism is not fusion.** `z_joint` (0.205) scores *below* `z_he`
   (0.246). The gain comes from ST-supervising the H&E tower, not from combining
   the two. A reader who takes "manifold alignment" to mean "concatenate both
   modalities" would draw the wrong conclusion.

Also: silhouette is ~0 across every run and baseline (-0.006 to +0.008), so ARI
carries the row alone; and no `raw_st` linear probe exists, so the probe-based
comparison against ST cannot be completed.

**Action: none now.** The report is the spec and the oracle and stays read-only
per `mcp_plan.md` §2. This is logged for the deliberate re-baseline before the
paper. When that happens the H2-A block should gain the `raw_he` row, the
`z_joint` row, and one sentence distinguishing supervision from fusion.

This is the third instance of the same pattern - the project computes more than
it reports (see also: the three `eval/*/summary.json` files each dropping a
different run). Worth a standing check rather than three separate fixes.

## known gaps, named honestly

**The contract is a skeleton, not the contract.** `eda_contract.json` encodes 10
checks. The notebooks declare far more - roughly 40-60 real decisions:
`MIN_MEDIAN_GENES` 500 · `NOVAE_MIN_SPOTS` 512 · `RHO_THRESHOLD` -0.1 · the 3-way
encoder routing · `MAD_TOP_FRACTION` · `MIN_NICHE_UMI` · `MIN_PATHWAY_SIZE/MAX` ·
`FDR_STRICT` · `OR_CUT` · IQR outlier flags · gene-panel intersection · effective
rank · clinical missingness. What is built proves the mechanism (contract ×
summary -> verdict); it does not yet carry the project's actual rules. Extracting
them from the notebooks is the highest-value work left in the EDA layer.

**`omicstra init` scaffolds too little.** It creates `project.json` + `steps/`.
A project home should be the shape of this repo minus the package - `data/inputs/`,
`data/embeddings/`, `runs/`, `steps/`, `notebooks/`, a `program.md` template, and
a `.gitignore` carrying the same policy as here, so a new cohort is its own git
repo from minute one and the trace/commit coupling has something to attach to.

## next

1. **`omicstra init` -> a real project home** (above), so cohorts have somewhere
   to live that is not an arbitrary path
2. **grow the contract** from 10 checks to what the notebooks declare
3. **`MODALITY_TEMPLATE.md`** - the contract a modality agent generates against.
   cited by three docs, does not exist. must land before the first generated step
   or the step defines the contract by accident.
4. **generated-step conventions** - `{project_dir}/steps/s{NN}_{modality}_{step}.py`
   plus `MANIFEST.json`; header carries `contract@sha256` + `trace_id`; commit
   message carries the trace id. the contract sha makes staleness computable.
5. **first generated step** - `s05_st_spatial_autocorr`: its output is an edge
   (it routes the encoder branch) and it is where the n=1 defect lives.
6. remaining EDA gaps: input-provenance per check · itemised funnel · platform
   floor · 3-way encoder routing · label-granularity NMI · annotation coverage.
7. **packaging** - `configs/` sits at repo root, so it is not inside the wheel.
   a non-editable install needs it under `src/omicstra/` or declared as package data.
8. `docs/reports/tnbc-92/index.html` stays **read-only** - spec and oracle.