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

## repo reorganisation - target recorded 2026-07-30, deferred

The current layout starts everything at the repo root. A cohort's material should
live under its own project root instead:

```
projects/tnbc-92/
├── data/        inputs · embeddings
├── runs/        routes + results
├── report/      the published view
├── context/     program.md · eda_summary.json · schema docs
├── scripts/     this cohort's implementation (pre-port)
└── notebooks/   this cohort's decision record
```

**It is one operation, not several.** Every script does
`ROOT = Path(__file__).resolve().parents[1]` and then reads `data/...` relative to
that. Moving `scripts/` alone silently redefines `ROOT` and breaks all 32; the same
holds for the 6 live notebooks. So scripts + notebooks + data + runs move together
or not at all - and that is the 151 GB operation.

**Sequencing.** Do it *after* the port behind `stages.py` (RERUN), when one place
resolves paths instead of 32. Doing it before means doing it twice.

**One split to settle first.** The notebooks currently serve two roles: tnbc-92's
decision record, and the exemplar corpus a future modality agent reads. Those
separate cleanly - the *derivation* stays with the cohort, the *derived rules* ship
in the package (contract, guards, routing table). That extraction is the "grow the
contract" item below, and it belongs before the move, not after.

The mechanism is already correct: `ProjectConfig.path()` resolves cohort paths
against the project root, and tnbc-92 declares `../../data/...` explicitly. Only
the physical layout is legacy, so the move is a data operation rather than a
redesign.

## open experiment · histological progression, never asked

**Clinical stage is not testable at this unit, and annotation would not fix it.**
Stage is a property of the patient, not a region. Every niche in a patient inherits
one value, so NMI(stage, patient_id) = 1.0 by construction - the archetype failure
(0.89) in a more extreme form. On a single section the value is constant, so there
is nothing to separate. The obstacle is the construct, not the missing annotation.

**Histological progression is testable, and was not tested.** `in situ` vs `Tumor`
and `Tumor region` is annotated per spot in Wang's 18-class set, varies within a
section, and is niche-level. That is almost certainly what a pathologist means by
"different stages on one slide," and this grid never asked it.

**Counted 2026-07-30: not answerable with the existing annotation.**

| what | state |
|---|---|
| label | `dominant_18class` in `biological_signals/morphology_labels.tsv`, center-spot in the niche join |
| `in situ` | **181 spots, 0.19% of 95,161** across 20 subarrays |
| `Tumor` | 27,104 spots, 28.5% |
| sections carrying both | 18, but **median 4 in-situ spots each** (range 1-29, 124 total) |
| verdict | **blocked.** four spots per section cannot support a within-section test, and 7-spot niches built on them would overlap almost entirely. for scale, `Lymphoid nodule` is 344 spots and the report already calls that sparse; `in situ` is half of it |

**What it would take:** re-annotation targeting in-situ regions, or a cohort where
DCIS is deliberately sampled. Not a compute problem, an annotation problem.

**Still worth stating in the writeup**, because the reasoning holds regardless of
whether the experiment runs: raw Virchow2 reaches 57.1% on the 5-class morphology
probe against a 30.7% majority-class floor, so morphology is encoded; and routing
predicts a decode-from-morphology-alone question goes to the **untrained**
projection rather than the contrastive runs, since B4 scores z=25.5 on the TLS
signature while every contrastive run falls below 1.6. Within a section, patient
identity is constant by construction, so the dominant confound of this project
disappears; CCA scored 0.667 within-subarray against 0.000 cross-subarray entirely
on that shortcut.

**For the writeup:** one sentence stating that stage is patient-level and therefore
not testable at niche resolution, while progression state is. Saying it closes the
question; omitting it invites a reviewer to ask for stage.

## 2026-08-04 · authority - a criterion declares where its warrant comes from

The EDA flagged its own weakest check in writing and the flag was lost in
propagation. `eda_biological_signal.ipynb` cell 22 states that the Moran's I
result is CN1/C1 only and that the cohort median is needed for the registered
report. `eda_summary.json` carries it as `risks[1]`. The gate read
`spatial_autocorrelation_pass: true` and passed it. **The defect is caveat loss
between the notebook and the gate, not a bad EDA.**

Every check now declares an `authority`, and the gate branches on it:

| authority | on this cohort | on a new cohort |
|---|---|---|
| `universal` | apply silently | apply silently |
| `cohort_calibrated` | evaluate, carry provenance | **escalate.** do not inherit, do not substitute a default |
| `advisory` | report | report. never contributes to a verdict |

Same discipline as the encoder prior in `decisions.md` B7: a locked default
escalates rather than silently substituting.

### where the numbers live, and where the warrant lives

`configs/eda_contract.json` keeps the thresholds as **defaults** and says, in
cohort-free terms, why a check's criterion is calibrated rather than universal.
The **warrant** - what was actually measured, on how many samples, by which
notebook - lives in `{project_root}/eda_calibration.json`, because the warrant is
precisely the thing a second cohort must not inherit. A cohort with no such file
has calibrated nothing, so every `cohort_calibrated` check escalates. That is the
correct default rather than a missing-file error.

### classification

| check | authority | basis |
|---|---|---|
| `cohort_counts` | universal | no threshold. a cohort with no units is not a cohort |
| `positive_markers` | cohort_calibrated | panel and detection floor both declared per project |
| `negative_markers` | cohort_calibrated | which genes count as absent is a property of the disease context |
| `spatial_autocorrelation` | cohort_calibrated | `I > 0.3`, `3 of 5` - **no external basis**, see below |
| `batch_structure` | cohort_calibrated | comparative half universal, the absolute floor is not |
| `encoder_input_decision` | universal | resolved-or-not is a property of the field; integer-ness a property of the matrix |
| `cross_modal_registration` | universal | boolean, no threshold |
| `model_tissue_fit` | universal | fixed category vocabulary, not a measured bar |
| `segmentation_qc` | universal | boolean; its platform minima are not in this contract |
| `multi_section_alignment` | universal | boolean |

`learned_checks` stay advisory and each now records the authority it **would**
carry on promotion - `platform_floor` and `label_granularity` to
`cohort_calibrated`, the other four to `universal` - so the classification is
settled before the promotion rather than during it.

**Provenance unknown is not a reason to promote.** `positive_markers`,
`negative_markers`, the batch floor and the label-granularity NMI cut all have
unstated sources; each is marked calibrated rather than assumed universal.

### the spatial autocorrelation threshold has no external basis

SVG literature selects by FDR-adjusted p or by rank, not by a raw magnitude bar.
Multi-sample practice aggregates ranks or combines p-values, never takes a median.
And the cohort-level question this check actually asks - *is this dataset
structured enough to justify a spatial encoder* - has no published precedent at
all. `0.3` and `3 of 5` are ours, defensible by argument only. That is exactly
what `cohort_calibrated` is for.

### the cohort recomputation, recorded not acted on

281 of 281 samples, streamed, memory-flat, 8.8 min.
`data/embeddings/_eda_cache/spatial_autocorrelation_cohort.json`.

| gene | cohort median | q25 | q75 | CN1/C1 | delta |
|---|---:|---:|---:|---:|---:|
| COL1A1 | 0.532 | 0.440 | 0.606 | 0.596 | -0.064 |
| ERBB2 | 0.363 | 0.262 | 0.469 | 0.431 | -0.069 |
| CD79A | 0.284 | 0.136 | 0.430 | 0.389 | -0.105 |
| CD37 | 0.173 | 0.091 | 0.318 | 0.323 | -0.150 |
| CD3E | 0.139 | 0.059 | 0.268 | 0.318 | -0.179 |
| CD3D | 0.165 | 0.076 | 0.280 | 0.495 | -0.330 |

**2 of 5 above threshold against 5 of 5 on CN1/C1.** CN1/C1 sits above the cohort
median on 9 of 10 genes, largest gaps on the immune markers. Structure is genuinely
present in stroma and tumour architecture; the immune markers collapse, which is
coherent for a cohort containing immune-cold cases.

This closes `eda_summary.json:risks[1]`, open since April. It does **not**
invalidate the molecular encoder routing, which had independent support under
`encoder_validation.novae` - spatial rho, effective rank, z-score tests. It
invalidates one justification.

**The verdict on tnbc-92 is unchanged.** `check_eda_gate` still returns
`proceed_with_caution` against a declared `proceed`, same three cautions, same six
advisories. The recomputation is recorded in the calibration file with its
provenance attached; adjudicating it is a human decision, and there is no external
standard to fall back to.

### escalation reuses the gate interrupt

No second mechanism. A `cohort_calibrated` check with no calibration record
becomes a caution, the verdict becomes `proceed_with_caution`, and the existing
`escalate` node's `interrupt()` fires - pauses hard, checkpoints, resumes at that
point, no default. The payload names what accepting forecloses, the same field
that carries the weight in the `route_question` tie escalation.

Verified on a scaffolded second cohort whose summary declares
`spatial_autocorrelation_pass: true`: the gate refuses to read that as a pass,
because the bar behind the boolean was calibrated somewhere else.

### surface

`omicstra gate` prints the authority per check plus an escalations block.
`check_eda_gate` returns `authority` and `provenance` per check and an
`escalations` list. `tests/` is new - 14 tests, including one asserting the
langgraph state schema is module-scope rather than leaving it as a comment.

---

## recorded 2026-08-04 · three findings, no code change

Each is a candidate contract entry. None is acted on today.

### F1 · k=6 is a hex-grid assumption on a square lattice - **confirmed**

Original ST (Ståhl 2016) is a cartesian lattice, not the hexagonal Visium packing
that k=6 comes from. Measured on CN1/C1, median distance to the j-th neighbour:

| j | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dist | 158.6 | 161.0 | 162.7 | 165.1 | 214.8 | 218.3 | 239.6 | 242.6 |

Four rook neighbours in a tight band at ~161, then four diagonals in two groups at
~216 and ~241. So `KDTree.query(k=7)` returns **4 rook plus exactly 2 of the 4
diagonals** - and it selects them by distance, consistently in the same two
directions across the whole array, which is a stronger statement than a tie-break:
it is a fixed directional anisotropy present in every weights matrix in the cohort.
The niche unit inherits it (centre + 6 neighbours, k matched deliberately).

**Do not change k.** It invalidates the niche unit and every downstream embedding.
k is platform-derived, so it belongs in the contract as `cohort_calibrated` when
the platform floor is promoted from advisory.

### F2 · the marker percentages are also n=1

`eda_biological_signal.ipynb` cell 13 loads `count_files[0]`; cell 16 reuses the
same frame. Every reported percentage - ESR1 2.6%, PGR 0.9%, MKI67 31.2%, CD3D
~8.9% - is CN1/C1, the subarray now known to sit above the cohort median on immune
markers. The triple-negative call is safe because the clinical table corroborates
it independently. The immune and proliferation percentages carry the same
optimistic bias the Moran's I did.

Recorded as a `scope` string on those values in `eda_calibration.json`. Not a
recompute.

**Sub-point, generalizable.** `% expressing = (vals > 0)` is computed on
log-normalised, per-gene batch-corrected floats. If the correction shifts
structural zeros off zero, the quantity is not a detection rate. This sits next to
the existing counts-are-raw check: **detection rates require raw integer counts.**

### F3 · the permutation p is a normal approximation

`_morans_i` runs 999 permutations, then discards the empirical distribution and
computes p from `norm.cdf` of a z against the permutation mean and variance. The
published critique of Moran's I calibration is specifically about normal
approximations. The empirical form costs nothing:

```
p = (1 + count(|I_perm| >= |I_obs|)) / (n_perm + 1)
```

This is `universal`, not cohort-derived. It also has leverage: the better criterion
- *fraction of samples where >= 3 of 5 markers reach FDR < 0.05* - needs per-sample
p-values, and `spatial_autocorrelation_streamed` currently stores only median,
quartiles and range. Retaining empirical p is what makes that criterion computable
later without reopening the methodology question now.

**Next change after the demo.** Not implemented today.

---

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
8. `docs/reports/internal/tnbc-92/index.html` stays **read-only** - spec and oracle.