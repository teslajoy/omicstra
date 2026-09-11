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

---

## 2026-08-04 .. 2026-09-11 · the month this log missed

This file stopped on 2026-08-04 and the build did not. Reconstructed from the
commit history and a repo audit on 2026-09-11, because a build log with a
five-week hole is how the next session re-derives what was already known.

### the levels reorg  (`e34e24c` `dd9027e` `9178416` `2719564`)

The three-layer idea in `CLAUDE.md` was about the *loop* a cohort goes through.
What the code needed was a different axis - what a piece of code is **allowed to
do** - and conflating the two was why `eda_chains.py` and `eda_steps.py` kept
growing into each other. They are now separate words:

| level | is | rule |
|---|---|---|
| 0 `graph.py` | the only entry, the only checkpointer | `discover` picks the arm |
| 1 `graphs/` | one per gate | **a thing gets its own graph only if it can ask a human** |
| 2 `protocols/` | order, applicability, authority | no maths |
| - `measures/` | the maths | no ctx, no order, no authority. callable from a notebook |

`measures/` moving out of `protocols/` (`2719564`) is the load-bearing split:
computing is not ordering, and a pure function is the only kind a notebook can
borrow without inheriting the protocol layer.

### the port  (`6937b45` `94ae635` `b353f04`)

`align` and `eval` behind typed stages, **13/13 metrics at delta 0.00e+00**
against the published grid. `contracts/` reads a declaration and decides nothing.
Two modes on every stage, and the record says which ran: `compute=False` resolves
what a previous run produced and refuses when absent; `compute=True` invokes the
script. `resolves_only` on every record is what lets a reader of the ledger tell
them apart.

### v1.0.0  (`caa7f44`, tagged 2026-09-09)

Ships the read path - inventory, EDA, gate, routing - over stdio and HTTP, plus
the seed cohort's evidence pack, so a fresh clone routes without training
anything. Packaging findings worth keeping:

- **core deps are core.** `mcp`, `langgraph`, `langchain-core` and the sqlite
  checkpointer are imported unconditionally at module scope, so shipping them as
  extras meant `pip install omicstra` installed a server that could not import
  itself. The CI job's bare-install step is what caught it, and it goes first.
- **the sdist is an allowlist, not exclusions.** hatchling ships everything not
  gitignored, which meant a whole virtualenv, the notebooks and 74 MB of demo
  video - 192 MB for a few hundred KB of python. Exclusions rot; an allowlist
  cannot. The wheel is 107 KB.
- **`anthropic` is deliberately absent.** The server holds no model and makes no
  outbound call. Declaring it would put the model back inside the server.

### v1.1 opens  (`d6d532f` `a3ce73f` `6841b11` `f8ee99f` `bcd89de`)

Encoder registry, tile geometry declared, tiling maths lifted, the oracle
widened to five subarrays, and the canonical adapter. Each is in its own commit
message; what they share is that none of them asserted anything it could measure.

**The pitch correction had a second consequence, and it was found rather than
remembered.** `extract_virchow2_niche.py` sizes the crop as
`patch_px = 128 * nn_hd / pitch`, so the pitch is an **input to the crop size,
not a label on it**. At the measured 150 um the crop covers 95.9 um, not the
128 um the constant names - a half-width of 48.0 um against a 50 um spot radius,
so every spot was clipped ~2 um per side and the H&E arm under-sampled every spot
it represented. Ported **as built** and declared, because the ten-arm comparison
is fair only while every arm saw identical tiles; the 128 um tile is recorded as
a declared alternative arm, never a silent fix.

---

## 2026-09-11 · repo audit, and the eight findings

A full read of the repo after a session was lost. Recorded here so it is not
done a third time.

**State:** 148 tests pass in 1.6s. `selftest` passes 23/23. The gate returns
`proceed_with_caution` against a declared `proceed`. The read-path claim holds.

Eight findings, all agreed, six closed the same day:

| # | finding | state |
|---|---|---|
| 1 | `ingest.json` wrote absolute `/Users/...` paths | **closed** |
| 2 | `omicstra eda` printed `verdict: None` on the fixture | **closed** |
| 3 | five commits on one disk, no CI | branch added to `release.yml` |
| 4 | `__version__` 0.0.1 vs pyproject 1.0.0 | **closed** |
| 5 | doc drift: CLAUDE.md, `knowledge/`, this file | **closed** |
| 6 | `protocols/evaluate.py` six stubs | known, v1.1 step 5 |
| 7 | 449 MB `.git` | deferred - strip notebook outputs when next touched |
| 8 | duplicated clamp in `tiling.py` | **closed** |

### finding 1 · a record that is portable by construction

`rel()` is now the only way a path enters `ingest.json`, and a source outside the
repo root **raises** rather than falling back to an absolute path - the fallback
is what put the home directory there. Three tests: the rule, the call sites, and
the artifact.

**Relative and wrong is not an improvement on absolute**, so a fourth test joins
every recorded path back to the root and asserts it exists.

### finding 2 · the arm decided correctly and the command did not say so

`omicstra eda` on a cohort **with** an evidence pack routes to the ask arm and
never enters the eda subgraph. That is the arm working. Printing `verdict: None`
made it read as a broken gate. The fix is output, not a flag: the command now
names the arm, says the gate did not run and why, and points at `omicstra gate`.

### finding 5 · CLAUDE.md described an architecture that was never built

`src/agents/`, `src/tools/`, `src/models/encoders/`, `src/eval/` and a top-level
`config/` - **nine paths, none of which exist.** It is the first file a session
reads, so it said the wrong thing first, and every session since the reorg paid
for it. Rewritten against the filesystem. `knowledge/` is marked git-ignored,
which it has always been despite two docs calling it committed; only
`scripts/build_pathway_class.py` reads it, so the read path is unaffected.

`mcp_plan.md` gets a header saying it predates the reorg, and is **not** edited -
it is the reasoning that produced the design, not a description of the build.

### finding 7 · 449 MB of git, deferred with a date rather than a shrug

**Deferred 2026-09-11. Revisit when any of the three notebooks is next edited, or
before the first external clone is asked for - whichever comes first.**

`.git` is 449 MB. It is not the wheel (107 KB) and not the sdist, both of which
are allowlisted and clean. It is the clone, and the readme's claim is that **a
fresh clone routes** - which is true, and is a 449 MB proposition on whatever
connection the reader has.

Where it is, measured:

| MB | file |
|---:|---|
| 73.0 | `docs/media/om-demo.mp4` |
| 72.1 | `notebooks/final/05_summary_umaps.ipynb` |
| 63.3 | `notebooks/final/05_summary_umaps_v3.ipynb` |
| 60.3 | `notebooks/final/05_summary_umaps_v3.html` |
| 35.0 | `docs/papers/komen_2025_pathorob.pdf` |

The three notebooks are base64 cell outputs, not code. The fix is cheap per file
and costs nothing scientifically - the outputs re-render from the notebook - but
stripping them rewrites nothing already pushed, so it only shrinks **new**
history unless the repo is filtered, which is a separate and more disruptive
decision.

**Why it is written down rather than fixed now:** it costs clone time and nothing
else. No test, no result and no claim depends on it. Recording it with a trigger
is the difference between a deferred decision and one that quietly disappears
until someone on a slow connection finds it for us.

### finding 8 · one clamp, and the branch nothing covered

`tile_boxes` and `cut_tiles` each computed the crop geometry. Worse, one used
`np.rint` and the other `int(round(...))`, which agree **only because both round
half to even** - a coincidence of python and numpy sharing a rounding mode, not
a decision. Both now go through `tile_centres` and one clamp.

The dedupe touched code **no test exercised**: no tile in this cohort clips an
image edge, so the black-pad branch never runs on real data, and there was no
synthetic test either. Unreachable-here code is the most likely to rot, so it got
the same oracle treatment as everything else - the pre-refactor body is
reproduced verbatim in the test and diffed pixel for pixel across every clamped
case plus a half-pixel coordinate, on noise rather than a flat fill, because a
constant image hides an offset.

---

## 2026-09-11 · the canonical ingest runs - and two things it found

`scripts/ingest_wang.py` over the whole cohort. **280 samples, 286,250 spots.**

**Verified against the cache, which is the oracle.** Every ingested coordinate
set diffed against `virchow2_niche/*_meta.tsv` - the script's own output while it
built the cache: 280/280 samples, 0 row-count mismatches, **0 coordinate
mismatches**. The counts were checked independently against R on named cells,
total counts and both dims: exact.

### the 281 -> 280 drop had the wrong reason attached

`discover()` derived the patient from the HD image filename and skipped a sample
when it could not. So `CN32/D2` was dropped as "no patient" when the patient is
**known** - `TNBC64_CN32_D2.jpg` sits in `imagesLarge/` - and what is actually
missing is the **HD image the morphology path cuts tiles from**.

Those are different facts. The patient map now reads both pyramids, and the two
reasons are separate and **recorded in `ingest.json` rather than dropped in
silence**:

| reason | meaning |
|---|---|
| `no_patient` | no image at any resolution names it. the confounder axis would be a guess |
| `no_hd_image` | patient known, counts present, no HD tile source. half a sample |

A count falling from 281 to 280 with no reason attached is exactly the kind of
thing that gets rediscovered a year later. It matches the cached grid, which is
280.

### `counts_written` was about to lie

The field was set to `not coords_only`, and **nothing in the script wrote a
`.h5ad` at all.** A full run would have recorded counts it did not have. The
counts path is now implemented, and the field reports what was **written**, never
what was asked for. The record is also flushed per sample: an ingest interrupted
at sample 200 otherwise leaves 200 files and no record of them.

Three things the writer asserts rather than assumes, each a gate input:

- **counts stay integer.** `% expressing` is a detection rate only on raw counts -
  the check `eda_contract.json` calls universal. R stores them as double and they
  are integerish; the writer narrows the dtype and rounds nothing.
- **gene ids stay versioned Ensembl, unmapped.** Symbols are display. Mapping at
  ingest would bake one annotation release into the artifact.
- **counts rows are reindexed onto the coordinate order, never zipped to it.**
  They agree on this cohort. A silent zip that is right by luck would not survive
  a cohort where they disagree, and would be undetectable when it failed.

R hands the matrix over as MatrixMarket rather than CSV: `cnts` is 1075 x 27567
and 84% zero on the *smallest* sample, so dense CSV is ~30M numbers through a
pipe, 280 times. MatrixMarket carries the nonzeros only - 3.2 s per sample.

### the counts, in full

**280 samples, 286,250 spots, 796,858,360 nonzero counts, 2.0 GB.** Spot-checked
across the cohort: int32 throughout, versioned Ensembl ids, one patient per
sample, and `.h5ad` row order identical to the `_spots.parquet` beside it.

**19 of 280 subarrays fall below Novae's 512-spot prototype floor.** That figure
was measured months ago by a different route entirely and is reproduced here from
the fresh ingest through the `platform_floor` gate - two independent paths to the
same number, which is the closest thing to an external check this cohort has.

### recorded, not acted on · the gene set is not constant across samples

**min 10,571 · median 25,033 · max 33,047 genes per sample.** Not a narrow spread
around a fixed panel - a three-fold range. The per-sample `n_genes` is in
`ingest.json#counts`, so the cohort's gene universe is computable rather than
assumed.

**This lands on the niche join (v1.1 step 4)**, which needs a declared common
universe and a stated rule for a gene absent from a sample - dropped, or
zero-filled. Zero-filling an absent gene and observing a zero are not the same
statement, and at this spread the choice moves real numbers rather than edge
cases. The existing pathway work already fixes a universe
(`feedback_niche_ea_pipeline`), and the join should match that convention rather
than re-decide it.

The low end is worth a second look on its own: `TNBC34_CN17_D2` has 1,714 spots
but only 209,928 nonzeros - about 122 detected genes per spot against roughly
4,400 elsewhere. Whether that is a failed array or real biology is a QC question
the funnel should answer explicitly rather than letting it average away.

---

---

## 2026-09-11 · 3.5 · the encoder port, and what "reproduces" means for a transformer

`run_one_he(coords, image, geometry)` is the whole H&E arm for one sample with no
cohort in it, and the order is the published one rather than an implementation
detail: **tiles -> encode every tile -> pool over the niche.** Encoding first and
pooling second is what makes a niche a mean of encoded tiles rather than an
encoding of a mean tile. Those are different vectors and only one is in the cache.

### bit-identity was not available, and the diff said why before we guessed

First run, CN1/C1 against `virchow2_niche/`: max |Δ| 1.06e-04 on vectors of row
norm ~32.7. Not zero. The question is whether that is the port.

| comparison | max abs |
|---|---:|
| cpu vs mps, **same code, same machine** | 8.97e-05 |
| mps vs cache | 1.06e-04 |
| cpu vs cache | 1.05e-04 |

**Our port sits the same distance from the cache as the two backends sit from
each other**, so nothing is attributable to the port. MPS was separately checked
to be bit-deterministic run-to-run here, so this is not run noise either. Align
hit 0.00e+00 on 13/13 because it is deterministic linear algebra; a
631M-parameter float32 transformer across backend kernels is the TOLERANCE_RUNS
side of the split `v1_1_scope.md` already declared.

### the criterion, declared before it was met

```
max |Δ| / row norm      <= 1e-5       observed 3.2e-06
min cosine per row      >= 1 - 1e-6   observed 1 - 2e-07
top-6 neighbour set     >= 99.9% of rows
```

The third is the load-bearing one. H1 is a retrieval metric, so *same
neighbours* is the property the grid rests on, and it can hold **exactly** while
the vectors do not.

### the cache's device was never recorded - and the numbers identified it

The extraction script wrote no device and no versions, so "vs cache" looked
cross-device by necessity. It is not, quite. On CN1/C1:

| | top-6 set |
|---|---:|
| cpu vs cache | **0.9991** |
| cpu vs mps | 0.9991 |
| mps vs cache | 0.9981 |

Our CPU run sits as close to the cache as it sits to our own MPS run, and closer
than MPS does. **The cache behaves like a CPU build.** So the comparison is run
on CPU, and the criterion was never too tight - the BACKEND was mismatched.
Relaxing a threshold because a mismatched backend missed it would have buried
that. Recorded in `ACCEPTANCE.cache_device_inferred` as inferred, not declared.

### a tie is not a disagreement

Four of five subarrays passed on CPU, three at exactly **1.0000**. The failure
was `TNBC51_CN26_D1`, 8 spots, at 0.6250 - and the reason is not the port:

```
spot 0,1,2 -> niche {0,1,2,3,4,6,7}      identical membership
spot 3..7  -> niche {1,2,3,4,5,6,7}      identical membership
```

With 8 spots every niche overlaps every other, so the pooled vectors are
**duplicates** - two distinct vectors repeated 3 and 5 times, agreeing to
4.8e-07. The similarity gap between the 6th and 7th ranked neighbour is
**exactly 0.00e+00**. Which one falls inside the top-6 is decided by `argsort`,
not by the data.

So `retrieval_invariant` now counts a row as agreeing when the differing members
are tied at the k-th boundary, and reports `knn_set_exact` and
`knn_resolved_by_tie` beside it so the mechanism is visible rather than folded
away. This is a general statement about the measure, not a carve-out for one
subarray - and the guard is a test asserting a genuine reordering still fails.

**Result: 5/5 accepted on BOTH backends**, `knn_set_agreement` 1.0000
throughout, 0-5 rows per subarray resolved by tie. Written to
`projects/tnbc-92/encode_slice_diff.json` with device, torch, timm and the model
revision attached - the provenance the cache should have carried.

### what this cost, and what it bought

Three wrong turns, each caught by measurement rather than review: a cosine
metric that returned >1 (row norms averaged instead of per-row), an acceptance
built on `self_retrieval_at_1` that reports 0.25 for a correct port on 8 spots,
and a test bar tighter than the declared criterion. None survived contact with
real vectors, which is the argument for running the slice-diff before the
interrupt wiring rather than after.

## next

1. **3.4 `protocols/encode.py`** - the gate half landed 2026-09-11: the five
   preflight gates computed from declarations plus the inventory's counts,
   resolved where `cohort.json` answers them and returned open where it does not.
   Nothing here interrupts - `graphs/encode.py` is what turns an open request
   into an `interrupt()`, and is not built. `assert_clear` refuses compute while
   any gate is open and names both what would answer it and what accepting
   forecloses. **`ComputeRefused` is deliberately not `ComputeUnavailable`**:
   "nobody has answered yet" and "the artifact is not there" are different states
   and the ledger has to tell them apart.

   **The five are now declared, and tnbc-92 fires zero preflight gates.** That is
   the precondition step 8's `omicstra run` needs: a gate firing on this cohort
   would be asking a question the published grid already answered.

   `below_floor_policy: drop` is declared **as built**, not as a preference, and
   the evidence is recorded rather than asserted:
   `novae_niche_full/run_manifest.json` carries
   `skip_too_small_for_novae: 19` against 281 enumerated, 262 parquets written
   (241 ok + 21 pre-existing), and the niche join carries 259 subarrays. The
   preflight gate recomputes the same 19 from the fresh canonical ingest against
   the encoder's declared `min_units` of 512. **Two independent paths, one
   number**, and a test fails if they ever diverge - at which point the
   declaration has stopped describing what was run.

   `mark` is a declared alternative with `run: false`, the same shape as the
   128 um tile in `platform.json`.

### the closed option set was closed on one path only

Writing the declarations found it: the contract says a gate's options are a
**closed** set, and that was enforced where a person picks one - but a
declaration could put any string into `answer` and it would travel into the
ledger looking exactly like a human choice. A set closed on one path is not
closed.

`_resolve_option` now lands every declared answer on exactly one option.
A readable shorthand is allowed - `drop` for `drop_below_floor`, because a person
writes this file - but only when it identifies one option; ambiguous or unknown
raises and names what was offered. A near-miss silently ignored is how a cohort
ends up running a policy nobody chose.

Value gates are exempt and say why: a token source and an output path are
**supplied, not selected**, so their `options` describe the shape of the decision
rather than enumerating legal answers.

### the pattern, now that it has happened three times

| what | as built | declared alternative |
|---|---|---|
| H&E tile | 95.9 um (`tile_px_hd` 338) | 128 um, the first crop containing a whole spot |
| below-floor subarrays | `drop` - 19 removed | `mark` - keep and flag |
| join gene universe | whatever `build_niche_join.py` did | union vs intersect |

Three instances is a pattern rather than a habit: **port the choice as built,
name it in the declaration, and record the alternative as its own arm.** The
reason is always the same one - the published grid is a fair comparison only
because every arm saw identical inputs, so changing an input retires the oracle
and every number that rests on it.

**Step 4's job for the join is therefore fixed in advance:** port what the script
did and name it in the manifest as `gene_handling: <what it did>`. The
union-vs-intersect question becomes the third declared alternative rather than a
decision made during the port. That is settled now so it is not reopened halfway.
2. **3.5 the compute path and the slice-diff** - `run_one(sid)` for H&E on the
   mac, then diff against the 75 GB cache on the five oracle subarrays. Tests our
   port rather than testing Virchow2, and runs in minutes. The same method proved
   the align port at delta 0.00e+00.

   **DONE 2026-09-11** - see the 3.5 section above. Reordered before
   `graphs/encode.py`, and the reorder paid: three wrong turns in the criterion
   were found by real vectors, and every one of them would have been carried into
   the interrupt work as a settled assumption.

   **Reordered 2026-09-11: 3.5 before `graphs/encode.py`.** The interrupt wiring
   is a known shape - it is the eda gate again - and the gates are already
   computed and proven silent on a declared cohort. The slice-diff is the step
   that can still fail for a reason nobody has seen: a real HD image, real
   Virchow2 on real tissue, diffed against `virchow2_niche/`. If it does not
   match, the interrupt work waits anyway, so it earns nothing by going first.
   Order is now 3.5 -> temporal (3.6) -> `graphs/encode.py` (3.7).
3. **3.6 temporal**, then **3.7 `graphs/encode.py`** - the interrupt half. tnbc-92
   firing zero gates is 3.7's fixture: it is what lets `omicstra run` complete
   unattended, and the test for it is already written.
4. **step 4, the niche join** - port `gene_handling` as built (above), union vs
   intersect becomes the third declared alternative
5. then step 5, `evaluate` as a chain: the six guards are still stubs

### still open from the 2026-08-04 list, carried forward

Three of that list closed in the meantime and are dropped: `omicstra init` now
scaffolds a real project home, `MODALITY_TEMPLATE.md` exists, and `configs/`
moved inside the wheel at `src/omicstra/configs/`. What remains:

- **grow the contract** from 10 checks to what the notebooks declare - roughly
  40-60 real decisions. `MIN_MEDIAN_GENES` 500 · `NOVAE_MIN_SPOTS` 512 ·
  `RHO_THRESHOLD` -0.1 · the 3-way encoder routing · `MAD_TOP_FRACTION` ·
  `MIN_NICHE_UMI` · `MIN_PATHWAY_SIZE/MAX` · `FDR_STRICT` · `OR_CUT` · IQR
  outlier flags · gene-panel intersection · effective rank · clinical
  missingness. Still the highest-value work left in the EDA layer.
- **generated-step conventions** - `{project_dir}/steps/s{NN}_{modality}_{step}.py`
  plus `MANIFEST.json`; header carries `contract@sha256` + `trace_id`; commit
  message carries the trace id. The contract sha makes staleness computable.
- **first generated step** - `s05_st_spatial_autocorr`: its output is an edge
  (it routes the encoder branch) and it is where the n=1 defect lives.
- **remaining EDA gaps**: input-provenance per check · itemised funnel · platform
  floor · 3-way encoder routing · label-granularity NMI · annotation coverage.
- `docs/reports/internal/tnbc-92/index.html` stays **read-only** - spec and oracle.
- **the spatial-autocorrelation recomputation is recorded and unadjudicated.**
  2 of 5 markers above threshold cohort-wide against 5 of 5 on CN1/C1. It closed
  `eda_summary.json:risks[1]` and invalidates one justification, not the routing.
  There is no external standard to fall back to, so a human decides.
