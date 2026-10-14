# 1.3 scope

> **a cohort the package had never seen goes from files to ITS OWN EVIDENCE PACK
> AND REPORT, through the same surface a client uses - and the pack states which
> rows it cannot fill and why.**

this is 1.2's opening line, unchanged. 1.2 shipped the first two thirds of it:
files to embeddings through the surface, for a cohort with no evidence of its
own. the pack is the third, and it is what makes the claim worth making - a
second cohort that embeds proves the adapter; a second cohort that produces its
own evidence and names its own refusals proves the product.

the clause after the dash is new and is the honest part. the second cohort
brought no spot labels and no pathway ground truth, so most rows cannot be
filled. a pack that omitted them would read as an incomplete analysis; a pack
that states them as refusals with their cause is the result.

---

## what 1.2 established that this builds on

| | |
|:--|:--|
| both modality agents reachable by a client | `he_encode`, `st_encode`, each with its own gate set |
| a run handle that survives the process | sqlite checkpointer, kill-and-resume on both executors |
| the ST port is exact | five subarrays byte-identical to the Novae cache |
| the EDA chain runs | first execution on any cohort; its table is measured |
| every refusal is named | 7 task families refuse on the second cohort, each with a cause |

---

## in scope

```
4  downstream reachable   align.run / evaluate.run behind the same run handle.
                          blocked in 1.2 by the EDA gate, correctly - this repo's
                          first constraint is that nothing runs past a failed gate
6b eda, the remainder     the 4 undeclared checks and the 3 unimplemented ones
5  the pack               promote + report for the second cohort. THE release
                          claim; everything else is in service of it
7  pathway agent          a package path for gpath2vec
```

## out

a third cohort. slurm. oauth. anything that makes the first cohort better.

---

## the build order inside 1.3

five items, in the order their dependencies allow. the state column is what has
been demonstrated, not what has been written.

| | | state |
|:--|:--|:--|
| A | `omicstra worker` - the process that serves the encode queue | **done** |
| B | `evaluate_run` behind the run handle, and `hypotheses` leaves `align_run` | **done** |
| C | one record store, and the records reach the report | **done** |
| D | level 0 wires align and evaluate; `design/graph.md` is the route | |
| E | Bedrock, as a test rather than a build | |

### A, and what it closed

submit-only was half a claim. `he_encode` and `st_encode` hand the shard list to
the scheduler and return at submission, which is only durable if some process is
polling - and until now that process existed solely inside the kill test, as a
worker the test assembled itself. so the test could pass while the command an
operator would run did not exist.

`omicstra worker` is that command. it holds both activity bodies by name, needs
no cohort and no project root, and stops gracefully on SIGTERM so the shard in
flight reports rather than timing out into a retry. `--max-seconds` exists for a
worker inside a job allocation with a wall clock, which is the ARC case.

two rules came with it, because a name crossing a scheduler is checked by nothing
in the language:

```
dispatch/bodies.py:shard_bodies()     the one name table. the submitter names a
                                      body, a different process holds it
test_rule_lints.py                    every fn_name= in graphs/ is a key in it.
                                      an arm that forgets the line fails a test
                                      rather than orphaning runs on a queue no
                                      worker can serve
```

the acceptance is the existing durable test, with the shipped entrypoint in place
of its inline worker: the submitter exits having encoded nothing, a worker it
never met finishes all four shards, and the handle still resolves from a third
process.

still untested beyond localhost - the kill test runs a dev server in-process, and
the `Temporal beyond localhost` line below is unchanged.

---

### B, and what it found

`align_run` accepted a `hypotheses` list and ignored it. the align graph resolves
a grid and scores nothing, so naming a hypothesis changed no behaviour while
implying the stage had returned a verdict over it. the parameter is gone and the
verdict has a stage of its own.

`evaluate_run` is judge #2's surface. it reads the artifacts the cohort's
`evidence_sources.json` names, runs the guards `routing_contract.json` requires
per task family, and states one of four things per hypothesis. no model is called
anywhere on the path, so `modelInsideServer: false` still holds.

three gates, all declared in `compute_contract.json` under `stage: evaluate`:

```
hypothesis_scope   which hypotheses this run answers. a hypothesis whose every
                   family is undeclared or declared unavailable is NOT a gate -
                   nothing to choose - and is reported not_run with its families
artifact_pinning   the sha256 of every cited artifact, and the hashes travel
                   WITH the question. a declared pin that does not match is a
                   REFUSAL: the pack's numbers were read from other bytes
guard_failure      a required guard that RAN and failed. demote, do not discard
```

the verdict is computed BEFORE the gates, which is the opposite of the encode
arm. `guard_failure` can only fire on a guard that ran, and the guards cost a few
file reads rather than a training run, so there is nothing to protect by
deferring.

**four claim states, and the fourth is the finding.** `biological`,
`diagnostic`, `not_run` - and `unguarded`, which had to be added because "no
guard failed" was true of a hypothesis where no guard ran at all, and read as a
clean bill of health for a number nothing checked. on the seed cohort:

| | claim | why |
|:--|:--|:--|
| H1 | `unguarded` | `cross_subject_eval` needs a within-subject score to compare against; nothing in the declaration carries one |
| H2 | `diagnostic` | H2-C's G3 ran on all 7 arms in `metrics_h2_ci.json` and **7 of 7 failed** - no arm suppresses patient identity below biology |
| H3 | `unguarded` | the pathway gate needs each arm's ST features, which is per-arm cohort knowledge and is declared, never guessed. `evidence_sources.json#run_st_features` is the pointer and the seed cohort does not carry it |

15 cited artifacts, all present, **none pinned**. so the pin gate asks on the one
cohort that has a published grid, which is the right answer: the pins were never
written down. answering `pin_these_bytes` records the 15 hashes as a PROPOSAL in
the ledger - the package does not write a cohort's declaration, because a pin is
a committed human act.

two artifacts the pack's prose names and nothing locates: `biology.parquet` (it
is per arm) and `perm_nulls.parquet` (a sibling). a bare filename is reported
rather than resolved against whatever directory appeared nearby in the same
sentence - that guess gets the shared file right and the per-arm one wrong.

**two gaps the contract now states rather than hides.** it requires seven guards;
this package implements six checks; the overlap is five.
`routing_contract.json#guard_implementations` declares the map with a reason per
guard: `raw_floor` is each task's `floor` field, computed by promote from the
cohort's declared floor reader, and a guard that re-derived it would be a second
answer to one question; `untrained_control` is an ARM of the grid, not a check
over a number, and its absence from a rollup is what `coverage()` reports.

### three things B did not close

```
the rollup coverage gaps     H2 drops B4_v3 and H3 drops R5_v3, which the
                             verdict now REPORTS per hypothesis. closing them
                             means scoring those arms, which is a compute run
run_st_features              one declaration away from H3 being guarded rather
                             than unguarded. it is the cohort's to write
no reader for four families  cross_modal_retrieval, tissue_state_grouping,
                             morphology_decode and pathway_discrimination report
                             `no_reader`: promote computes their candidates from
                             the same declaration, and what is missing is a
                             CHECK over the number, not the number
```

---

### C, and the three divergences it found

there were two record stores. the routing path appended to a file-backed
`RecordStore` at `runs/{id}/records/decisions.jsonl`; every gate arm wrote plain
dicts into the langgraph checkpoint, readable only through `runs_record(run_id)`
and only by a caller holding the handle. so a cohort's ledger said "N routing
decisions" while the gates, the dispatches and the pins were elsewhere - and
`report.py`, whose own first line is *"a pack and a ledger in, markdown out"*,
read the pack and no ledger at all. `section_provenance` was already printing
`len(records)`; nothing ever supplied the key.

the arms' dicts could not be VALIDATED as Records, which is why nobody had tried:

```
actor      records.py said deterministic | model | human, and the gate nodes
           have been writing `client` and `system` since the actor vocabulary
           was closed. `actor_ratio()` initialised three keys and did
           out[r.actor] += 1, so the first client record would have raised
kind       no `dispatch`. handing N shards to an executor produces no artifact
           and judges nothing, so it is neither a transform nor a diagnostic
chosen     a str. a gate arm answers every open gate in ONE call, which is a
           mapping, and flattening it loses which answer belonged to which gate
```

three divergences between a model and the only code writing to it, none of them
caught, because nothing read both. that is the whole argument for one store.

what now holds it: `records.coerce()` is the door - unknown keys land in
`params`, which is the field that exists for them, and `record_id` is content
addressed over the decision with the clock excluded, the same exclusions
`stable_id()` already stated. `RecordStore.append_new()` is idempotent on that
id, which matters because the arms carry their whole record list forward in
state: without it a three-gate run resumed twice would read as nine decisions,
and the actor ratio is a published number.

`_persist` is called from `_handle`, the one place every arm and every resume
passes through, so there is one sink rather than a call per node. the handle
returns `recorded: N`.

**two views, not two stores.** `runs_record(run_id)` still reads the run's
checkpoint, because "what did THIS run decide" and "what has this cohort
decided" are different questions - and it now also names the cohort's ledger and
its size, so a reader is not left thinking the handle is the only way in.

---

## step 6's remainder, split by who can close it

| | check | what is missing | whose |
|:--|:--|:--|:--|
| declaration | `positive_markers` | a panel as `{gene: why}` | cohort |
| declaration | `negative_markers` | same, and whether `ERBB2` is a negative at all - the notebook calls it "low-moderate" and 6 patients are HER2 1+ | cohort |
| declaration | `spatial_autocorrelation` | a gene list, and the `I > 0.3` threshold's basis | cohort |
| declaration | `batch_structure` | FOUR: a gene universe to stack at all, then `sample_meta`, `technical`, `biological` | cohort |
| **package** | `cross_modal_registration` | declared in the contract, **no implementation** | package |
| **package** | `model_tissue_fit` | same | package |
| **package** | `multi_section_alignment` | same | package |

the three unimplemented checks are the ones with teeth. the seed cohort hid them
because every value in its `eda_summary.json` is hand-written and the file says so
(`produced_by: manual`). a second cohort has nobody to hand-write them, so they
are permanently unanswerable there unless something is built. deciding whether to
build them or to state them as structurally unanswerable is a 1.3 scope call and
is on the critical path to the pack.

---

## the level rule, restated - step 4 forced it

the rule said two things that step 4 put in conflict:

```
align and evaluate are LEVEL 2        protocols: order, applicability, authority,
                                      no maths, NO INTERRUPT
anything that can interrupt()         gets its own LEVEL 1 graph
```

`graphs/align.py` interrupts, so the second clause admits it and the first
forbids it. the resolution, and the reason:

**the gate is level 1; the protocol stays level 2.** `protocols/align.py` gained
`preflight_align`, which is deterministic, side-effect free and asks nobody - it
computes a gate list from a recorded verdict, exactly as `protocols/encode.py`
does. `graphs/align.py` holds the interrupt and nothing else. so align is level 2
and *has* a level-1 gate, which is what encode already was and what the rule
meant.

what would have violated it: putting the verdict decision inside
`protocols/align.py`, or letting the graph compute a metric. neither happened.

this is a restatement rather than a new layer, so the 2 sep stop condition holds:
no file moved level, and `graphs/` gained a module for a gate that already
existed as a sentence in CLAUDE.md.

### and `align_run` does not align

it RESOLVES. `compute=True` is accepted and reaches `run_align`, but alignment
has never executed through the package - the scripts are still the method, and
the 21 sep status on that is unchanged. the tool's description says so in its
first sentence. the claim "align runs through the surface" waits for a
`compute=True` run that reproduces a trained arm, which is 1.3 work and is not
done.

---

## open questions carried forward, with their evidence

### the spatial check thresholds a raw statistic and discards its own null

`measures.spatial_autocorrelation` computes Moran's I with a permutation null per
gene per section, keeps only `I`, and passes when 3 of the top 5 genes exceed
`I > 0.3`. the `z` and `p` it computed are thrown away, so `n_perm` changes the
runtime and nothing else.

this is the check whose own docstring calls it *"THE decision that changes the
pipeline's shape: it routes the molecular encoder to a spatial or a non-spatial
model"*, and `0.3` is the exploratory notebook's eyeballed cut-off. the project's
framing is that every number has a null; this one has a null and ignores it.
either the criterion should use the `z` already computed, or `n_perm` should go,
so nobody reads a null into the verdict.

### `negative_markers` discards 278 measurements because of 2

two sections carry neither `ESR1` nor `PGR`: `TNBC34_CN17_D2` (20,159 genes) and
`TNBC51_CN26_D1` (10,571 - the thinnest in the cohort). one unresolvable section
raises, and the aggregation turns that into `not_run` for the whole cohort check.
per-section `not_applicable` with the other 278 aggregated is the alternative.
the contract declares an aggregation rule per check, so this is a contract
decision.

### the mitochondrial-fraction exclusion is not implemented

the proposal lists *"ST spots with <200 UMI or mitochondrial fraction >25%"* as an
exclusion to be reported with counts. nothing in the package computes a
mitochondrial fraction. the two cohorts are asymmetric about it: the second ships
`pct_counts_mito` precomputed in `obs`, and the seed cohort's canonical `.h5ad`
carries no QC columns at all.

### the seed cohort's verdict disagreement, investigated

the align gate surfaced that tnbc-92 computes `proceed_with_caution` while its
summary declares `proceed`. the pack shipped under `proceed`, so this needed
checking rather than noting. from `run_gate('tnbc-92')`, 0 violations, 0
escalations, 3 cautions:

```
1  segmentation_qc: null          "cannot distinguish 'not applicable' from
2  multi_section_alignment: null   'not measured'; EDA.md schema has no marker"
3  risks[]: 4 entry(s) recorded and not marked accepted
```

**no biology hides here.** 1 and 2 are a SCHEMA gap - the summary has no way to
say "not applicable" - and the package's own graph already resolves what the
schema cannot express, reaching `not_applicable` on `segmentation_qc` with "no
segmentation masks - spot-resolution platform". 3 is the four `open_risks` the
summary itself records: Novae compatibility on this platform, Moran's I computed
on one subarray, MKI67's low I, and 17 subarrays under 500 spots. all four were
carried deliberately.

so the declared `proceed` is a person's judgement that those were acceptable,
and the computed `proceed_with_caution` is the absence of a marker for that
judgement. **two closable things follow:** give `EDA.md`'s schema a
not-applicable marker, and mark the four risks accepted with an actor. neither
changes a number.

### the gene universe, measured before declaring

| | sections | per section | intersection | union |
|:--|:--|:--|:--|:--|
| seed | 280 | 10,571 - 33,047 (median 25,041) | **7,288** | 54,714 |
| second, above the ST floor | 25 | 14,861 - 18,853 (median 17,627) | **0** raw / **11,284** mapped | 21,523 |

the second cohort's raw intersection is zero because 17 of those sections are on
Ensembl and 8 on symbols. so a universe rule has to be stated over **the
encoder's axis**, not over the sections' own `var_names`, or it is undefined on a
pooled cohort. union-with-zeros is the option to reject explicitly: a gene absent
from a section was not measured there, and a zero would read as a measurement.

---

## the three funded biological deliverables, audited against artifacts

the proposal names three outputs "beyond the reusable software infrastructure".
audited 2026-10-12 by opening each artifact. absences are stated, not explained.

| # | funded deliverable | state | artifact |
|:--|:--|:--|:--|
| 1 | integrated multi-modal cohort characterisation | **partial** | retrieval half: `runs/tnbc-92_v3/eval/H1/summary.json`, cited. comparative half: `eval/H2/mc_coherence.parquet` rows `raw_he` / `raw_st`, uncited |
| 2 | TME readability profiles | **partial** | `eval/H2/compartment_cosine.parquet` (108 = 12 compartments x 9 runs), `compartment_welch_contrasts.parquet` (54 = 6 contrasts x 9 runs). computed, uncited, and the prediction is inverted |
| 3 | pathway-morphology spatial maps | **partial** | association: `runs/tnbc-92/eval/H3/pathway_cca/dag_full/compartment_table.parquet` (395 nodes x 7 compartments), uncited. **no map in tissue space exists** |

**the deliverable-table rule:** every row names an artifact path or says "no home".
a row whose state is `built` must also be cited by the evidence pack - an artifact
nobody cites is not a delivered result. no row is `built`.

### 1 is broader than retrieval, which is why it is partial

verbatim: *"the shared embedding space combines morphological and transcriptomic
signals into a joint representation that is **more informative for TME
characterization than either modality alone**, enabling cross-modal retrieval
that links tissue architecture to molecular state across patients."*

retrieval is the second clause and is what the first ENABLES. the primary claim
is comparative, and the comparison exists:

```
                     ARI      silhouette
raw_he             0.1641       -0.0058
raw_st             0.0458       -0.0724
best aligned R5    0.2458       -0.0139
```

against the STRONGER single modality the gain is +50%, not the 5x against
`raw_st` quoted earlier in this session - the proposal says "either modality
alone", so `raw_he` is the bar. and on silhouette, which the proposal asks for
alongside ARI, raw H&E is **better** than every aligned run. neither number is
cited by the pack.

### 2's prediction is inverted, with the numbers

the proposal predicts *"morphologically distinct compartments (ex. tumor, TLS,
necrosis) will show higher matched-pair cosine similarity than morphologically
ambiguous compartments (ex. high-TIL vs. low-TIL stroma)"*. on R4_v3:

| distinct | ambiguous | mean distinct | mean ambiguous | n dist / n amb | z | direction |
|:--|:--|--:|--:|--:|--:|:--|
| Tumor | High TIL stroma | 0.5785 | 0.6175 | 3659 / 76 | -3.26 | **INVERTED** |
| Tumor | Low TIL stroma | 0.5785 | 0.6386 | 3659 / 268 | -15.41 | **INVERTED** |
| Lymphoid nodule | High TIL stroma | 0.6303 | 0.6175 | 24 / 76 | +0.68 | ns |
| Lymphoid nodule | Low TIL stroma | 0.6303 | 0.6386 | 24 / 268 | -0.55 | ns |
| Necrosis | High TIL stroma | 0.5947 | 0.6175 | 1018 / 76 | -1.89 | ns |
| Necrosis | Low TIL stroma | 0.5947 | 0.6386 | 1018 / 268 | -10.30 | **INVERTED** |

**zero of six support it. three inverted, three underpowered** - Lymphoid nodule
is 24 niches and High TIL stroma 76. the `direction` column has read INVERTED
since the parquet was written in May and has never been read into a claim.

**coverage.** `morphology_labels.tsv` carries `dominant_18class` with 18 `frac_`
columns; `compartment_cosine` covers 12. six absent: `Nothing`, `Artefacts`,
`Hole (whitespace)` are non-tissue, and `Tumor region`, `Nerve`, `Heterologous
elements` are tissue categories with no row. the proposal says 17 categories and
the artifact has 18; neither number is 12.

**the missing arm is B4_v3** - the untrained random-init control, 9 of 10 runs
present. that is the arm that WINS H2-B.2 at z=25.5, so the readability profile
omits the only run that beat every trained one on a related readout.

**cross-modal cosine is not a signal-to-noise ratio.** the proposal commits to
"per-compartment signal-to-noise analysis"; what exists is mean matched-pair
cosine per compartment with a Welch contrast. that is a different statistic and
goes in `projects/tnbc-92/proposal_deviations.md` as a substitution, not a
rename.

### 3 is planned, not built

the map belongs to `report.py`, not to a script: the renderer already takes a
pack and a ledger and emits markdown, and a figure a script writes separately is
a figure the pack cannot cite. the plan, deliberately one section and one
pathway:

```
input   compartment_table.parquet (395 nodes x 7 compartments)
        + projects/tnbc-92/data/canonical/{sid}_spots.parquet for x, y
scope   ONE pre-registered pathway of the five, ONE section, named in the
        pack rather than chosen at render time
output  a tissue-space panel the report cites by path, next to the
        compartment row it illustrates
not     a per-pathway atlas. the deliverable is "spatial maps generated showing
        pathway-morphology correspondence", and one correspondence shown
        honestly is that; 395 panels is a different artifact
```

---

## untested, and listed so it is not mistaken for working

```
embedding extraction on genuinely new data    the second cohort ran 3 sections
                                              of each arm. 3 is a smoke test, not
                                              a cohort run. the 25 above-floor
                                              original-ST sections have never
                                              been embedded, and nothing has run
                                              on a cohort outside these two
AWS / Bedrock                                 never attempted. the routing layer
                                              holds no model, so a Bedrock-backed
                                              client is a client-side swap in
                                              principle - untested in fact
ARC (the OHSU cluster)                        never attempted. COMPUTE_BACKEND
                                              declares slurm and the durable
                                              executor works on Temporal locally;
                                              neither has run on ARC. the
                                              interpreter there is unknown, and
                                              the clone verifier now pins the CI
                                              matrix precisely because a local
                                              interpreter drifted to 3.15
Temporal beyond localhost                     the kill test runs a dev server
                                              in-process. no remote cluster
```

these are listed rather than planned. each would change what the release can
claim, and none is claimed now.

---

## grant commitments outstanding, not release work

```
Zenodo deposition        embeddings and the niche join. a data-sharing commitment
                         in the proposal and the only unchecked box in the
                         README's deliverables. the checkpoint and PyPI are done
H3 spatial maps          one of the three named biological deliverables. the CCA
                         landed, the maps did not
registration             declared, not built. both cohorts are same-section, so
                         nothing here needs it; a PDAC cohort with serial sections would
```

these are tracked so they are neither forgotten nor pulled into a release.

---

## what 1.2 cost in package edits, by the kind its own rule names

the rule: a change inside the package is a FINDING, and it is one of two kinds.

**cohort knowledge wearing a package's clothes** - none. the thing the rule exists
to catch did not occur.

**a path no cohort had taken** - all of them:

```
the compute arm had never run           hidden by the ask arm for a month
the encode node never dispatched        gated a run that never happened
a gate answered on the run reopened     a resume could never finish
the workflow id used hash(str)          salted per process; a resume could never
                                        find its own run
a failed shard read as still running    the in-process ShardReport was dropped
the floor gate measured nothing         no unit counts reached it; it read as a pass
a declaration could not reach its step  nothing filled params_space from a cohort
an empty marker panel passed            0 == 0
a panel could not meet its own matrix   symbols against versioned Ensembl
the gene axis was declared per cohort   it is a per-section fact the ingest observes
```

the last one is the only edit that began as a mistake of mine rather than a gap:
I asked the cohort to declare what the ingest already recorded, and the claim was
wrong for 68 of 108 sections of a cohort that pools three studies.
