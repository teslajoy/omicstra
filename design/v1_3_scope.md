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
