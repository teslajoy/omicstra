# 1.2 scope

> **1.2 = a second cohort goes from files to its own evidence pack and report
> with no package edit.**

read this before adding anything. the line above ends a scope argument; a
judgement about whether the work is good does not.

1.1 shipped the chain on the cohort it was written against. that proves the
chain runs. it does not prove the chain is a package, because a pipeline shaped
around one dataset passes its own tests every time. the only test that
discriminates is a cohort whose declarations someone else wrote.

---

## what counts as a package edit

nothing in `src/omicstra` changes to accommodate the second cohort. permitted
instead:

| where | what |
|:--|:--|
| `scripts/ingest_<cohort>.py` | the cohort's native format, converted once |
| `projects/<cohort>/*.json` | platform, project, cohort declarations |
| a gate answer | recorded, with an actor |

a change inside the package is not forbidden - it is a FINDING, and it means the
thing being changed was cohort knowledge wearing a package's clothes. record it
as that rather than as a fix.

---

## what this release has been

the theme, stated after the fact because it was not written down first: replacing
declared numbers with measured ones, so that a cohort nobody validated against
can be trusted or refused on evidence.

- the lattice is measured, and a declaration it contradicts escalates
- a platform may answer that disagreement in advance, once, with a reason -
  rather than a standing override that silently wins every run
- a run's cost is estimated from measured throughput and footprint, per device
- the capacity gate measures free space instead of asserting it
- a peak-memory figure that had never been measured was corrected

the pattern underneath: a number the code trusts must say where it came from.

---

## the geometry result, which is the reason for the line at the top

"original ST" is not one geometry.

| cohort | declared | measured |
|:--|:--|:--|
| seed | 150 um | the lattice, corrected 2026-08-12 from 200 |
| second | 200 um | 191.3 um median over 108 samples, 178.0-197.7 |

both are real arrays of the same nominal platform. a pipeline that trusted
`platform == "original_st"` to imply a pitch would be ~30% wrong on niche
footprint here - the same class of error as the seed cohort's own correction,
which threw every micron-denominated figure out by a third.

the control matters as much as the finding: Visium returned 100.0 against a
declared 100, and Xenium refused to derive a pitch at all rather than test a
declaration against itself. the measurement is right, and it declines where it
would be circular.

---

## in scope

```
a  ingest + inventory + geometry on the second cohort          done
b  the encoder-fit relation: what unit an encoder was built
   for, against what unit a cohort supplies                    <- here
c  drive the existing surface on the second cohort. the CLI
   and the server are two hosts on ONE graph, so a script
   around them would test nothing
d  its EDA gate, evidence pack, report
e  the report's second target: stepwise stats, figures, and
   the trace that produced each decision
```

## out

a third cohort. slurm. oauth. anything that makes the first cohort better.

---

## the eda chain's unit

> **the eda chain profiles a cohort the way the notebooks did - per sample
> through the canonical adapter, aggregated by a rule the contract declares.
> no single-object path.**

the seed cohort's `eda_summary.json` records `produced_by: manual`, from two
notebooks that iterate per sample and read each one's own files. there is no
concatenated object anywhere in them.

the graph path was written later to replicate that, and encoded a shape the
notebooks never used - one object, reached through `adata_path`. nothing
produces `adata_path`; no inventory step declares it; the "inventory A7 step"
cited in two docstrings does not exist. so the eda chain has never run through
the graph on any cohort, and the seed cohort hid it by always taking the ask arm.

the package is already split on this, which is the evidence:

| | shape |
|:--|:--|
| `protocols/eda.py` | 5 steps read `ctx["adata_path"]` - one object |
| `measures.spatial_autocorrelation_streamed` | takes `load_sample_fn` + ids, iterates |

cohort-level checks aggregate per-sample results plus declarations - the subject
column, the clinical table - not a stacked matrix. so nothing needs assembling.

### what changes

- each check declares its `unit`: `section` or `cohort`
- a `section` check declares its aggregation rule. default: worst status wins,
  and the per-sample rows stay in the record
- the five steps take the adapter-injected sample list, not a path
- `adata_path` and the A7 references go. **no override flag** - a flag for a
  shape the design rejected is how the shape comes back
- a lint holds it: no module in `protocols/` may read a single-object path, the
  same way the AST lint keeps format readers out of the package

### acceptance is the seed cohort, and it is not "reproduce the file"

`eda_summary.json` was assembled by hand, so it is a record rather than an
oracle. four cases, named before starting so none is discovered mid-port:

| checks | target |
|:--|:--|
| positive_markers, negative_markers, spatial_autocorrelation, cross_modal_registration | boolean pass in the summary - must match exactly |
| encoder_input_decision, model_tissue_fit | prose and a dict - compare content, not status |
| segmentation_qc, multi_section_alignment | recorded as null. the graph must reach not_run or not_applicable, never invent a pass |
| cohort_counts, batch_structure | absent from the summary. NOTHING to diff, and cohort_counts is universal authority - a check every cohort must answer that the seed cohort never did. recorded as a finding, not quietly filled in |

a check that disagrees is named with its reason. nothing is adjusted to make it
match.
