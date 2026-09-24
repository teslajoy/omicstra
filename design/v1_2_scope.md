# 1.2 scope

> **a second cohort goes from files to its own pack and report THROUGH THE SAME
> SURFACE A CLIENT USES. onboarding the cohort touches nothing in `src/` -
> anything it forces there is recorded as a finding.**

read this before adding anything. the line above ends a scope argument; a
judgement about whether the work is good does not.

the second half is the falsifiable half, and it is about the COHORT, not the
build. steps that expose an agent or add a graph are package work and are the
product being built. the rule applies at the moment a cohort comes through:
once the surface exists, a new cohort should be its declarations plus an ingest
script. when it forces a change in `src/` anyway, one of two things is true -

  the package held cohort knowledge     a seed-cohort assumption. the bug this
                                        rule exists to catch
  the package held a path no cohort     a finding, and a legitimate edit. the
  had taken                             compute arm was hidden by the ask arm
                                        for a month

both go in "open, found" below with which kind they were.

the first half is the one that changed on 24 sep. the funded proposal puts
"MCP server implementation; agent orchestration; embedding extraction
pipelines" on ONE line as ONE deliverable, and lists modality-specialised
agents with persistent context and client-callable MCP interfaces as items (i)
and (vi) of one system. extraction is what the agents DO. a script that calls
the package proves the package; it does not prove the product, and three
cohort-specific lines make it a notebook.

duration is not the objection it was taken for. the proposal answers it in the
design - human-in-the-loop escalation at confidence boundaries, persistent
execution context, traces - which is `interrupt()`, the checkpointer and the
durable executor. all three are built. none is reachable from a client.

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

### open, found by the acceptance run - not worked on

- **marker panels are undeclared.** `positive_markers`, `negative_markers` and
  `spatial_autocorrelation` each need a panel naming genes expected in this
  tissue. the exploratory notebook held one inline; the calibration record says
  it is declared in `project.json`, where it is not. all three now refuse with a
  reason rather than raising.
- **the gene axis is undeclared.** sections span 10,571 to 33,047 genes, so
  `batch_structure` cannot stack their means without a rule - intersection, union
  with zeros, or a declared panel. the join declares unit semantics and states
  that the gene universe is settled upstream; no cohort declaration names it.
  it refuses rather than choosing.

both are cohort declarations, not package work. neither is a defect in the port.


---

## order, and what each step is accepted on

```
0  scope line       this file                                              <- here
1  commit           ingest_hest image fix
2  he agent         he.encode -> run_id, runs.status/resume/record
3  st agent         graphs/st.py over the novae port, st.encode
6  eda acceptance   panels + gene axis declared; tnbc-92 four-way table
4  downstream       align.run / evaluate.run behind the same run handle
5  hest pack        promote + report
7  pathway agent    package path for gpath2vec (may slip to 1.3)
8  tag v1.2.0       0-6 green in a clean clone
```

6 sits before 4 deliberately. this repo's own first constraint is that no
alignment or evaluation happens until the EDA gate is satisfied, and the second
cohort's gate currently returns stop. running the pipeline past it would break
a rule the project enforces on itself.

2 and 3 still run BEFORE 6, because what they test is the surface. their
acceptance goes THROUGH the gate rather than around it: the stop verdict fires
as an interrupt, the client answers "proceed, recorded as dissent", and the
record carries `actor=human`. that is the mechanism being exercised. a script
that bypasses the gate and records nothing is what happened on 24 sep and is
the thing not to repeat.

three acceptance criteria worth stating rather than assuming:

- **step 2 includes kill-and-resume.** `server.json` claims the server is
  stateless because a run handle is an ordinary parameter rather than a
  session. that is only true if the run's state is in sqlite, so the test is:
  start a run, kill the process, restart, `runs.resume` the same run_id. with
  an in-memory saver it passes in one process and fails on any restart, and the
  claim is false.
- **step 3 is "exact if deterministic, otherwise the tolerance is recorded with
  its cause."** the H&E port needed 3.02e-06 once the encoder was in the loop. a
  non-zero diff on the molecular side is a finding, not a failure to paper over.
- **step 6 leaves p untouched.** disagreements with the recorded summary are
  named with their reason. a threshold moved to make a cohort reproduce is
  circular, and the summary is the weaker record - it rests on one subarray.

## rejected, with the reason

```
a generic run_encode with a modality= switch    the gate sets genuinely differ -
                                                gated_weights and tile geometry
                                                against platform_floor and edge
                                                scale. one tool hides the thing
                                                each agent exists to ask
new checks or measures                          this is a port and an exposure
improving a check "while in there"              write it below and continue
```
