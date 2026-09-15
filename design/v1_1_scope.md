# 1.1 scope

> **1.1 = `omicstra run` produces the question x method matrix and report from
> files on tnbc-92, mac only.**

1.0.0 ships before any of this: the read path, the alignment and evaluation
stages verified bit-identical against the published grid, and the seed cohort's
evidence pack so a fresh install can route. what follows is the release after.

read this before adding anything. the rule that ends an argument about scope is
the line above, not a judgement about whether the work is good.

---

## the two decisions, made up front

these are settled so the build aims at them rather than asking halfway.

### 1 - the acceptance grid is B1-B3 plus one capped contrastive arm

the full ten arms is a long night on a mac. it is a **documented command, never a
test**. three tiers, and the top one is not negotiable-by-convenience:

| tier | what | where | why |
|:--|:--|:--|:--|
| CI | fixture-pinned tests only | github, every push | Virchow2 is gated and the inputs are 76 GB. the end-to-end **cannot** run here at any grid size |
| pre-tag gate | reduced grid, files -> pack | this mac, by hand | the acceptance test for 1.0.0 |
| long run | the full ten arms | documented command | reproduces the published grid |

B1-B3 are deterministic linear algebra, so they assert **exact** equality. the
capped contrastive arm covers the training path under tolerance. this is the
`EXACT_RUNS` / `TOLERANCE_RUNS` split the suite already makes - it is extended,
not invented.

**encode's acceptance is a slice-diff, not a re-extraction.** 75 GB of embeddings
are already cached and they are the oracle: extract a slice, diff against the
cache. that tests our port rather than testing Virchow2, and it runs in minutes.
the same method proved the align port at delta 0.00e+00 on 13/13 metrics.

### 2 - promote proposes, and never authors the notes

winners, margins, refusals and contraindications are computed deterministically.
notes and caveats are **human fields**, taken at a gate, recorded with
`actor=human`.

made checkable rather than asserted: the pack carries per-section provenance, so
a test fails if a `value` or `margin` is ever authored, or a `note` ever
generated. the readme already says "generating it would invent the judgements it
records" - this is the assertion that keeps it true.

---

## the report is a template, not tnbc-92's report

the renderer takes **a pack and a record ledger** and knows nothing about any
cohort. this is the package/project axis the codebase already enforces:
cohort-free code in the wheel, cohort measurements in `projects/{id}/`.

sections, all driven by what the cohort actually ran:

```
1  cohort              cohort.json / platform.json / inventory record
                       what was declared, what bound, what was refused
2  admissibility       eda records: which checks ran, which were not_applicable
                       and why, the gate verdict, who accepted what
3  what was computed   encode / join / align records: encoders, funnel, arms
                       trained, backend, shards, time
4  per-hypothesis      one block per H that ran; a one-line
                       "not applicable: <reason>" for each that did not.
                       ground truth named per H, confounder score on any
                       derived reference
5  the matrix          question x method, from the pack: winner, margin, CI,
                       refusal, contraindication, floor and reference per row
6  caveats + notes     the pack's human-authored fields, verbatim, marked as such
7  provenance          record ids, protocol revision, actor ledger, version
```

three rules that keep it a template:

- **no cohort names, numbers or prose in the renderer.** the test: render a
  synthetic two-method one-task pack and the tnbc-92 pack with the same code.
  both must produce a valid report.
- **a section appears because a record exists**, not because the code expects
  one. a cohort that never ran H3 has no H3 section and says so in one line.
- **the tnbc-92 internal report is the reference OUTPUT, not the template.**
  step 6 is accepted when rendering tnbc-92's pack reproduces that report's
  tables - winners, margins, refusals - with the narrative reduced to the pack's
  notes.

this is what makes hest-breast's report free later: same renderer, different
declarations, and every "not applicable" line is a true statement about that
cohort rather than a gap in the code.

---

## order

evidence-driven, not anticipatory. shards are last because nothing in the
definition above requires resumability until a full run proves it does.

```
1   this file is the boundary. 1.0.0 tags first, from the read path   done
2   [encode] extra: torch, timm, novae - must not leak into core
3   encode as compute: Virchow2 + Novae run here. 5 preflight gates
    fire, pre-answered from cohort.json. acceptance = slice-diff
4   run_niche_join as compute. funnel + no-circular-supervision guard
5   protocols/evaluate as a chain: H1, H2 with the confounder gate on
    any derived reference, H3 cross-patient. H2/H3 get H1's
    fixture-pinned treatment
6   graphs/promote.py: deterministic proposal, human gate,
    per-section provenance
7   report renderer: cohort-free, from pack + ledger
8   omicstra run end-to-end on tnbc-92; diff against the curated pack
9   merge the route-graph branch (promote sits beside it)
10  shards + local dispatcher - ONLY if 8 shows it is needed
```

## out until later still

hest-breast compute, oauth, the harness, a second institution. none of these is
in the line at the top of this file.

**temporal came in early**, at 3.6, rather than waiting for step 8 to prove it
was needed. the evidence that justified it: a full H&E extraction is 280
subarrays at ~0.073 s/spot, so **~5.8 hours** on this mac - a laptop run that
will be interrupted, which is step 3's problem rather than step 8's. Recorded
here because the boundary file is what ends a scope argument, and it should say
what is true rather than what was planned.

## the next goal, and why it is a new branch

**Extraction at scale is not built, and it is the thing a second cohort needs
most.** What exists is the extraction PATH, verified against a cache that
already existed - `virchow2_niche/` is 280 subarrays dated 9 April, produced by
the original script. This session's port produced 13 subarrays, all for
verification. On tnbc-92 there is no reason to run more; on a cohort with no
cache, those 5.8 hours are real and unavoidable work.

So the next version's goal is the **SLURM / ARC dispatch backend**: a third
executor beside `local` and `temporal`, submitting one job per shard against the
same `Shard` contract and the same output-file idempotency. The pieces are
already shaped for it -

- `dispatch/` takes a backend by configuration, and `run_shards` already
  chooses one. a third is an addition, not a redesign
- `scripts/slurm/` holds working sbatch files from the gpath2vec cohort runs
- `settings.compute_backend` already declares `local | cloud | slurm`
- the compute contract's `capacity` gate exists precisely for a scheduler

and two things are known to be missing:

- **`cohort.json` declares `compute_backend: "mac"`, which is not one of the
  declared options** (`local | cloud | slurm`). a value outside its own set is
  the same defect the gate option work fixed, one layer over
- a scheduler pause is not a laptop pause. the compute contract already says
  this: a scheduled job paused for a person is burning walltime, and if it hits
  its limit while waiting the answer is lost with it. that is what the mid-run
  gate and the durable workflow are for, and it is untested against a real queue
### the first declared alternative worth running

**the lattice-true Novae edge scale.** the cache was built at
`scale_to_microns` 1.2195 (200/164). on five subarrays the lattice-true scale
keeps global geometry (CKA >= 0.998) but changes about half of within-subarray
top-6 neighbour sets - more than same-size isotropic noise
(`projects/tnbc-92/platform.json#platforms.original_st.st_graph.scale_measurement`).
of the declared alternatives it is the only one that reaches the neighbourhoods
H1 is scored on, so it is the one that could move H1.

it is a 1.2 experiment, not a 1.1 blocker: all 262 subarrays re-embedded
through `StGraph` at the declared scale, the join rebuilt, the grid retrained,
H1-H3 scored against the as-built pack. the niche alternatives (k=4, k=8,
`platform.json#niche_geometry`) stay declared and unqueued.
