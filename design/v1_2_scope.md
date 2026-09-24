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
