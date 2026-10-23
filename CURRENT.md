# CURRENT

the one task in flight. read this at the start of every session, before any
design doc and before any code.

this file holds ONE task. when it is done, it is replaced, not appended to.
everything that is not the one task goes to the parking lot at the bottom and
is not worked on, discussed, or measured until the task card is closed.

---

## the task

**one MCP call on a cohort returns a verdict.**

that sentence is the whole card. today a client can reach a computed verdict
only on a cohort that already has fitted runs and a built join - which means
the seed cohort and nothing else. four things stand in the way, in this order.

### 1 · the bare install crashes on its second command

```
$ pip install omicstra && omicstra init --project-dir X
$ omicstra inventory --project-dir X
  File ".../omicstra/protocols/inventory.py", line 30
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

a traceback where `omicstra gate` says *"no eda_summary.json in X - the EDA gate
has not been run for this cohort"*. the encoder registry gets this right -
`EncoderUnavailable` naming the extra - so the pattern exists and one module
missed it. three modules import numpy at module level; `protocols/inventory.py`
is the one on a CLI path.

CI's `a bare install serves` job asserts `--help` works and the server lists
tools. it never asserts that a numerics command REFUSES legibly, which is why
this is green.

**DONE 2026-10-20.** three causes, not one:

- `protocols/inventory.py` imported numpy at module level, so the crash came
  before any step ran. the five functions that need it import it themselves
- `shape`'s per-sample `except Exception` recorded a missing reader as an
  unreadable SAMPLE, reporting `0/1 readable` - which blames the cohort's data
  for an absent package. it re-raises `ImportError` now
- `build_protocol`'s `_link` catches everything by design, so nothing reached a
  `try` in the CLI. the CLI reads the records and translates

and `error` now counts as failure at the write gate - it did not, so a chain
whose reader was absent still wrote an `inventory.json` with one step errored
and six `not_run`.

proof, in a clean venv with no numerics:

```
$ omicstra inventory --project-dir /tmp/ci_bare
Error: `omicstra inventory` measures data, which needs the numerics extra -
'anndata' is not installed.

    pip install 'omicstra[measure]'
exit 1, and no inventory.json written
```

CI's `a bare install REFUSES legibly` job asserts the non-zero exit, that the
message names the extra, that it is not a traceback, and that nothing was
written.

### 2 · one join, wired

`JOIN_STEPS` is 11 steps and 99 tests, exact against the 259 published
parquets, and nothing calls it. the graph calls `run_niche_join`, which
resolves a cached directory or shells out to `scripts/build_niche_join.py` - a
script that names this cohort 27 times.

**DONE 2026-10-20.** `run_niche_join(compute=True)` runs `JOIN_STEPS` through
`adapters/join.py`, a declared reader with four loaders resolved by name.
`dispatch` owns the loop; the manifest is written from what the chain recorded.

verified on two sections, through the function the align graph calls: every
vector column EXACT against the published parquet - `virchow2_niche`, the
8,960-wide token block, `gpath2vec_niche`, `novae_niche` standardised after the
drop, `mc_weights_niche` - the scalar column exact at float32, the unit sets
identical, and the funnel counts equal to the published manifest's.

the one intended difference: where the script wrote `-1` into the label's own
value space, the chain writes NaN and a `{label}__present` column. a test
asserts the two agree about which units are absent and about every unit that is
present. that is open defect #7 closed in the package, with the published
artifact left alone.

ledger `#22` is mooted: the wired join never reads
`scripts/build_niche_join.py:54`, and the script keeps its historical role as
the record of what produced the published parquets.

three defects the real data found, all in the reader:
- a label table's declared column selection did not apply to an EMPTY slice, so
  a sample the table does not cover came back three columns wide and its pooled
  value was 3 NaNs where the artifact has one
- 280 token arrays have no meta of their own; they borrow the niche assay's
  index, which only a comment recorded. declared as `index_from`, with a length
  check, because reading it wrong silently permutes every vector against its unit
- a token block is (k+1, dim) and parquet stores 1-D, so the flatten belongs in
  the writer and the shape goes in the manifest

### 3 · evaluate reachable - DONE 2026-10-20

`adapters/evaluation.py` reads the declaration and refuses by field;
`graphs/evaluate.py` has a `measure` arm routed at entry; `mcp/server.py`
exposes `evaluate_measure` as the 20th tool. 20 tier-1 cases, 4 at cohort
scale, and the graph's verdicts match the chain invoked directly.

what it does NOT yet do is work on a cohort without fitted runs, which is 2.

### 4 · one call, end to end

done: on a cohort scaffolded by `omicstra init`, with canonical input and its
declarations filled in, a single MCP call returns a verdict per hypothesis -
`biological | diagnostic | unguarded | not_run` - with no script in the path
and every absent declaration refused by name.

that is the release claim for 1.3, and the second cohort's pack is what proves
it. the record already says so: *"the second cohort's evidence pack did NOT
ship and is 1.3's whole claim."*

### 5 · the verdict as something a reader opens - DONE 2026-10-20

a verdict in a JSON tool result is not a deliverable. `unit_notebook` (21st
tool) and `omicstra notebook UNIT` write an ANALYSIS UNIT as a rerunnable
notebook plus a markdown trace, from the cohort's own decision ledger. the unit
is the key, not the run: one current record per unit, regenerated, with the run
named in the header.

verified on tnbc-92/R4_v3, generated and EXECUTED:

```
pass             label_granularity          reproduced 5 of 6 step(s)
not_applicable   paired_identifiability
pass             structure_and_its_null     H1: not_run, H2: diagnostic, H3: not_run
not_run          held_out_comparison        - identical to the recorded run
not_applicable   association_and_specificity
pass             hypothesis_verdict
```

four defects the execution found, three of them in the record rather than in the
notebook:

- a record's `params` was empty for every step in every chain. `_link` filters a
  cohort's declaration to each step's `params_space`, so only that frame knows
  what the step saw, and nothing wrote it down - the first rerun re-derived
  nothing and said `reproduced`
- `duration_s` was the NODE's time on every record it added, and three places
  sum the field: 874.3s reported for a 147.8s run. timed per step now, and
  `timed_node`'s `or` fallback - which replaced a known 0.0 - tests `is None`
- `structure_and_its_null` put G1's finding in G5's note, so the guard table
  read "G5_noise_floor pass - the label encodes the confounder"
- the footer printed `reproduced` when every step was `not_run`, and the setup
  cell formatted an adapter name of `None` into a module path for five of the
  seven units

---

## where the next session starts

```
branch        v1.3-design-and-catalogue
suite         1055 passed, 1 skipped   (venv/bin/python -m pytest tests/ -q)
tree          DIRTY - two commits prepared, neither made. messages written at
              design/_scratch/commit_records.txt and commit_notebook.txt
              (git-ignored). commit the record fixes FIRST:
                protocols/__init__.py  records.py  protocols/evaluate.py
                tests/test_one_ledger.py  tests/test_evaluate_slicediff.py
              then everything else.
```

sections 1-5 above are done and each carries its proof. what is left of the card
is the one thing a package cannot do for itself:

```
cohort two      the release claim. `omicstra init`, canonical input, the
                declarations filled in, then ONE MCP call to a verdict. the two
                human decisions that block it are the marker panels and whether
                cross_modal_registration / model_tissue_fit /
                multi_section_alignment get built or get declared unanswerable
publish         1.3 is not on PyPI; latest there is 1.2.0
```

three items this card surfaced and did NOT fix, all in `docs/open_items.md`:

```
41  a cohort with no declared work_dir writes under the PACKAGE. the default
    cannot be changed in passing - several tests isolate by redirecting
    runs_dir, and a declared work_dir ignores that redirect
42  `omicstra inventory --max-samples 2` REPLACED the committed 280-sample
    projects/tnbc-92/inventory.json. restored from git. a capped run must not
    be able to overwrite an uncapped record
43  the CLI step chains persist nothing to the ledger - only the graph arms do,
    through mcp/server.py:_persist. so report, trace and notebook see nothing
    for a cohort driven from the CLI
```

---

## not in this card

```
any science review          the record is ahead of it. document precedence and
                            scope-before-quote are now rules in CLAUDE.md
fusion                      an arm nothing can score is not an arm
the gpath2vec fix           that package's call. the bug is unfixed on main at
                            0bdd9c6, so there is nothing to install from
promote / route on cohort 2  a pack is a curated human act, by design
```

---

## parking lot

the project's own open-items ledger is the authority; these are the entries this
card touched or created. everything else lives where the record puts it.

- `noise_floor` passes at a hardcoded `p < 0.05` while the chain declares its
  BH alpha
- three cohort strings are package defaults on `ProjectConfig` - `h2_label`,
  `supervision`, `headline_view`. the threshold lint is numeric-only and cannot
  see them
- 86 slice-diff tests are data-gated, so no reproduction claim runs in CI. a
  committed downsampled fixture would let CI assert the shape of each diff
- the TF gene filter is an undeclared input and reached the published build
  through a hand-written command
- `measures/evaluation.py` has no Hanley-McNeil interval and should not grow one
  - nothing routes on it
- gpath2vec has no CI and no test on `cli.py`, which is why the `_w` bug shipped
