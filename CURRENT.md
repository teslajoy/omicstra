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

done: the import is inside the functions, the refusal names the `measure`
extra, and the CI job asserts the refusal.

### 2 · one join, wired

`JOIN_STEPS` is 11 steps and 99 tests, exact against the 259 published
parquets, and nothing calls it. the graph calls `run_niche_join`, which
resolves a cached directory or shells out to `scripts/build_niche_join.py` - a
script that names this cohort 27 times.

done: `run_niche_join(compute=True)` runs `JOIN_STEPS` through a declared
file-to-assay reader, the same shape as `adapters/evaluation.py`. the script
keeps its historical role and stops being the method. the 45 tier-2 cases still
pass.

note: `scripts/build_niche_join.py:54` hardcodes a path that no longer exists
after the 2026-05-14 rename. fix before any rebuild through the old path.

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
