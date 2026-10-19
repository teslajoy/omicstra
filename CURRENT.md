# CURRENT

the one task in flight. read this at the start of every session, before any
design doc and before any code.

this file holds ONE task. when it is done, it is replaced, not appended to.
everything that is not the one task goes to the parking lot at the bottom and
is not worked on, discussed, or measured until the task card is closed.

---

## the task

**translate the niche-join into `protocols/join.py` as `JOIN_STEPS`.**

a niche join takes per-unit outputs from N assays plus per-unit and per-subject
label tables and emits one row per unit, fixed-width, with a funnel that
explains every row it did not emit.

### done-criterion

all six tier-1 synthetic cases pass, and nothing else counts as done:

```
1  a unit with no pathway signal          -> dropped, counted, reason recorded
2  a unit present in one assay only       -> dropped at the shared-unit step
3  a subject absent from one assay         -> the whole sample is not_applicable
4  a sample with zero units after the join -> refuses, does not emit an empty row set
5  a non-data numeric column in an input   -> read from the declared list, never inferred
6  normalise before vs after the drop      -> the two give different numbers; the
                                              declared order is the one that runs
```

plus, on every step:

- every parameter is a declaration. absent means refuse with a reason, never a
  default and never a `KeyError`
- `n_in`, `n_out`, `reason` on every step. an unexplained drop is an error
- no cohort constant anywhere in the module. not a subarray id, not a patient
  id, not a dimension

### inputs

- `script_map/build_niche_join.md` - the line-by-line map. every step cites the
  map rows it mirrors. a step that cites nothing is not a translation
- `scripts/build_niche_join.py` - the specification. the script is the truth;
  the docstrings, the design docs and the notebook prose are all downstream of
  it and have each been stale
- `src/omicstra/protocols/join.py` - already holds `FunnelStage` with a required
  `reason`. reuse it
- `src/omicstra/protocols/pathway.py` - the shape a chain takes here:
  `list[Step]`, `_declared(ctx, field)`, `applicable_when`

### out of scope

not started, not measured, not discussed until the card closes:

```
fusion                      early and late, both
evaluate                    every hypothesis gate
the seed cohort's numbers    tier 2. a regression test, after tier 1 passes
hest                        tier 3. separate, and not in this repo's tests
any new protocol            join.py is the only module this task touches
any new guard or lint       except one the six cases force
```

---

## parking lot

real, written down so it stops re-entering the build, and not worked on:

- the two package defaults the threshold lint does not reach: `eda.py:430`
  `max_pct_negative`, `inventory.py:455` `PITCH_TOLERANCE`
- `ddof` in the per-subarray z-score must match the script and be cited to it
- a lint for `.get(key, <number>)` - a default supplied at the read site is the
  same defect the threshold lint catches at the write site
- an identifier guard: a cohort id, subject id or dimension literal inside
  `src/` is a cohort constant in the package
- inductive vs fitted at the transform boundary
- a pathway-only arm, its probes, and permutation importance from the existing
  checkpoint
- how absence is ENCODED rather than dropped: a unit with no pathway signal is
  information, not missingness
- the docs commit that limits the reproduction claim to gene-set equivalence
- a scope row in the pack for the subjects and samples absent from every
  published number
