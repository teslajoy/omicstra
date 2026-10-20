# CURRENT

the one task in flight. read this at the start of every session, before any
design doc and before any code.

this file holds ONE task. when it is done, it is replaced, not appended to.
everything that is not the one task goes to the parking lot at the bottom and
is not worked on, discussed, or measured until the task card is closed.

---

## the task

**make `EVALUATE_STEPS` reachable: a client asks, and a verdict is COMPUTED.**

**DONE 2026-10-20.** `adapters/evaluation.py` reads the declaration and refuses
by field; `graphs/evaluate.py` gained a `measure` arm routed at entry;
`mcp/server.py` exposes `evaluate_measure` (20 tools). 20 tier-1 cases, 4 at
cohort scale, and the graph's verdicts match the chain invoked directly:
H1 biological, H2 diagnostic, H3 not_run on R1_v3.

the next card is the JOIN duplication - one join instead of
run_niche_join(compute=True) shelling out to the script.

the chain exists, has 57 tier-1 cases and 18 at cohort scale, and no client can
reach it - `grep` finds no caller outside its own module and its tests. the
evaluate graph reads an evidence pack and recomputes nothing; its own note says
so. so a cohort with no pack still needs a script.

the blocker looked like a missing contract and is not. `project.json` already
declares every input the chain needs:

```
niches_dir        the join parquets - labels and continuous signals
runs_dir          {run}/embeddings_test.parquet - the views
h2_label          which label the clustering is scored against
headline_view     which view carries the claim
supervision       the G6 target, which must not be a feature
k_neighbors seed  the geometry and the seed
```

so this is a contract READER, not a new contract.

### done-criterion

```
1  a declared reader turns a cohort's files into the chain's arrays. it reads
   project.json and refuses, naming the field, when one is absent
2  graphs/evaluate.py gains a COMPUTE arm. compute=False stays the default and
   stays the pack-reading path - a node that computes by default turns every
   caller into a 4-minute job
3  the two arms do not re-judge each other. the pack arm states what the pack
   claims; the compute arm states what this run measured
4  an MCP tool reaches the compute arm, and a client gets a verdict with no
   script
5  tier 1: the reader refuses every absent declaration by name; the compute arm
   on synthetic arrays reaches a verdict; the pack arm is unchanged
6  tier 2: the computed verdict on tnbc-92 matches what the chain already
   produced directly - same ARI, same AUC, same NMI
```

### out of scope

```
the JOIN duplication   run_niche_join(compute=True) shells out to
                       build_niche_join.py, and JOIN_STEPS is the package's
                       port of that script. one join needs the same reader
                       shape, one level further back, and it is the next card
fusion                 still parked
gpath2vec steps 2-3    blocked on the one-line fix in that package
cohort two             after the join reader
```

---

## parking lot

- the TF gene filter is an undeclared input: `knowledge/pathway_commons/
  tf_genes_controls_expression.json`, 1,658 PathwayCommons v12 symbols, reached
  the published build through a hand-written command. step 1 reproduces exactly
  with it (680 pathways, 252 zero-sig niches on TNBC1_CN1_C1) and does not
  without it (781, 274). it belongs in the cohort's contract
- gpath2vec: `cli.py:305` `r._w` cannot resolve - pandas renames any
  leading-underscore column in `itertuples`. deterministic, input-independent.
  and its `__version__` says 1.0.0 while the distribution says 3.0.0
- `noise_floor` passes at a hardcoded `p < 0.05` while the chain declares its
  BH alpha
- the full gpath2vec fit is an 80 GB / 11.4 h cluster job that OOM'd once
- measured on this mac, CPU: virchow2 206 s / 11.3 GB per section, novae 17 s /
  4.3 GB. 11.3 GB peak is what rules out laptop parallelism
- scope row owed: 259 of 281 subarrays, 92 of 94 patients
- the three tests the matrix cites with no home: rank-matched control, the
  categorical FTU contrast, the rotation-equivalence sanity check
- `docs/v2/` holds the v3 numbers; `docs/v3/` does not exist
- the two package defaults the threshold lint does not reach
