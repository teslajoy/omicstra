# CURRENT

the one task in flight. read this at the start of every session, before any
design doc and before any code.

this file holds ONE task. when it is done, it is replaced, not appended to.
everything that is not the one task goes to the parking lot at the bottom and
is not worked on, discussed, or measured until the task card is closed.

---

## the task

**the statistics the routing matrix cites, as `measures/evaluation.py`.**

**DONE 2026-10-19.** three machines and eight statistics, 45 synthetic cases
and 17 against the published runs. the AUC, the permutation moments and p, and
the BH q reproduce exactly; CKA, the alignment gap and the CI convention within
the script's own float32 rounding; ARI and silhouette exactly at the dtype the
artifact used. the next card is `EVALUATE_STEPS` - the chain over these.

today the package can STATE a verdict over artifacts an evidence pack cites and
cannot PRODUCE one of the numbers. `omicstra route` works because it reads a
pack; a pack exists because someone ran 2,132 lines of cohort scripts by hand.
that is the whole reason there is one cohort in `projects/`.

`measures/` is the maths: pure functions, no ctx, no order, no authority,
callable from a notebook. this task puts the matrix's statistics there. it does
NOT build a chain over them - that is the next card.

### done-criterion

**three machines, each existing exactly once:**

```
permutation z + p    z = (obs - null mean) / null sd
                     p = (1 + #{null >= obs}) / (1 + n), one- AND two-sided,
                     and the caller says which. never 0, never anti-conservative
BH-FDR               over the family actually tested, and the family is an
                     argument rather than whatever happened to be in scope
bootstrap CI         resampled BY A DECLARED GROUP, not by unit
```

**the statistics, as pure functions over matrices:**

```
paired retrieval     AUC, recall@k, MRR, alignment gap
geometric agreement  linear CKA
clustering           ARI, silhouette, rank-matched variant
decodability         linear probe accuracy, kNN purity
contrast             two-sample z between two groups of paired scores
association          univariate CCA - the top canonical correlation, via
                     least squares, with its permutation null
```

**and every one of them:**

- takes arrays, returns a number or a small dict. no file paths, no run ids,
  no cohort names, no DataFrame schema
- states its own preconditions in the signature or refuses. the CKA trap below
  is the reason this is a criterion and not a nicety
- every threshold declared. `EVALUATION.md` §8 decides which kind each is
- a test that FAILS if the statistic is computed a second way and disagrees

**tier 2:** the cited artifacts are the oracle. the six that
`routing_evidence.json` names all exist under `runs/tnbc-92_v3/eval/`, so each
statistic is diffed against the published number it produced.

### inputs

- `script_map/evaluation_maths.md` - the map. every statistic, which script
  holds it, which artifact it produced, and its parameters
- the six cited artifacts and their provenance, listed in the map
- `EVALUATION.md` - what each test proves and the condition under which it does
- `src/omicstra/configs/routing_contract.json` - the 7 task families and the
  guards each requires. this task computes nothing new; it computes what the
  contract already names

### out of scope

```
EVALUATE_STEPS        the chain over these measures. the next card
fusion                after the chain, because an arm nothing can score
                      is not an arm
the DAG decomposition the v1-pathed task row. deferred, and recorded as such
new task families     the three tests with no home are a contract edit,
                      not a measures edit
cohort two            the card after next
```

---

## parking lot

real, written down so it stops re-entering the build, and not worked on:

- the three tests the matrix cites with no home in the contract: the
  rank-matched control, the categorical FTU contrast, and the
  rotation-equivalence sanity check
- `docs/v2/` holds the v3 numbers and `docs/v3/` does not exist; the fullest
  report is a 1.6 MB hand-built HTML that `report.py` did not render
- a cited artifact's writer lives in `scripts/_scratch/`
- `align.py:_EVAL_STAGES` names four scripts; six wrote the cited artifacts
- the two package defaults the threshold lint does not reach: `eda.py:430`
  `max_pct_negative`, `inventory.py:455` `PITCH_TOLERANCE`
- a lint for `.get(key, <number>)` - a default at the read site is the same
  defect the threshold lint catches at the write site
- an identifier guard: a cohort id, subject id or dimension literal in `src/`
- the docs commit limiting the join reproduction claim to the four sections
  it was verified on
- a scope row in the pack for the subjects and samples absent from every
  published number
