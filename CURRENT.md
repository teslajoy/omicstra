# CURRENT

the one task in flight. read this at the start of every session, before any
design doc and before any code.

this file holds ONE task. when it is done, it is replaced, not appended to.
everything that is not the one task goes to the parking lot at the bottom and
is not worked on, discussed, or measured until the task card is closed.

---

## the task

**`EVALUATE_STEPS`: the chain that COMPUTES a verdict, in
`protocols/evaluate.py`.**

**DONE 2026-10-19 (tier 1).** 6 steps, 42 synthetic cases, 876 pass. G2, G4 and
G5 all run, asserted by a test. tier 2 against the cohort's artifacts is NOT
done and is the next card.

the gates that already exist read a cohort's evidence pack and judge the numbers
in it. four of their guards cannot run on a pack, and each one says so by name:

```
evaluate.py:271   G2 held-out honesty   needs "a within-subject score to compare"
evaluate.py:272   G5 noise floor        needs "a permutation null over ARI"
evaluate.py:301   G4 specificity        needs "per-pathway axis vectors"
evaluate.py:302   G5 noise floor        needs "the permutation null over CCA"
evaluate.py:346   G5 noise floor        needs "the permutation nulls behind each z"
evaluate.py:422   G4 specificity        needs "per-pathway axis vectors
                                         (pathway_axes.parquet)"
```

**the last one names a file that cannot satisfy it.** `pathway_axes.parquet`
holds `pathway_name`, `pathway_id`, `coverage_frac`, `set_size` and `testable` -
metadata, no vectors. so G4 has been unrunnable for a reason that reads as a
missing file rather than as a statistic nobody computed.

`measures/evaluation.py` can now produce every one of those inputs. this card
is the chain that does it.

### done-criterion

**the four guards stop being `not_run`, each for a real reason:**

```
G2  compute the within-subject score beside the cross-subject one, so
    "held out" is a measured difference rather than a declared split
G4  compute the per-pathway axis VECTORS - top_canonical_correlation with
    with_weights=True - and pass them to the existing specificity_matrix
G5  compute the permutation null for each statistic the chain reports, and
    let the existing noise_floor guard read it
```

**and:**

- every statistic comes from `measures/evaluation.py`. a number computed in a
  protocol is a number no notebook can check
- every parameter declared. absent means refuse with a reason
- `n_in` / `n_out` / `reason` on every step, as `JOIN_STEPS` has
- a verdict per hypothesis in the existing vocabulary:
  `biological | diagnostic | unguarded | not_run`
- the chain does not re-judge what the pack-reading gates judge. two answers to
  one question is the defect; this answers the question they could not

### tier 1

synthetic, no cohort data:

```
1  a statistic with no null reported  -> the verdict is `unguarded`, not a pass
2  a label that encodes the subject   -> `diagnostic`, and the score is still
                                         reported
3  axes that alias                    -> the cosine travels with every
                                         magnitude
4  within beats cross                 -> the split is not holding anything out
5  a guard whose input is absent      -> `not_run` NAMING the input, never a pass
6  a hypothesis with no step that ran -> `not_run`, distinct from "no guard failed"
```

### tier 2

the cohort's artifacts are the oracle. the chain's computed z, q and specificity
are diffed against `per_pathway_cca.parquet`, `perm_nulls.parquet` and
`mc_coherence.parquet` - all verified reachable by the previous card.

### inputs

- `protocols/evaluate.py` - the six guards, the gates, and the four `not_run`
  lines above. this card extends the module, it does not replace it
- `measures/evaluation.py` - the maths, as of the previous card
- `script_map/evaluation_maths.md` - which statistic answers which hypothesis
- `configs/routing_contract.json` - the 7 task families and the guards each
  requires. the chain computes what the contract already names
- `EVALUATION.md` - PROVES / ONLY IF / INSTEAD READS per test

### out of scope

```
fusion                 the card after this one
new guards             G1-G6 are the set. this card makes four of them runnable
new task families      the three tests with no home are a contract edit
the DAG decomposition  deferred, and the task row is still v1-pathed
cohort two             the last card
```

---

## parking lot

real, written down so it stops re-entering the build, and not worked on:

- `noise_floor` passes at a hardcoded `p < 0.05` while the chain declares its
  BH alpha. two alphas, one declared and one not
- EVALUATE_STEPS tier 2: the chain's computed z, q and specificity against
  `per_pathway_cca.parquet`, `perm_nulls.parquet` and `mc_coherence.parquet`
- the three tests the matrix cites with no home in the contract: the
  rank-matched control, the categorical FTU contrast, and the
  rotation-equivalence sanity check
- `pathway_axes.parquet` carries no axis vectors; whatever writes it should
- the published FDR family is 36 cells and the matrix's prose says 4
- two artifacts of one run disagree about CKA in the last digits, and the
  published `z_A` disagrees with its own stored moments at 1e-10
- the Hanley-McNeil AUC interval is computed and nothing routes on it
- `docs/v2/` holds the v3 numbers and `docs/v3/` does not exist; the fullest
  report is a 1.6 MB hand-built HTML that `report.py` did not render
- a cited artifact's writer lives in `scripts/_scratch/`
- `align.py:_EVAL_STAGES` names four scripts; six wrote the cited artifacts
- the two package defaults the threshold lint does not reach: `eda.py:430`
  `max_pct_negative`, `inventory.py:455` `PITCH_TOLERANCE`
- a lint for `.get(key, <number>)`; an identifier guard for cohort literals
- the docs commit limiting the join reproduction claim to its four sections
- a scope row in the pack for the subjects absent from every published number
- ruff finds 30 pre-existing issues in files outside this work; they are not
  this card's and a broad `--fix` has already cost a revert once
