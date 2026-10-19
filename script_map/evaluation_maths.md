# script map: the statistics the routing matrix cites

the maths behind `docs/v2/tnbc92_routing_matrix.md` and the 9-column matrix in
`docs/reports/internal/tnbc-92/index.html`, traced to the code that computed it
and the artifact it wrote. this is the specification for
`measures/evaluation.py`.

read in the order the rule requires: the artifact's provenance, then the script,
then the parameters measured out of the artifact, then the design doc to find
what it does not cover.

---

## 1 · the six cited artifacts, and who wrote them

`projects/tnbc-92/routing_evidence.json` names these. all six exist.

| task family | artifact | written by |
|:--|:--|:--|
| `cross_modal_retrieval` | `eval/H1/summary.json`, `{run}/metrics_h1_ci.json` | `align.py` computes, `eval.py` rolls up, `eval_h1_bootstrap_ci.py` intervals |
| `structural_agreement` | `eval/H1/summary.json` (`CKA_after`, `alignment_gap`) | same |
| `tissue_state_grouping` | `eval/H2/mc_coherence.parquet`, `metrics_h2_ci.json#h2a` | `eval_h2_mc_coherence.py`, `eval_h2_bootstrap_ci.py` |
| `subject_identity_suppression` | `eval/H2/metrics_h2_ci.json#h2c` | `eval_alignment_biology.py`, `eval_h2_bootstrap_ci.py` |
| `morphology_decode` | `eval/H2/tls_signature_cca.parquet` | **`scripts/_scratch/eval_h2_tls_signature.py`** |
| `pathway_transfer` | `eval/H3/pathway_cca_gpath2vec_v3/per_pathway_cca.parquet` + `perm_nulls.parquet` | `eval_h3_pathway_cca_gpath2vec_v2.py` |

two provenance facts worth stating before any port:

**the H1 statistics are not in an eval script at all.** `eval.py:102` reads
`metrics_h1_raw.json` and relabels its keys; the arithmetic is
`align.py:401 retrieval_metrics`, run at TRAINING time. a reader looking for
"where is the AUC computed" finds a rollup first.

**a cited artifact's writer is in `_scratch/`.** `morphology_decode` is one of
the seven routable families and the only one whose producing script sits in the
scrapped directory. the artifact is real and the number is cited; the script's
location says the opposite.

`align.py:_EVAL_STAGES` names four scripts. six wrote the cited artifacts.

---

## 2 · the three machines

everything in section 3 reduces to these. each is implemented more than once
today.

### permutation null -> z and p

| line | form |
|:--|:--|
| `eval_h3_pathway_cca_gpath2vec_v2.py:316` | `perm_p(nulls, obs) = (1 + #{null >= obs}) / (1 + n)` - **one-sided** |
| `:325` | `perm_p_two(nulls, obs) = (1 + #{|null| >= |obs|}) / (1 + n)` - **two-sided** |
| `eval_alignment_biology.py:214, 235` | `biology_perm_test`, `patient_perm_test` - the z form |

both p forms drop NaN nulls and return NaN when the null is empty or the
observation is NaN. the `1 +` in numerator and denominator is deliberate: a
p of exactly 0 is not a measurement, it is a permutation count.

**the two tails are not interchangeable and were conflated once.** the DAG
result was reported as 78% and as 80.3%; they are the two-sided and one-sided
tails of the same test. a port where the caller does not SAY which tail it
wants reproduces that.

### BH-FDR

`eval_alignment_biology.py` uses `statsmodels.multipletests`. the family is
whatever the caller passed, and the matrix's own guards record two decisions
about family membership: an underpowered pathway was excluded from the family
and reported separately, and the DAG family is 395 nodes rather than 5.

### bootstrap CI

`eval_h2_bootstrap_ci.py:92` - `rng.choice(unique_pats, size=n_pat,
replace=True)`, then gather every unit of the sampled subjects WITH
multiplicity, then recompute. **resampled by SUBJECT, not by unit.** a unit
bootstrap on 35,594 niches from 14 subjects would give intervals far too narrow,
because the units are not independent.

---

## 3 · the statistics

### paired retrieval · `align.py:401`

| line | operation | note |
|:--|:--|:--|
| 403 | `sim = z_he @ z_st.T` | assumes rows are already L2-normalised |
| 405 | `ranks = (-sim).argsort(1).argsort(1)[diag]` | double argsort, 0-indexed rank of self. `np.argsort` default is quicksort, so **ties break arbitrarily** |
| 406-408 | `R@k = mean(ranks < k)` | k in 1, 5, 10 |
| 409 | `MRR = mean(1 / (ranks + 1))` | |
| 410 | `median_rank = median(ranks + 1)` | 1-indexed here and 0-indexed above |
| 411-414 | `gap = mean(diag) - mean(off-diag)` | all off-diagonal, not a sample |
| **415** | **the AUC's negatives are SUBSAMPLED** | `rng(0).choice(mismatched, size=min(10*N, len(mismatched)), replace=False)` |
| 420 | `roc_auc_score(labels, scores)` | NaN on ValueError |

two parameters hide in line 415 and neither is declared anywhere: the
**negative ratio 10** and the **seed 0** - which is not the 42 the rest of the
chain uses. the published AUC is over a seed-0 sample of negatives, and nothing
in the artifact says so.

### geometric agreement · linear CKA · implemented THREE times

```
align.py:427                 cka_linear(X, Y)    centres internally, +1e-12
eval_h1_bootstrap_ci.py:61   cka_linear(Xc, Yc)  expects PRE-CENTRED, +1e-12
eval_rank_matched_cka.py:62  linear_cka(X, Y)    centres internally, NaN on den 0
```

all three compute `‖XᵀY‖²_F / (‖XᵀX‖_F · ‖YᵀY‖_F)` and agree to 1e-16 on
centred input - verified. they differ in two ways:

- **who centres.** the middle one takes pre-centred input, and handed raw data
  returns **0.96 where the others return 0.054** - same formula, same function
  name, 17x apart. this is the single strongest argument for one implementation
- the degenerate case: two add an epsilon, one returns NaN

### clustering · `eval.py:123 cluster_quality`, `eval_h2_mc_coherence.py`

```
KMeans(n_clusters=k, n_init=10, random_state=42) -> fit_predict
ARI = adjusted_rand_score(y, preds)
silhouette: subsample 5000 via default_rng(42).choice, THEN
            silhouette_score(..., sample_size=n_sil, random_state=42)
```

the silhouette is subsampled twice - once by the caller picking 5,000 rows and
again inside sklearn, with the same seed.

**`cluster_quality` is defined twice, with the same name and a different `k`.**
`eval.py:123` defaults to `N_ARCHETYPES = 9`; `eval_h2_mc_coherence.py:41`
defaults to `N_MC = 14`. the bodies are otherwise identical except that the
second also returns `n`. the cited `mc_coherence.parquet` came from the k=14
one, and the matrix records that the label was corrected from `archetype` to
`mc_megacluster` - but the function that still carries the old `k` is live in
the other script. two functions, one name, and the answer depends on which
module a reader imported.

ARI is also **geometry-biased**, which the matrix states: a manifold embedding
scores low against KMeans' globular assumption while a linear probe on the same
space is at parity. so ARI alone is not a reading of biology, and the
rank-matched variant exists for that reason.

### decodability

```
linear probe   _scratch/eval_h2_mc_diagnostics.py:131  LogisticRegression,
                                                       split BY SUBJECT
kNN purity     _scratch/eval_h2_mc_diagnostics.py:80   per k, fraction of a
                                                       unit's k neighbours
                                                       sharing its label
```

both load-bearing - the probe is what showed the ARI result was metric-induced -
and both in `_scratch/`.

### association · univariate CCA · `eval_h3_pathway_cca_gpath2vec_v2.py:331`

```
centre both -> QR both -> SVD of Q_ZᵀQ_Y -> top singular value, clipped to [-1, 1]
weights by solve(R, u), falling back to lstsq on LinAlgError
```

the top canonical correlation, not a Pearson r on a chosen axis. a
`value_only` twin exists at :349 for the permutation loop, and the two must not
drift - which is a test, not a comment.

### contrast · the two-sample z

the matrix's categorical FTU row cites a one-sided Welch z between two groups
of paired cosines. **grep finds no `welch` anywhere** in `EVALUATION.md`, the
routing contract, `evaluate.py`, or `eval.py`. the number is published and the
statistic has no home.

---

## 4 · the parameters, and where each is evidenced

| parameter | value | evidenced by |
|:--|:--|:--|
| seed | 42 | every script's `SEED` |
| AUC negative ratio | 10 | `align.py:415`, nowhere else |
| AUC negative seed | **0** | `align.py:415`. not 42 |
| CKA epsilon | 1e-12 | two of the three implementations |
| KMeans `n_init` | 10 | `eval.py:127`, `eval_h2_mc_coherence.py` |
| k for clustering | 14 | `mc_coherence.parquet`; `eval.py` still names it `N_ARCHETYPES` |
| silhouette subsample | 5000 | `eval.py:130` |
| H3 permutations | 500 | `pathway_cca_gpath2vec_v3/provenance.json#n_perms` |
| H2-C permutations | 10000 | `align.py:_AS_BUILT`, and the script DEFAULTS to 100000 |
| rho null permutations | 1000 | `_AS_BUILT`, `biology_rho.parquet` |
| bootstrap resamples | 200 | `_AS_BUILT`, `metrics_h2_ci.json#n_bootstrap_h2a` |
| held-out subjects | 3 of 14 | `provenance.json#n_test_pats` |
| min testable set | 5 | `provenance.json#min_testable_set` |
| BH-FDR alpha | 0.05 | the matrix's own cells |

**a script default is not a declaration.** `eval_alignment_biology.py` defaults
to 100,000 permutations and the published artifact was built with 10,000.
`align.py:_AS_BUILT` already records that, and it is the model for how the port
should carry these.

---

## 5 · what no measure will cover, and why

| thing | why not |
|:--|:--|
| loading a run's embeddings, `z_matrix`, spot joins | a measure takes arrays. the chain assembles them |
| run ids, `RUNS_ROOT`, output paths | the run record owns paths |
| the rollup that relabels keys | `eval.py:102` is bookkeeping, not a statistic |
| plotting | `eval_compare_and_plot.py` stays a script |
| the DAG decomposition | deferred, and the task row is still v1-pathed |

---

## 6 · the three the matrix cites with no home anywhere

checked by grep against `EVALUATION.md`, `routing_contract.json` and
`evaluate.py`:

```
rank-matched control            rejects a collapse-driven reading of retrieval
                                eval_rank_matched_cka.py exists; no family, no guard
the categorical FTU contrast    the Welch z above. no family either - the
                                CONTINUOUS version has one, the categorical does
                                not, and that is the row where routing FLIPS
rotation-equivalence sanity     two classical arms must agree on within-space
                                metrics and diverge on cross-modal ones. a check
                                on the HARNESS, so the most transferable thing
                                in the matrix, and the only one with nowhere to sit
```

these are a contract edit rather than a measures edit, and they are in the
parking lot, not this card.

---

## 7 · tier 2 · what was diffed, and what each diff found

`tests/test_evaluation_measures.py` (45 cases, synthetic) and
`tests/test_evaluation_slicediff.py` (17 cases, against the published runs).

| statistic | criterion | result |
|:--|:--|:--|
| AUC | exact | `roc_auc_score` on the run's stored labels and scores reproduces `metrics_h1_raw.json#auc` to all 16 digits |
| negative ratio | exact | `neg_sample_cosines.size / matched_cosines.size` = 10, confirming the undeclared parameter |
| permutation mean and sd | **bit-identical** | the stored `null_A_mean` and `null_A_std` on every pathway of three runs, which also pins the convention: **ddof=0** |
| permutation p | exact to 1e-12 | `p_two_A` on every pathway of three runs |
| BH-FDR | exact to 1e-12 | once the family is the right one - see below |
| linear CKA | 1e-6 relative | the script accumulated over a materialised float32 matrix; this module sums in float64 |
| alignment gap | 1e-6 relative | same cause. computed here from the column sums rather than a 35,594² matrix - same number, five orders of magnitude less memory |
| ARI, silhouette | exact at float32 | see the dtype finding below |
| bootstrap CI convention | 1e-6 relative | the central two-tailed quantile reproduces both published intervals from the run's stored 200 draws |

### the published FDR family is the grid, not the run

the matrix's cell reads "BH-FDR over 4 testable". over 4, the one significant
pathway's q is **0.0479**. the artifact's `fdr_A` is **0.0227**, which is BH
over **36** - nine runs times four testable pathways, one view. both are valid
corrections; they are corrections of different claims, and only one is what the
stored number means.

this is the whole argument for the family being an argument. `bh_fdr`
reproduces the artifact exactly when handed the 36-cell family and gives 0.0479
when handed the 4-cell one, and a function that corrected over "whatever was in
scope" could not have told the two apart.

### precision is part of the method for an iterative estimator

the first version of `cluster_agreement` upcast its input to float64, as every
other measure here does. the published ARI is **0.2339287118** and that version
returned **0.2338499031** - same seed, same `n_init`, same data.

a seed makes KMeans deterministic at a GIVEN PRECISION and not across
precisions, so the dtype selects which local optimum is found. the closed-form
statistics are different: there float64 is strictly more accurate and the
published float32 values differ from it only in the last bits.

so `cluster_agreement`, `linear_probe` and `knn_purity` take the caller's dtype
and record it, and the closed-form ones accumulate in float64. a port that had
tidied the dtype would have moved a published number with nothing to show for it.

### two artifacts of one run disagree about one statistic

`metrics_h1_raw.json#cka_after` and `metrics_h1_ci.json#cka` are the same
statistic on the same run, written by two scripts, and they differ in the last
digits. so does the published `z_A` column against the quotient of its own
stored `null_A_mean` and `null_A_std`, at 1e-10. neither moves a verdict - every
q in the family is unchanged - and both are the drift that one implementation
removes and three accumulate.

### the parametric AUC interval is computed and not used

`metrics_h1_ci.json` carries two AUC intervals: a Hanley-McNeil closed form and
a bootstrap. the evidence pack cites the bootstrap. the closed form is not
implemented here, because nothing routes on it.
