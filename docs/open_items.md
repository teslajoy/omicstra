# open items

every item this project already knows about, in its own words, with a citation.
the point of one file is that the next audit reads it instead of rediscovering
39 items across two repos, two agent-memory directories, `data_contract.json`,
three TODO docs and a scratch audit - which is what it cost to assemble this.

**maintain the `status` column here.** a row that has been fixed gets
`resolved` and the commit; a row the user has settled gets
`settled - do not raise` and stays, so nobody re-opens it.

**precedence, when two sources disagree** (the rule is in `CLAUDE.md`):

```
1  projects/{id}/routing_evidence.json   the PACK - adjudicated, with its own
                                         caveats and open_flags
2  docs/v2/tnbc92_results_summary.md     the stated verdict per hypothesis
3  docs/v2/*.md, docs/reports/           methods and reports
4  design/*.md                           WORKING docs. a concern here may
                                         already be answered at 1 or 2
```

a `design/` row below is a concern in the state it was written in. check 1 and 2
before treating it as open.

**scope.** `#1` and `#2` are the same defect in two places, so this is 39
distinct items. the four `deliberate` rows at the bottom are settled decisions,
not work.

---

## in the current card's path

| # | item | where | kind | the record's own words | status |
|---|---|---|---|---|---|
| 20 | a guard's alpha is hardcoded against the declared one | `CURRENT.md:87` | defect | "`noise_floor` passes at a hardcoded `p < 0.05` while the chain declares its BH alpha" | open - folded into the card |
| 22 | a hardcoded path in the join script no longer resolves | memory `project_gpath2vec_v1_vs_armA_provenance.md:28` | defect | "**latent breakage:** `scripts/build_niche_join.py:54` hardcodes ... which no longer exists post-2026-05-14 rename" | open - mooted when JOIN_STEPS is wired; confirm the wired join never reads it |
| 5 | the chain is not cohort-parameterised | `docs/TODO_report_generation.md:30` | defect | "**35 of 37 scripts in `scripts/` hardcode the cohort.** Not parameterised." | open - the wired chains are the answer; do not expand scope |
| 33 | `align_run` resolves a grid and never trains | `design/v1_3_scope.md:526` | deferred | "alignment has never executed through the package - the scripts are still the method ... which is 1.3 work and is not done." | deferred - same answer as 5 |

## needed before cohort two

| # | item | where | kind | the record's own words | status |
|---|---|---|---|---|---|
| 4 | the report is not reproducible, and that is the stated blocker | `docs/TODO_report_generation.md:3` | defect | "The pipeline is reproducible. The report is not. That asymmetry is the blocker for running this on a second cohort" | open |
| 21 | the published build's TF gene filter is an undeclared input | `CURRENT.md:78` | defect | "reached the published build through a hand-written command. it belongs in the cohort's contract" | open |
| 15 | a package default a cohort silently inherits | `design/v1_3_scope.md:253` | defect | "a cohort that declares `negative` but not `max_pct_negative` silently inherits 10%. the two-argument `.get` is the pattern the rule exists to stop" | open (recorded, not rewritten) |
| 16 | one platform's lattice tolerance applied to every cohort | `design/v1_3_scope.md:258` | defect | "`PITCH_TOLERANCE = 0.02` ... 2% was chosen on one platform's lattice and is applied to every cohort's geometry check" | open (recorded, not rewritten) |
| 14 | Moran's I gates the encoder branch off one of 281 subarrays | `projects/tnbc-92/eda_summary.json:85` | defect | "single subarray only - compute median across all 281 for registered report" | open since 2026-04-01 |
| 19 | three EDA checks declared with no implementation | `design/v1_3_scope.md:483` | defect | "`cross_modal_registration` ... declared in the contract, **no implementation**" | open |
| 18 | a declared proposal exclusion is not implemented | `design/v1_3_scope.md:562` | defect | "nothing in the package computes a mitochondrial fraction" | open |
| 17 | the spatial check computes a null and throws it away | `design/v1_3_scope.md:541` | defect | "the `z` and `p` it computed are thrown away, so `n_perm` changes the runtime and nothing else" | open |
| 13 | the gene-mapping release is unrecorded and unrecoverable | `projects/tnbc-92/cohort.json:42` | defect | "UNRECORDED and not recoverable from the artifact ... the release is pinned by SHA rather than named." | open |

## paper and record - the user's call, not the agent's

| # | item | where | kind | the record's own words | status |
|---|---|---|---|---|---|
| 3 | v3 determinism check unsatisfied before publication | `projects/tnbc-92/v3_phase2_plan.md:183` | defect | "AUCs must agree to 4 decimal places ... **MUST FIX before publishing**" | open |
| 6 | the banned circular supervision is still live in a committed script | `docs/TODO_fusion_experiments.md:171` | defect | "REMOVE mc_weights from ST input (currently included as 14d in the existing script - this is the circular dependency violation)" | open - COHORT-SPECIFIC: mc_weights is this cohort's NMF ground truth. the general rule is G6 over the declared `supervision`, already in the package. check against the grid path per 9 |
| 9 | the legacy grid is not regenerable and is still cited | `src/omicstra/configs/data_contract.json:528` | defect | "an earlier grid exists whose pathway side is not regenerable. several older documents point at it rather than the live grid" | open - check the grid path before citing any number |
| 7 | the -1 sentinel fill is counted as a class | `src/omicstra/configs/data_contract.json:518` | defect | "the -1 fill reaches subject-level coherence evaluation as an extra class because consumers filter with .notna() and -1 is not NaN" | open - `adapters/evaluation.py` now reports the count; dropping them would move a published number |
| 8 | H2-C permutation scope may self-cancel | `src/omicstra/configs/data_contract.json:523` | defect | "permutation is performed at niche level while the label is per-subject; numerator and denominator may partly self-cancel" | unverified |
| 10 | permutation counts differ between jointly reported evaluations | `src/omicstra/configs/data_contract.json:538` | defect | "N_PERMS is 1000 in some evaluation scripts and 500 in others, so p-value floors differ between evaluations that are reported together" | open |
| 11 | a superseded encoder config is still in two scripts | `src/omicstra/configs/data_contract.json:548` | defect | "a superseded encoder configuration is still present in two scripts" | open |
| 26 | cited floors have no artifact | `projects/tnbc-92/routing_evidence.json:614` | unbacked-claim | "the bio/patient raw_he (0.137) and raw_st (0.088) floors ... exist in no machine-readable artifact ... declared, not chased." | OPEN |
| 27 | a published statistic has no implementation anywhere | `script_map/evaluation_maths.md:173` | unbacked-claim | "**grep finds no `welch` anywhere** ... the number is published and the statistic has no home." | open - `measures/evaluation.two_sample_z` now implements it; the published number predates it |
| 28 | a pilot headline number retired but never re-validated | memory `project_mc_weights_column_ambiguity.md:58` | unbacked-claim | "'Immune x MC7 = +0.77' should be treated as not reproducible and likely artifactual until re-validated" | open |
| 32 | the v3 DAG decomposition is deferred and three sources disagree | `projects/tnbc-92/evidence_sources.json:115` | unreconciled | "the report, deck and session record disagree on the numbers. resolve to one artifact before this row exists" | not_available / deferred |
| 36 | citation text blocked pending the user's own verification | gpath2vec memory `project_mc_citation_unresolved.md:9` | decision-not-acted-on | "asked me not to commit citation text until they verify which paper the actual NMF weights come from" | open (pending user) |

## gpath2vec

| # | item | where | kind | the record's own words | status |
|---|---|---|---|---|---|
| 1, 2 | `end2end` crashes; `_w` renamed by `itertuples` | `CURRENT.md:85`; `gpath2vec/cli.py:305` | defect | "`r._w` cannot resolve - pandas renames any leading-underscore column in `itertuples`. deterministic, input-independent." | **FIXED on `fix/v3.0.1` (6b1d9e9), verified end-to-end 2026-10-20: 680 pathways, 252 zero-sig, (823, 512) embeddings, 16.4 s. not yet merged or released** |
| 23 | the importable package and the distribution disagree on version | `gpath2vec/__init__.py:10` vs `setup.py:3` | defect | `__version__ = "1.0.0"` while the distribution says 3.0.0 | **FIXED on `fix/v3.0.1` (40d1da5), verified: both report 3.0.1** |
| 24 | Zenodo auto-archiving will not happen for future releases | gpath2vec memory `project_v3_release.md:9` | defect | "the webhook was never enabled, so future releases will NOT auto-archive; a new version must be uploaded by hand" | open - 3.0.1 needs a manual upload into concept DOI ...141 |
| 25 | an abstract sentence rests on a lost artifact | gpath2vec memory `project_manuscript_status.md:13` | unbacked-claim | "image and code are GONE ... The abstract sentence \"in controlled tests ... recovers the expected relationships\" is currently unbacked" | open / must-do |
| 29 | a cited reproducibility test does not test what is claimed | `gpath2vec/paper_draft.md:78` | unbacked-claim | "We verified this end-to-end and ship the verification as a unit test (`tests/test_metapath_topology.py`)." | resolved-but-unverified |
| 35 | the level filter prunes the hierarchy it is meant to describe | gpath2vec memory `project_aucell_standardize.md:14` | deferred | "the `level` filter prunes parent pathways from the graph backbone in net.py, gutting the hierarchy ... was raised and PARKED per user." | deferred / parked |
| - | no seed-stability analysis of the 512d space | gpath2vec `MEMORY.md`; no stated gap found | defect | searched both repos for seed stability / sensitivity / across seeds / seed sweep: nothing. only two builds exist, both `_s1234` | open - the only genuinely new item from the 2026-10-20 review. one section at two more seeds is ~16 s each; compare with `linear_cka` |

## deferred, cohort two is the whole claim

| # | item | where | kind | the record's own words | status |
|---|---|---|---|---|---|
| 34 | the second cohort's evidence pack did not ship | memory `project_session_handoff.md:20` | deferred | "the second cohort's evidence pack did NOT ship and is 1.3's whole claim." | open / deferred |

## found while building, not yet acted on

| # | item | where | kind | the record's own words | status |
|---|---|---|---|---|---|
| 41 | a scaffolded cohort writes under the PACKAGE, not its own project root | `src/omicstra/settings.py:121` `work_root` | defect | "the fallback resolves under `repo_root`, which is right for a source checkout and wrong for an installed server" | open - the default cannot simply be changed: several tests isolate by redirecting `runs_dir`, and a declared `work_dir` ignores that redirect, so a seed-cohort test would write into the ledger a report reads. fix belongs with the isolation mechanism |
| 42 | `omicstra inventory` overwrites the committed full-cohort record with a capped run | `src/omicstra/cli.py:213` `inventory` | defect | `--max-samples 2` rewrote `projects/tnbc-92/inventory.json` from 280 samples to 2, and the only warning was "CAPPED" in the console | open - a capped run should not be able to replace an uncapped record. either write capped output to a separate name or refuse the overwrite |
| 43 | the CLI step chains write no ledger records | `src/omicstra/cli.py` `inventory`, `eda` | gap | `inventory` ran 8 steps and `runs/{id}/records/decisions.jsonl` stayed empty; only the graph arms persist, through `mcp/server.py:_persist` | open - so `report`, `trace` and `omicstra notebook` see nothing for a cohort driven from the CLI. the one-ledger rule covers the arms and not this path |

## hygiene

| # | item | where | kind | the record's own words | status |
|---|---|---|---|---|---|
| 30 | a fixed defect still reads as open in a live doc | `src/omicstra/configs/data_contract.json:533` | doc-drift | "FIXED 2026-08-24 - docs/data_structure_flow.md still lists this as open and is stale on that row" | resolved-but-unverified downstream |
| 31 | `knowledge/` is gitignored while three docs call it committed | memory `feedback_scripts_dir_ignored.md:20` | unreconciled | "`knowledge/` IS still gitignored, while `README.md`, `CLAUDE.md` and `design/mcp_plan.md` all describe it as committed. unreconciled" | open |

## settled - do not raise

| # | item | where | kind | the record's own words | status |
|---|---|---|---|---|---|
| 37 | the released tag points at a tree lacking the fixes | gpath2vec memory `project_v3_release.md:13` | deliberate | "the `v3.0.0` tag ... points at `31430b7`, which LACKS the aucell zscore fix" | **settled - "Do NOT re-raise or 'helpfully' retag"** |
| 38 | the stack table's model rows are kept on purpose | `design/_scratch/doc_drift_audit.md:154` | deliberate | "**model rows: WON'T FIX, decided 2026-08-24.** ... not drift, do not re-flag" | **settled - do not raise** |
| 39 | two TODO docs are parked by decision | `design/_scratch/doc_drift_audit.md:180` | deliberate | "`docs/TODO_mihc_modality.md`, `docs/TODO_gpath2vec_notf.md` \| - \| parked \| deliberately" | **settled - do not raise** |
| 40 | the F2 topology bug is pinned by a test rather than fixed | `gpath2vec/tests/test_metapath_topology.py:33` | deliberate | "this is the F2 bug paper section 4.5 inadvertently claims is fixed." | **settled - pinned pending the F2 decision** |
| 12 | the H&E tile is smaller than the spot it represents | `projects/tnbc-92/platform.json:365` | deliberate | "REAL. ... the tile is smaller than the spot it represents." | **as-built, already declared** |
