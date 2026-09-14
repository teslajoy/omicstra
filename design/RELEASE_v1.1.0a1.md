# v1.1.0a1 — compute-path groundwork

**A pre-release.** `omicstra run` — the command that *defines* v1.1 in
`design/v1_1_scope.md` — is not built. This is step 3 of that file's ten, tagged
so the work has a citable archive rather than because the milestone is done.

`pip install omicstra` still resolves to **1.0.0**; a pre-release is not
installed by default.

---

## what v1.0 shipped, unchanged

The read path — inventory, EDA, admissibility gate, routing — over stdio and
HTTP, with the alignment and evaluation stages verified bit-identical against
the published grid and the seed cohort's evidence pack committed, so a fresh
clone routes without training anything.

## what this pre-release adds

**The cohort, ingested once.** `scripts/ingest_wang.py` converts the seed
cohort's native R format to the package's declared input contract — `.h5ad` plus
a coordinates table — so nothing in `src/omicstra` imports an R reader or knows
this cohort came from R. 280 samples, 286,250 spots, raw integer counts with
versioned Ensembl ids. Verified against the cache that produced the published
grid: **0 coordinate mismatches across all 280**, and counts checked against R on
named cells, total counts and both dimensions.

**The H&E encoder as compute.** `run_one_he` is the whole H&E arm for one sample
with no cohort in it — tiles, then encode every tile, then pool over the niche.
The tile geometry is declared in `platform.json` rather than hard-coded, which is
how the 2026-09-11 pitch correction was found to have a second consequence: the
published tiles cover 95.9 µm, not the 128 µm the code constant named, so each
spot was clipped ~2 µm per side. Ported **as built** and declared, because the
ten-arm comparison is fair only while every arm saw identical tiles.

**An acceptance criterion for a transformer port.** Bit-identity is available for
the alignment stages — deterministic linear algebra, and they hit 0.00e+00 on
13/13 metrics — and is *not* available for a 631M-parameter float32 encoder
across backend kernels. The criterion is declared with its reason:

```
max |Δ| / row norm     <= 1e-5        observed 3.2e-06
min cosine per row     >= 1 - 1e-6    observed 1 - 2e-07
top-6 neighbour set    preserved for >= 99.9% of rows
```

The third carries the weight: H1 is a retrieval metric, so *same neighbours* is
the property the grid rests on, and it holds exactly while the vectors do not.
**5/5 oracle subarrays accepted on both CPU and MPS.**

**Preflight gates that resolve from declarations.** The compute contract's five
gates are computed from `cohort.json` plus the inventory's counts. tnbc-92
declares all five and fires **zero**, which is what lets an end-to-end run
complete unattended. `below_floor_policy: drop` is declared *as built* — the
Novae run manifest recorded `skip_too_small_for_novae: 19`, and the gate
recomputes the same 19 from the fresh ingest against the encoder's declared
512-spot floor. Two independent paths, one number, and a test that fails if they
diverge.

**A dispatcher that resumes.** One activity per subarray, three attempts, then
the shard is recorded failed and the run continues — one unreadable image must
not cost the other 279. Idempotent on the **output file**, not a ledger, because
the file is what the next stage reads. Two kill-and-resume proofs, and they are
different claims: locally the output file survives the process; on Temporal the
workflow history survives the **worker**, so a destroyed worker is replaced and
the run finishes from history with no resubmission. Without `TEMPORAL_ADDRESS`
the dispatcher runs in-process, and that is the default rather than a fallback.

## also

- `pyarrow` is now a declared `measure` dependency. The package's input contract
  names parquet and pandas pulls no engine, so `pip install "omicstra[measure]"`
  produced a reader that could not read the one format the contract names.
- `scripts/verify_like_ci.sh` runs the suite from a **clone** with only the
  declared extras. A green local run is not evidence: this machine has every
  extra installed and 151 GB of cohort data where the tests look.
- The DOI previously advertised in the README and `CITATION.cff`
  (`10.5281/zenodo.22666752`) was **reserved and never published — it returns
  404**. It has been removed rather than left pointing at nothing. The concept
  DOI minted by the GitHub–Zenodo integration replaces it.

## what is not here

`omicstra run`, the niche join as compute, the evaluation chain (six guards are
still `NotImplementedError`), the promotion step, and the report renderer. Steps
4 through 8 of `design/v1_1_scope.md`.

## verification

209 tests. CI green on Python 3.11, 3.12 and 3.14. Clone-verified with only
`measure,dev` installed, which is what an installed server actually has.
