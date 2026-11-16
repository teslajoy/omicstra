# 1.4 scope

> **a second cohort goes from files to its own evidence pack and report
> through the same MCP surface a client uses, without changes to `src/`.**

this is 1.3's opening claim, unchanged. 1.3 shipped the infrastructure for
it - all 11 workstreams in `design/completion_roadmap.md` closed - but the
claim itself did not land. the two remaining gaps are concrete and scoped:

- `align_run` executes through the package, not through scripts (open_items #33)
- a real second cohort produces its own routing evidence and report (open_items #34)

everything else is done. the three-layer architecture (schema standards,
source normalization, analytical bindings) is in place. the question-to-execution
path is wired. provenance reaches every record. the alternate-cohort
generalizability test passes on a synthetic visium fixture (WS 10, 7/7 tests).

---

## what 1.3 established that this closes

| | |
|:--|:--|
| schema standards written | `design/schema_standards.md` - layer boundaries, linkml as tooling not runtime |
| source mapping declared | `AlignmentContract.from_project_dir()`, `SourceMapping` per cohort |
| analytical bindings declared | roles in encode.py, evaluate.py; h2_label/supervision/headline_view default-free |
| question-to-execution wired | `execute()` -> `discover()` -> route -> plan or evaluate |
| pathway persistence | `save_scores` step, `aucell_v1.parquet`, `project.json#pathway_scores` |
| graph + provenance complete | WS 6-9: interrupt/resume, delivery bundle, _supplied() stamping |
| alternate-cohort test | WS 10: synthetic visium (uni2 1536d, leiden_cluster) loads cleanly across all adapters |
| wheel installs clean | `_package_repo_root()` falls back to `cwd()` - 62 wheel venv failures eliminated |

---

## in scope

```
F  align_run through the package    scripts/align.py and scripts/align_classical.py
                                    are still what produced the grid. align_run resolves
                                    artifacts but never trains. this gap means the package
                                    cannot reproduce its own grid without reaching outside
                                    itself. (#33)

G  second cohort evidence pack      a real cohort (not a synthetic fixture) goes through
                                    the full chain - ingest, encode, align, evaluate -
                                    producing its own routing_evidence.json and report.
                                    named refusals for rows it cannot fill. (#34)
```

## out

a third cohort. SLURM. OAuth. linkml entities implemented (schema_standards.md
says linkml is tooling - a build step - not a runtime dependency; the entities
come after the two gaps above are closed). anything that makes the first cohort's
results better.

---

## the build order inside 1.4

two items. F must precede G: the second cohort's evidence pack is only
meaningful if the package produced it.

| | | state |
|:--|:--|:--|
| F | `align_run` trains through the package; scripts are deprecated | **not done** |
| G | second cohort: ingest -> encode -> align -> evaluate -> pack -> report | **not done** |
| H | publish 1.4.0 to PyPI (`twine upload dist/*`) | **user action** |

### F — aligning through the package

the gap: `scripts/align.py` and `scripts/align_classical.py` produced the
published grid. the `align_run` MCP tool and `graphs/align.py` resolve that grid
from artifacts but have never trained a model. this means:

- the grid is not reproducible through the package
- the package cannot extend the grid to a new cohort
- open_items #33 tracks it; `proposal_deviations.md` records it as a scope deviation

the acceptance is `align_run(compute=True)` training the R1 contrastive arm on
the seed cohort and producing `runs/{id}/embeddings_test.parquet`, then diffing
its metrics against the published grid within the declared variance window.

### G — second cohort

the second cohort (PDAC or the next candidate) must:
- have an `ingest_{cohort}.py` script converting its native format to canonical
- have `project.json`, `platform.json`, `cohort.json` with all required declarations
- run through `omicstra chain` (or `ask`) without code changes to `src/`
- produce `routing_evidence.json` - even if most rows are refused with a named cause
- the pack states `open_flags` for each refused row; it does not omit them

the acceptance is the same as 1.3's stated acceptance: the second cohort's pack
is committed to `projects/{id}/routing_evidence.json` and differs from the seed
cohort's only in its own measurements. the report renders.

---

## stopping condition

F and G demonstrated. the wheel installs cleanly (`python -m build` + fresh
venv + full test suite). the second cohort's pack is committed. `twine upload`
ships 1.4.0 to PyPI.

a reduced run (synthetic fixture, partial grid) is not a candidate for winner.json
or for the 1.4 claim. the second cohort must be real and the grid must be the
package's output.
