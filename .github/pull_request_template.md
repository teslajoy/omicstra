## scope

| | |
|:--|:--|
| **scope step** | <!-- from the live design/v1_*_scope.md. if it is not a step there, say so --> |
| **funded item** | <!-- a row of the funded-deliverable table, or "none - why" --> |
| **claim it supports** | <!-- one sentence. "none" is a valid answer --> |
| **new graph / tool / module / layer** | <!-- none, or the name AND the ruling that admits it --> |
| **human decisions touched** | <!-- none, or which --> |

<!--
  this table is the review. it exists because tests catch correctness and not
  altitude: six of one session's findings were scope or semantics, and the
  cheapest place to catch "right code, wrong step" is before the diff is read,
  stated by whoever wrote it.
-->

## what it does

<!-- the change, and the evidence. numbers trace to an artifact path. -->

## checks

- [ ] `venv/bin/python -m pytest tests -q`
- [ ] `bash scripts/verify_like_ci.sh` — the clean clone, which catches tests guarded on committed records
- [ ] every number quoted above resolves to an artifact a reader can open
