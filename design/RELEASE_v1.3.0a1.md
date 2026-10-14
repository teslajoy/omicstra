# v1.3.0a1 — a client reaches a verdict, and the server holds no model

**A pre-release.** The thing that *defines* v1.3 in `design/v1_3_scope.md` — a
cohort the package had never seen producing ITS OWN evidence pack and report —
is not built. That is step 5. This is the build order A–E of that file, tagged so
the work has a citable archive rather than because the milestone is done.

`pip install omicstra` still resolves to **1.2.0**; a pre-release is not
installed by default. `pip install --pre omicstra` gets this one.

---

## what v1.2 shipped, unchanged

Both modality agents reachable by a client, each with its own gate set. A run
handle that survives the process, proven by killing the server mid-shard and
finishing the run from a second process given only the `run_id`. The ST port
byte-identical to the Novae cache on five subarrays. The EDA chain running on a
second cohort for the first time, with seven task families refusing and each
refusal naming its cause.

## what this pre-release adds

**A process that serves the queue.** Submit-only was half a claim: `he_encode`
and `st_encode` hand the shard list to the scheduler and return, which is durable
only if something is polling — and until now that something existed solely inside
the kill test, as a worker the test assembled for itself. So the test could pass
while the command an operator would run did not exist. `omicstra worker` is that
command. It holds both activity bodies by name, needs no cohort and no project
root, and stops gracefully on SIGTERM so the shard in flight reports rather than
timing out into a retry. `--max-seconds` bounds a worker inside a job allocation
with a wall clock.

A name crossing a scheduler is checked by nothing in the language, so
`dispatch/bodies.py` is the single name table and a lint asserts every `fn_name=`
in `graphs/` is a key in it. An arm that forgets that line fails a test rather
than orphaning runs on a queue no worker can serve.

**A verdict is a stage with gates.** `align_run` accepted a `hypotheses` list and
ignored it — the align graph resolves a grid and scores nothing, so naming a
hypothesis changed no behaviour while implying the stage had returned a verdict
over it. The parameter is gone and `evaluate_run` is the stage. It reads the
artifacts the cohort's `evidence_sources.json` names, runs the guards
`routing_contract.json` requires per task family, and states one of four things
per hypothesis. It recomputes no published number.

Three gates, declared under `stage: evaluate`: which hypotheses are in scope; a
sha256 per cited artifact, travelling *with* the question, where a declared pin
that does not match is a refusal rather than a question; and what a required
guard that ran and failed does to the claim.

The verdict is computed *before* the gates, which is the opposite of the encode
arm and deliberate: a guard-failure gate can only fire on a guard that ran, and
the guards cost file reads rather than a training run.

**`unguarded`, the fourth claim state.** It had to be added because *"no guard
failed"* was true of a hypothesis where no guard ran at all, and read as a clean
bill of health for a number nothing had checked. On the seed cohort: H1
`unguarded` — `cross_subject_eval` needs a within-subject score to compare and
nothing declared carries one. H2 `diagnostic` — H2-C's G3 ran on all seven arms
in `eval/H2/metrics_h2_ci.json` and all seven failed. H3 `unguarded` — the
pathway gate needs per-arm ST features, which is cohort knowledge and is declared
rather than guessed.

The contract requires seven guards, this package implements six checks, and the
overlap is five. The gap is declared in `routing_contract.json` with a reason per
guard rather than discovered, because a required guard silently not run is the
one failure a guard library exists to prevent.

**One record store.** There were two: the routing path appended to a file-backed
ledger, every gate arm wrote plain dicts into the langgraph checkpoint readable
only by a caller holding the handle. So a cohort's ledger said "N routing
decisions" while the gates, the dispatches and the pins were somewhere else — and
`report.py`, whose own first line is *"a pack and a ledger in, markdown out"*,
read the pack and no ledger at all.

The arms' dicts could not be *validated* as records, which is why nobody had
tried. The actor vocabulary said `deterministic | model | human` while the gate
nodes have been writing `client` and `system` since that vocabulary was closed;
`kind` had no `dispatch`; a selection's `chosen` was a string while a gate arm
answers several gates in one call. Three divergences between a model and the only
code writing to it, none of them caught, because nothing read both.

**Level 0 reaches the verdict.** Align and evaluate were built, tested and
connected to nothing — the same thing that had happened to the encode gate a
release earlier. The compute arm is a chain now, and it stops at the first stage
that leaves work in flight: an encode with `compute=True` returns at submission,
so the embeddings the next stage reads are not on disk when the next node would
run. `omicstra chain` is the caller that holds the whole thing, and
`design/graph.md` is the route, with a lint asserting every `graphs/*.py` appears
on it in the order the code wires them.

**`modelInsideServer: false` is a test.** No module in the package imports a
model SDK, no declared or optional dependency is one, and both judges are
deterministic by source. The other half is `scripts/probe_bedrock.py`: it hands
the server's own tool list to a Bedrock-backed client, executes whatever tool the
model asks for, then calls the same tool with the same arguments in process with
no model anywhere — and diffs the two. Three runs took three different routes
through the surface, and every server answer was identical. The model picks which
tool to call and answers gates from their enumerated options; the server picks
what the answer is.

---

## the findings, which is the part worth reading

Every one is a path no cohort had taken, found by taking it.

```
9 of 18 tools served     the __main__ guard sat in the MIDDLE of mcp/server.py,
                         above the resources and the nine compute tools, so
                         `python -m omicstra.mcp.server` called main() before
                         those decorators had run. a client discovered half a
                         server and nothing said so. the documented command uses
                         `python -c "... serve()"`, which imports the module
                         first and was correct by accident - which is why no test
                         caught it: every other test imports `srv` and sees 18

evaluate on one arm      it reads the artifacts an evidence pack cites, and a
                         cohort WITH a pack takes the ask arm. the first wiring
                         put the one stage needing a pack on the one arm that
                         means there is none. terminal on both arms now

1,846 rows, 7 decisions  the routing ledger appended per CALL rather than per
                         record_id - 995 rows for one decision. a record_id is
                         content addressed over the contract version, the pack
                         version and the request, precisely so two ledgers join
                         on it. it is not a call counter, and the actor ratio is
                         a published number

one key, two meanings    `arm` is ask|compute|refuse at level 0 and the run
                         handle's he|st|align|evaluate in a subgraph, shared by
                         name. so a level-0 thread handed to a resume would have
                         re-entered the H&E graph - a tile geometry resolved for
                         a cohort nobody asked about one, under someone else's
                         handle. a refusal names it now

tests writing to runs/   two tests appended to the repo's own ledger, because
                         runs_dir resolves under the package repo root and a
                         subprocess inherits it. invisible: the files are
                         git-ignored, nothing failed, and the next report would
                         simply have been wrong about who decided what

a stale CI branch list   the release workflow listed its push branches
                         explicitly and the list had gone stale, so a pushed
                         branch got no CI at all with nothing saying so
```

## what is NOT in this pre-release

```
step 5, the pack         promote + report for a second cohort. THE 1.3 claim
step 6b, eda remainder   4 checks a cohort must declare, and 3 the package has
                         never implemented. the seed cohort hid them because
                         every value in its eda_summary.json is hand-written and
                         says so; a second cohort has nobody to hand-write them
step 7, the pathway arm  a package path for gpath2vec. the open branch holds a
                         port of the wrong chain and is not merged
alignment training       `align_run` RESOLVES. compute=True reaches run_align and
                         alignment has never executed through the package. the
                         scripts are still the method
the scheduler at scale   a compute node hosts the server and reaches a
                         Bedrock-backed client; a durable encode has never run
                         on a cluster, and no Temporal server runs there
```

## reproducing it

```
pip install --pre omicstra
python -m omicstra.mcp.server          # 18 tools over stdio
omicstra chain --project-id <id>       # the whole chain, resolve-only
omicstra worker                        # serves the encode queue, needs a scheduler
```

586 tests. A clean clone with only the declared extras installs, imports, serves
and routes without a tensor library — which is what `scripts/verify_like_ci.sh`
exists to keep true.
