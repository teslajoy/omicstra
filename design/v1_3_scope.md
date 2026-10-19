# 1.3 scope

> **a cohort the package had never seen goes from files to ITS OWN EVIDENCE PACK
> AND REPORT, through the same surface a client uses - and the pack states which
> rows it cannot fill and why.**

this is 1.2's opening line, unchanged. 1.2 shipped the first two thirds of it:
files to embeddings through the surface, for a cohort with no evidence of its
own. the pack is the third, and it is what makes the claim worth making - a
second cohort that embeds proves the adapter; a second cohort that produces its
own evidence and names its own refusals proves the product.

the clause after the dash is new and is the honest part. the second cohort
brought no spot labels and no pathway ground truth, so most rows cannot be
filled. a pack that omitted them would read as an incomplete analysis; a pack
that states them as refusals with their cause is the result.

---

## what 1.2 established that this builds on

| | |
|:--|:--|
| both modality agents reachable by a client | `he_encode`, `st_encode`, each with its own gate set |
| a run handle that survives the process | sqlite checkpointer, kill-and-resume on both executors |
| the ST port is exact | five subarrays byte-identical to the Novae cache |
| the EDA chain runs | first execution on any cohort; its table is measured |
| every refusal is named | 7 task families refuse on the second cohort, each with a cause |

---

## in scope

```
4  downstream reachable   align.run / evaluate.run behind the same run handle.
                          blocked in 1.2 by the EDA gate, correctly - this repo's
                          first constraint is that nothing runs past a failed gate
6b eda, the remainder     the 4 undeclared checks and the 3 unimplemented ones
5  the pack               promote + report for the second cohort. THE release
                          claim; everything else is in service of it
7  pathway agent          a package path for gpath2vec
```

## out

a third cohort. slurm. oauth. anything that makes the first cohort better.

---

## the build order inside 1.3

five items, in the order their dependencies allow. the state column is what has
been demonstrated, not what has been written.

| | | state |
|:--|:--|:--|
| A | `omicstra worker` - the process that serves the encode queue | **done** |
| B | `evaluate_run` behind the run handle, and `hypotheses` leaves `align_run` | **done** |
| C | one record store, and the records reach the report | **done** |
| D | level 0 wires align and evaluate; `design/graph.md` is the route | **done** |
| E | Bedrock, as a test rather than a build | **done** - 7 tool calls, 0 mismatches |

### A, and what it closed

submit-only was half a claim. `he_encode` and `st_encode` hand the shard list to
the scheduler and return at submission, which is only durable if some process is
polling - and until now that process existed solely inside the kill test, as a
worker the test assembled itself. so the test could pass while the command an
operator would run did not exist.

`omicstra worker` is that command. it holds both activity bodies by name, needs
no cohort and no project root, and stops gracefully on SIGTERM so the shard in
flight reports rather than timing out into a retry. `--max-seconds` exists for a
worker inside a job allocation with a wall clock, which is the ARC case.

two rules came with it, because a name crossing a scheduler is checked by nothing
in the language:

```
dispatch/bodies.py:shard_bodies()     the one name table. the submitter names a
                                      body, a different process holds it
test_rule_lints.py                    every fn_name= in graphs/ is a key in it.
                                      an arm that forgets the line fails a test
                                      rather than orphaning runs on a queue no
                                      worker can serve
```

the acceptance is the existing durable test, with the shipped entrypoint in place
of its inline worker: the submitter exits having encoded nothing, a worker it
never met finishes all four shards, and the handle still resolves from a third
process.

still untested beyond localhost - the kill test runs a dev server in-process, and
the `Temporal beyond localhost` line below is unchanged.

---

### B, and what it found

`align_run` accepted a `hypotheses` list and ignored it. the align graph resolves
a grid and scores nothing, so naming a hypothesis changed no behaviour while
implying the stage had returned a verdict over it. the parameter is gone and the
verdict has a stage of its own.

`evaluate_run` is judge #2's surface. it reads the artifacts the cohort's
`evidence_sources.json` names, runs the guards `routing_contract.json` requires
per task family, and states one of four things per hypothesis. no model is called
anywhere on the path, so `modelInsideServer: false` still holds.

three gates, all declared in `compute_contract.json` under `stage: evaluate`:

```
hypothesis_scope   which hypotheses this run answers. a hypothesis whose every
                   family is undeclared or declared unavailable is NOT a gate -
                   nothing to choose - and is reported not_run with its families
artifact_pinning   the sha256 of every cited artifact, and the hashes travel
                   WITH the question. a declared pin that does not match is a
                   REFUSAL: the pack's numbers were read from other bytes
guard_failure      a required guard that RAN and failed. demote, do not discard
```

the verdict is computed BEFORE the gates, which is the opposite of the encode
arm. `guard_failure` can only fire on a guard that ran, and the guards cost a few
file reads rather than a training run, so there is nothing to protect by
deferring.

**four claim states, and the fourth is the finding.** `biological`,
`diagnostic`, `not_run` - and `unguarded`, which had to be added because "no
guard failed" was true of a hypothesis where no guard ran at all, and read as a
clean bill of health for a number nothing checked. on the seed cohort:

| | claim | why |
|:--|:--|:--|
| H1 | `unguarded` | `cross_subject_eval` needs a within-subject score to compare against; nothing in the declaration carries one |
| H2 | `diagnostic` | H2-C's G3 ran on all 7 arms in `metrics_h2_ci.json` and **7 of 7 failed** - no arm suppresses patient identity below biology |
| H3 | `unguarded` | the pathway gate needs each arm's ST features, which is per-arm cohort knowledge and is declared, never guessed. `evidence_sources.json#run_st_features` is the pointer and the seed cohort does not carry it |

15 cited artifacts, all present, **none pinned**. so the pin gate asks on the one
cohort that has a published grid, which is the right answer: the pins were never
written down. answering `pin_these_bytes` records the 15 hashes as a PROPOSAL in
the ledger - the package does not write a cohort's declaration, because a pin is
a committed human act.

two artifacts the pack's prose names and nothing locates: `biology.parquet` (it
is per arm) and `perm_nulls.parquet` (a sibling). a bare filename is reported
rather than resolved against whatever directory appeared nearby in the same
sentence - that guess gets the shared file right and the per-arm one wrong.

**two gaps the contract now states rather than hides.** it requires seven guards;
this package implements six checks; the overlap is five.
`routing_contract.json#guard_implementations` declares the map with a reason per
guard: `raw_floor` is each task's `floor` field, computed by promote from the
cohort's declared floor reader, and a guard that re-derived it would be a second
answer to one question; `untrained_control` is an ARM of the grid, not a check
over a number, and its absence from a rollup is what `coverage()` reports.

### three things B did not close

```
the rollup coverage gaps     H2 drops B4_v3 and H3 drops R5_v3, which the
                             verdict now REPORTS per hypothesis. closing them
                             means scoring those arms, which is a compute run
run_st_features              one declaration away from H3 being guarded rather
                             than unguarded. it is the cohort's to write
no reader for four families  cross_modal_retrieval, tissue_state_grouping,
                             morphology_decode and pathway_discrimination report
                             `no_reader`: promote computes their candidates from
                             the same declaration, and what is missing is a
                             CHECK over the number, not the number
```

---

### C, and the three divergences it found

there were two record stores. the routing path appended to a file-backed
`RecordStore` at `runs/{id}/records/decisions.jsonl`; every gate arm wrote plain
dicts into the langgraph checkpoint, readable only through `runs_record(run_id)`
and only by a caller holding the handle. so a cohort's ledger said "N routing
decisions" while the gates, the dispatches and the pins were elsewhere - and
`report.py`, whose own first line is *"a pack and a ledger in, markdown out"*,
read the pack and no ledger at all. `section_provenance` was already printing
`len(records)`; nothing ever supplied the key.

the arms' dicts could not be VALIDATED as Records, which is why nobody had tried:

```
actor      records.py said deterministic | model | human, and the gate nodes
           have been writing `client` and `system` since the actor vocabulary
           was closed. `actor_ratio()` initialised three keys and did
           out[r.actor] += 1, so the first client record would have raised
kind       no `dispatch`. handing N shards to an executor produces no artifact
           and judges nothing, so it is neither a transform nor a diagnostic
chosen     a str. a gate arm answers every open gate in ONE call, which is a
           mapping, and flattening it loses which answer belonged to which gate
```

three divergences between a model and the only code writing to it, none of them
caught, because nothing read both. that is the whole argument for one store.

what now holds it: `records.coerce()` is the door - unknown keys land in
`params`, which is the field that exists for them, and `record_id` is content
addressed over the decision with the clock excluded, the same exclusions
`stable_id()` already stated. `RecordStore.append_new()` is idempotent on that
id, which matters because the arms carry their whole record list forward in
state: without it a three-gate run resumed twice would read as nine decisions,
and the actor ratio is a published number.

`_persist` is called from `_handle`, the one place every arm and every resume
passes through, so there is one sink rather than a call per node. the handle
returns `recorded: N`.

**two views, not two stores.** `runs_record(run_id)` still reads the run's
checkpoint, because "what did THIS run decide" and "what has this cohort
decided" are different questions - and it now also names the cohort's ledger and
its size, so a reader is not left thinking the handle is the only way in.

### what reading the real ledger then showed

with the arms appending, the seed cohort's `decisions.jsonl` was opened and
counted. 1,846 rows for seven decisions:

```
995  route::tissue_state_grouping     one decision
775  route::cross_modal_retrieval     one decision
 20  route::subject_identity_suppression
...
```

`routing.py:emit` called `append`, not `append_new`, so every invocation of the
same question against the same pack wrote another row - and `record_id` is
content addressed over the contract version, the pack version and the request
precisely so that "two ledgers join on this id and either differ on outcome, or
do not differ". it does not count calls. the actor ratio is a published number
and cannot be a call counter, so the routing path appends the same way the arms
do now. the existing rows are left where they are; they are a record.

and two tests had been writing into the repo's own `runs/` - 42 rows under
`synthetic/`, 5 under `tnbc-92/` - because `settings.runs_dir` resolves under the
package repo root and a subprocess inherits it. invisible: the files are
git-ignored, nothing failed, and the next report would simply have been wrong
about who decided what. `tests/conftest.py` now points `runs_dir` at tmp for
every test, by fixture and by environment variable so a subprocess gets it too.

### TODO · two package defaults the threshold lint does not yet cover

`tests/test_rule_lints.py` asserts an analysis step holds no threshold of its
own. it is scoped to modules that name AND define a chain, so two existing
defaults are outside it and are recorded here rather than rewritten - changing a
passing gate to satisfy a new lint is how a rule becomes a refactor.

```
eda.py:430        max_pct_negative=c["params"].get("max_pct_negative", 10.0)
                  a cohort that declares `negative` but not `max_pct_negative`
                  silently inherits 10%. the two-argument `.get` is the pattern
                  the rule exists to stop: a default by another name

inventory.py:455  PITCH_TOLERANCE = 0.02
                  a module constant deciding whether a measured lattice pitch
                  AGREES with the declared one. 2% was chosen on one platform's
                  lattice and is applied to every cohort's geometry check
```

both are the third kind of threshold in `EVALUATION.md` section 8 - a bar chosen
by looking, with no external basis and no null behind it. the fix is the same
for each: read it through the declaration, refuse when absent, and let the seed
cohort declare what it has been running at. the lint's scope then widens to
`eda.py` and `inventory.py`.

---

### TODO · a cohort has no declared WRITE root

`settings.runs_dir` resolves under the PACKAGE repo root. that is right for this
repo, where the seed cohort's `project.json` declares its data and runs at the
root explicitly, and wrong for an installed server with no repo: it puts a
cohort's ledger next to site-packages. four call sites read it - `ledger()`,
`report.py:_ledger_path`, `figures.py`, and the `decisions` CLI command.

**the obvious fix is itself a bug, which is why this is written down.** the first
version of this entry said "have `ledger()` prefer the cohort's own declared runs
root". but `project.json#runs_dir` points at the PUBLISHED GRID - the seed cohort
declares `../../runs/tnbc-92_v3`, which `read_only_inputs` lists under
`policy: refuse_writes`. aiming the server's one writable artifact at the oracle
every port is verified against would be refused by the dispatcher, correctly.

so `runs_dir` means **where the finished grid is read from**, and what is missing
is a separate **where this server may write**. the two have been conflated
because on this repo they coincide under the root.

```
1  a declared write root, optional: project.json#work_dir
2  undeclared falls back to today's behaviour, so nothing moves and no
   existing ledger orphans
3  the four call sites above
4  REFUSE a work root that falls inside a declared read-only path.
   `_refuse_protected` and `assert_writable` already exist; this is the part
   that makes the fix worth more than moving a path
5  tests: resolution with and without the declaration, and the refusal
```

half a day, low risk. **land it before anything that writes new per-cohort
artifacts**, so that work targets the right root rather than needing a migration
afterwards.

**DONE.** `settings.work_root` resolves `project.json#work_dir`, falling back to
the historical path so nothing on disk moved, and `assert_work_root_writable`
refuses a work root inside a declared oracle. all four writers go through it, and
a test asserts no module resolves `runs_dir` directly any more.

### TODO found while doing it · a malformed declaration protects nothing

`dispatch.read_only_inputs` catches every exception from `ProjectConfig.load`
and returns `[]`, with the comment "undeclared is fine". undeclared IS fine - a
cohort with no published results has no oracle - but the same branch swallows a
cohort whose `project.json` is present and UNPARSEABLE, and that cohort then has
no write protection at all while appearing to have declared some.

the two cases need telling apart: absent file -> no oracle; present but
malformed -> refuse, because a typo in a declaration must not silently remove a
guard. pre-existing, and not folded into the work-root change.

---

### D, and where the wiring was wrong

align and evaluate were built, tested and connected to nothing - the same thing
that had happened to `graphs/encode.py` a release earlier, when the slot existed
in `build_omicstra_graph` and no caller filled it. level 0 now takes all five
stages and chains them:

```
discover --compute--> eda -> carry_counts -> encode -> st -> align ─┐
         --ask-----> route ─────────────────────────────────────────┤
                                                                    ▼
                                                            evaluate -> END
```

**the chain stops at the first stage that leaves work in flight.** `submitted` is
the usual reason and is why this is not a straight chain: an encode with
`compute=True` returns at submission, so the embeddings the next stage reads are
not on disk when the next node would run. walking on would have align report
`unavailable` on a run that is working correctly. `graph.py:STOP_AFTER` is that
list; the run handle is how the chain continues.

**evaluate on the compute arm alone was wrong, and the first walk found it.**
evaluate reads the artifacts the evidence pack cites, and a cohort WITH a pack
takes the ASK arm - so the first wiring put the one stage that needs a pack on
the one arm that is taken when there is none. the seed cohort walked, took `ask`,
and reached END with no verdict. it is terminal on both arms now: the verdict on
the ask arm, a named refusal on the compute arm.

a second thing the walk found: the routing subgraph states its outcome in
`resolution` rather than in a report, so the stage keeper logged the one stage
that had actually decided something as `unknown`.

`omicstra chain` is the caller that holds the whole thing, which is what the
question *"can you ask it the three hypotheses and have it go through the chain"*
was asking for:

```
omicstra chain --project-id tnbc-92 --task cross_modal_retrieval \
  --answer artifact_pinning=evaluate_unpinned \
  --answer guard_failure=report_as_diagnostic

  arm: ask - cross_modal_retrieval has recorded evidence on this cohort
  route        recommend
  ended at: evaluated
  H1     claims unguarded    guards ran 0, failed 0
  H2     claims diagnostic   guards ran 9, failed 7
  H3     claims unguarded    guards ran 0, failed 0
```

without `--answer` it stops at the gate and prints the options, because the
system does not pick.

**`design/graph.md` is the route**, one page, and a lint asserts every
`graphs/*.py` appears on it in the order the code wires them. a diagram that
drifts from the wiring is worse than none - it is read as authoritative and is
wrong.

### the one key that means two things

`arm` is shared by name between level 0 and every subgraph:

```
level 0      ask | compute | refuse        which arm answers the question
a subgraph   he | st | align | evaluate    which arm the RUN HANDLE belongs to
```

so a subgraph wired under level 0 receives `compute` in the field the MCP reads
as a handle arm, and `ARMS.get(arm, ARMS["he"])` would have re-entered the H&E
graph - a tile geometry resolved for a cohort nobody asked about one, under
someone else's handle. `_runner_for` raises `NotAnArmHandle` now, and the three
handle tools return it as a refusal rather than a traceback. both names are
published on their own side, so this is named rather than renamed.

---

### E, and the bug it found before Bedrock was even called

E is a test rather than a build, and it splits into a half that can run with no
credentials at all and a half that needs a model client.

**the half that runs here.** `modelInsideServer: false` is a claim in the
registry entry, so it is now a test: no module in the package imports a model SDK
(anthropic, boto3, langchain_aws, ... - langgraph and langchain-core are state
machines and are not on the list), no model SDK is a declared or optional
dependency, and both judges are deterministic by source. a claim like that is
false the moment somebody imports an SDK to try something, which is why it is
checked rather than trusted.

**the half that needs a model client.** `scripts/probe_bedrock.py` starts the server
over stdio, hands its own tool list to Bedrock's Converse API as a toolConfig,
executes whatever tool the model asks for, and then calls the same tool with the
same arguments IN PROCESS with no model anywhere - and diffs the two. the diff is
the test: a difference means something on the server's path is not a lookup.
`--dry-run` does everything except call Bedrock and passes here. boto3 is
deliberately not a dependency of this package in any extra; a model client
belongs to the client.

**what the dry run found: `python -m omicstra.mcp.server` served 9 of 18 tools.**
the `if __name__ == "__main__": main()` block sat in the MIDDLE of the module,
above the resources and the nine compute tools, so running it as a module called
`main()` before those decorators had run. a client discovered half a server and
nothing said so. the README's documented command happens to be
`python -c "from omicstra.mcp.server import serve; serve()"`, which imports the
whole module first and was therefore correct by accident - which is why no test
caught it: every other test imports `srv` and sees all 18.

the guard is last in the file now, and two tests hold it there: one asserts the
two start paths serve the same tool set, and one asserts the guard is the last
thing in the module, so a tool appended after it fails a test rather than
vanishing from the transport.

**it ran**, on a Bedrock-backed client, twice - and the two runs took DIFFERENT
paths through the surface, which is the interesting part: 7 tool calls then 5,
and 0 mismatches both times.

the model chose a different route through the surface each time and **every
server answer was byte-identical to the same tool called in process with no model
anywhere**. that is the claim `modelInsideServer: false` makes, tested rather
than asserted: the model picked which tool to call, the server picked what the
answer was.

it also answered gates - `verdict_disagreement`, `artifact_pinning`,
`guard_failure` - which is exactly the narrow job the registry entry describes,
choosing among enumerated options. the answers are recorded `actor: client`,
because a tool call is what the server saw.

three things the probe found before Bedrock was even reached, and one after:

```
9 of 18 tools served     the __main__ guard sat in the MIDDLE of server.py, so
                         `python -m omicstra.mcp.server` called main() before
                         nine decorators had run. the README's documented command
                         uses `python -c "... serve()"` and was correct by
                         accident, which is why no test caught it
an invented model id      `us.anthropic.claude-sonnet-5-20260514-v1:0` does not
                         exist. list-inference-profiles is the authority, and the
                         real id carries no date suffix
temperature rejected      "`temperature` is deprecated for this model". dropped;
                         what determinism this test rests on is the server's
a 60-line ExceptionGroup  an invalid model id and a missing invoke permission
                         read nothing alike and both arrived as the same wall of
                         traceback. one line now, and an invalid id lists the
                         profiles that are actually available
```

a scheduler run is still outstanding, and compute does not belong on a login
node. that is the remaining half of the cluster line below.

---

## step 6's remainder, split by who can close it

| | check | what is missing | whose |
|:--|:--|:--|:--|
| declaration | `positive_markers` | a panel as `{gene: why}` | cohort |
| declaration | `negative_markers` | same, and whether `ERBB2` is a negative at all - the notebook calls it "low-moderate" and 6 patients are HER2 1+ | cohort |
| declaration | `spatial_autocorrelation` | a gene list, and the `I > 0.3` threshold's basis | cohort |
| declaration | `batch_structure` | FOUR: a gene universe to stack at all, then `sample_meta`, `technical`, `biological` | cohort |
| **package** | `cross_modal_registration` | declared in the contract, **no implementation** | package |
| **package** | `model_tissue_fit` | same | package |
| **package** | `multi_section_alignment` | same | package |

the three unimplemented checks are the ones with teeth. the seed cohort hid them
because every value in its `eda_summary.json` is hand-written and the file says so
(`produced_by: manual`). a second cohort has nobody to hand-write them, so they
are permanently unanswerable there unless something is built. deciding whether to
build them or to state them as structurally unanswerable is a 1.3 scope call and
is on the critical path to the pack.

---

## the level rule, restated - step 4 forced it

the rule said two things that step 4 put in conflict:

```
align and evaluate are LEVEL 2        protocols: order, applicability, authority,
                                      no maths, NO INTERRUPT
anything that can interrupt()         gets its own LEVEL 1 graph
```

`graphs/align.py` interrupts, so the second clause admits it and the first
forbids it. the resolution, and the reason:

**the gate is level 1; the protocol stays level 2.** `protocols/align.py` gained
`preflight_align`, which is deterministic, side-effect free and asks nobody - it
computes a gate list from a recorded verdict, exactly as `protocols/encode.py`
does. `graphs/align.py` holds the interrupt and nothing else. so align is level 2
and *has* a level-1 gate, which is what encode already was and what the rule
meant.

what would have violated it: putting the verdict decision inside
`protocols/align.py`, or letting the graph compute a metric. neither happened.

this is a restatement rather than a new layer, so the 2 sep stop condition holds:
no file moved level, and `graphs/` gained a module for a gate that already
existed as a sentence in CLAUDE.md.

### and `align_run` does not align

it RESOLVES. `compute=True` is accepted and reaches `run_align`, but alignment
has never executed through the package - the scripts are still the method, and
the 21 sep status on that is unchanged. the tool's description says so in its
first sentence. the claim "align runs through the surface" waits for a
`compute=True` run that reproduces a trained arm, which is 1.3 work and is not
done.

---

## open questions carried forward, with their evidence

### the spatial check thresholds a raw statistic and discards its own null

`measures.spatial_autocorrelation` computes Moran's I with a permutation null per
gene per section, keeps only `I`, and passes when 3 of the top 5 genes exceed
`I > 0.3`. the `z` and `p` it computed are thrown away, so `n_perm` changes the
runtime and nothing else.

this is the check whose own docstring calls it *"THE decision that changes the
pipeline's shape: it routes the molecular encoder to a spatial or a non-spatial
model"*, and `0.3` is the exploratory notebook's eyeballed cut-off. the project's
framing is that every number has a null; this one has a null and ignores it.
either the criterion should use the `z` already computed, or `n_perm` should go,
so nobody reads a null into the verdict.

### `negative_markers` discards 278 measurements because of 2

two sections carry neither `ESR1` nor `PGR`: `TNBC34_CN17_D2` (20,159 genes) and
`TNBC51_CN26_D1` (10,571 - the thinnest in the cohort). one unresolvable section
raises, and the aggregation turns that into `not_run` for the whole cohort check.
per-section `not_applicable` with the other 278 aggregated is the alternative.
the contract declares an aggregation rule per check, so this is a contract
decision.

### the mitochondrial-fraction exclusion is not implemented

the proposal lists *"ST spots with <200 UMI or mitochondrial fraction >25%"* as an
exclusion to be reported with counts. nothing in the package computes a
mitochondrial fraction. the two cohorts are asymmetric about it: the second ships
`pct_counts_mito` precomputed in `obs`, and the seed cohort's canonical `.h5ad`
carries no QC columns at all.

### the seed cohort's verdict disagreement, investigated

the align gate surfaced that tnbc-92 computes `proceed_with_caution` while its
summary declares `proceed`. the pack shipped under `proceed`, so this needed
checking rather than noting. from `run_gate('tnbc-92')`, 0 violations, 0
escalations, 3 cautions:

```
1  segmentation_qc: null          "cannot distinguish 'not applicable' from
2  multi_section_alignment: null   'not measured'; EDA.md schema has no marker"
3  risks[]: 4 entry(s) recorded and not marked accepted
```

**no biology hides here.** 1 and 2 are a SCHEMA gap - the summary has no way to
say "not applicable" - and the package's own graph already resolves what the
schema cannot express, reaching `not_applicable` on `segmentation_qc` with "no
segmentation masks - spot-resolution platform". 3 is the four `open_risks` the
summary itself records: Novae compatibility on this platform, Moran's I computed
on one subarray, MKI67's low I, and 17 subarrays under 500 spots. all four were
carried deliberately.

so the declared `proceed` is a person's judgement that those were acceptable,
and the computed `proceed_with_caution` is the absence of a marker for that
judgement. **two closable things follow:** give `EDA.md`'s schema a
not-applicable marker, and mark the four risks accepted with an actor. neither
changes a number.

### the gene universe, measured before declaring

| | sections | per section | intersection | union |
|:--|:--|:--|:--|:--|
| seed | 280 | 10,571 - 33,047 (median 25,041) | **7,288** | 54,714 |
| second, above the ST floor | 25 | 14,861 - 18,853 (median 17,627) | **0** raw / **11,284** mapped | 21,523 |

the second cohort's raw intersection is zero because 17 of those sections are on
Ensembl and 8 on symbols. so a universe rule has to be stated over **the
encoder's axis**, not over the sections' own `var_names`, or it is undefined on a
pooled cohort. union-with-zeros is the option to reject explicitly: a gene absent
from a section was not measured there, and a zero would read as a measurement.

---

## the three funded biological deliverables, audited against artifacts

the proposal names three outputs "beyond the reusable software infrastructure".
audited 2026-10-12 by opening each artifact. absences are stated, not explained.

| # | funded deliverable | state | artifact |
|:--|:--|:--|:--|
| 1 | integrated multi-modal cohort characterisation | **partial** | retrieval half: `runs/tnbc-92_v3/eval/H1/summary.json`, cited. comparative half: `eval/H2/mc_coherence.parquet` rows `raw_he` / `raw_st`, uncited |
| 2 | TME readability profiles | **partial** | `eval/H2/compartment_cosine.parquet` (108 = 12 compartments x 9 runs), `compartment_welch_contrasts.parquet` (54 = 6 contrasts x 9 runs). computed, uncited, and the prediction is inverted |
| 3 | pathway-morphology spatial maps | **partial** | association: `runs/tnbc-92/eval/H3/pathway_cca/dag_full/compartment_table.parquet` (395 nodes x 7 compartments), uncited. **no map in tissue space exists** |

**the deliverable-table rule:** every row names an artifact path or says "no home".
a row whose state is `built` must also be cited by the evidence pack - an artifact
nobody cites is not a delivered result. no row is `built`.

### 1 is broader than retrieval, which is why it is partial

verbatim: *"the shared embedding space combines morphological and transcriptomic
signals into a joint representation that is **more informative for TME
characterization than either modality alone**, enabling cross-modal retrieval
that links tissue architecture to molecular state across patients."*

retrieval is the second clause and is what the first ENABLES. the primary claim
is comparative, and the comparison exists:

```
                     ARI      silhouette
raw_he             0.1641       -0.0058
raw_st             0.0458       -0.0724
best aligned R5    0.2458       -0.0139
```

against the STRONGER single modality the gain is +50%, not the 5x against
`raw_st` quoted earlier in this session - the proposal says "either modality
alone", so `raw_he` is the bar. and on silhouette, which the proposal asks for
alongside ARI, raw H&E is **better** than every aligned run. neither number is
cited by the pack.

### 2's prediction is inverted, with the numbers

the proposal predicts *"morphologically distinct compartments (ex. tumor, TLS,
necrosis) will show higher matched-pair cosine similarity than morphologically
ambiguous compartments (ex. high-TIL vs. low-TIL stroma)"*. on R4_v3:

| distinct | ambiguous | mean distinct | mean ambiguous | n dist / n amb | z | direction |
|:--|:--|--:|--:|--:|--:|:--|
| Tumor | High TIL stroma | 0.5785 | 0.6175 | 3659 / 76 | -3.26 | **INVERTED** |
| Tumor | Low TIL stroma | 0.5785 | 0.6386 | 3659 / 268 | -15.41 | **INVERTED** |
| Lymphoid nodule | High TIL stroma | 0.6303 | 0.6175 | 24 / 76 | +0.68 | ns |
| Lymphoid nodule | Low TIL stroma | 0.6303 | 0.6386 | 24 / 268 | -0.55 | ns |
| Necrosis | High TIL stroma | 0.5947 | 0.6175 | 1018 / 76 | -1.89 | ns |
| Necrosis | Low TIL stroma | 0.5947 | 0.6386 | 1018 / 268 | -10.30 | **INVERTED** |

**zero of six support it. three inverted, three underpowered** - Lymphoid nodule
is 24 niches and High TIL stroma 76. the `direction` column has read INVERTED
since the parquet was written in May and has never been read into a claim.

**coverage.** `morphology_labels.tsv` carries `dominant_18class` with 18 `frac_`
columns; `compartment_cosine` covers 12. six absent: `Nothing`, `Artefacts`,
`Hole (whitespace)` are non-tissue, and `Tumor region`, `Nerve`, `Heterologous
elements` are tissue categories with no row. the proposal says 17 categories and
the artifact has 18; neither number is 12.

**the missing arm is B4_v3** - the untrained random-init control, 9 of 10 runs
present. that is the arm that WINS H2-B.2 at z=25.5, so the readability profile
omits the only run that beat every trained one on a related readout.

**cross-modal cosine is not a signal-to-noise ratio.** the proposal commits to
"per-compartment signal-to-noise analysis"; what exists is mean matched-pair
cosine per compartment with a Welch contrast. that is a different statistic and
goes in `projects/tnbc-92/proposal_deviations.md` as a substitution, not a
rename.

### 3 is planned, not built

the map belongs to `report.py`, not to a script: the renderer already takes a
pack and a ledger and emits markdown, and a figure a script writes separately is
a figure the pack cannot cite. the plan, deliberately one section and one
pathway:

```
input   compartment_table.parquet (395 nodes x 7 compartments)
        + projects/tnbc-92/data/canonical/{sid}_spots.parquet for x, y
scope   ONE pre-registered pathway of the five, ONE section, named in the
        pack rather than chosen at render time
output  a tissue-space panel the report cites by path, next to the
        compartment row it illustrates
not     a per-pathway atlas. the deliverable is "spatial maps generated showing
        pathway-morphology correspondence", and one correspondence shown
        honestly is that; 395 panels is a different artifact
```

---

## untested, and listed so it is not mistaken for working

```
embedding extraction on genuinely new data    the second cohort ran 3 sections
                                              of each arm. 3 is a smoke test, not
                                              a cohort run. the 25 above-floor
                                              original-ST sections have never
                                              been embedded, and nothing has run
                                              on a cohort outside these two
AWS / Bedrock                                 RAN, both halves - see E above. a
                                              second account or region is
                                              untested.
                                              see E above
ARC (the OHSU cluster)                        nothing has run through the
                                              scheduler. COMPUTE_BACKEND
                                              declares slurm and the durable
                                              executor works on Temporal locally;
                                              neither has run on ARC. the
                                              interpreter there is unknown, and
                                              the clone verifier now pins the CI
                                              matrix precisely because a local
                                              interpreter drifted to 3.15
Temporal beyond localhost                     the kill test runs a dev server
                                              in-process. no remote cluster
```

these are listed rather than planned. each would change what the release can
claim, and none is claimed now.

---

## grant commitments outstanding, not release work

```
Zenodo deposition        embeddings and the niche join. a data-sharing commitment
                         in the proposal and the only unchecked box in the
                         README's deliverables. the checkpoint and PyPI are done
H3 spatial maps          one of the three named biological deliverables. the CCA
                         landed, the maps did not
registration             declared, not built. both cohorts are same-section, so
                         nothing here needs it; a PDAC cohort with serial sections would
```

these are tracked so they are neither forgotten nor pulled into a release.

---

## what 1.2 cost in package edits, by the kind its own rule names

the rule: a change inside the package is a FINDING, and it is one of two kinds.

**cohort knowledge wearing a package's clothes** - none. the thing the rule exists
to catch did not occur.

**a path no cohort had taken** - all of them:

```
the compute arm had never run           hidden by the ask arm for a month
the encode node never dispatched        gated a run that never happened
a gate answered on the run reopened     a resume could never finish
the workflow id used hash(str)          salted per process; a resume could never
                                        find its own run
a failed shard read as still running    the in-process ShardReport was dropped
the floor gate measured nothing         no unit counts reached it; it read as a pass
a declaration could not reach its step  nothing filled params_space from a cohort
an empty marker panel passed            0 == 0
a panel could not meet its own matrix   symbols against versioned Ensembl
the gene axis was declared per cohort   it is a per-section fact the ingest observes
```

the last one is the only edit that began as a mistake of mine rather than a gap:
I asked the cohort to declare what the ingest already recorded, and the claim was
wrong for 68 of 108 sections of a cohort that pools three studies.
