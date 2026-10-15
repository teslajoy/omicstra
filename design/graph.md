# the graph

the route a question takes through this server, and which file owns each step.
one page, because the thing a session needs first is not the design rationale -
it is *where does this go*.

the level rule decides every placement on it:

```
a thing gets its own GRAPH only if it can ask a human.
everything else is a PROTOCOL.
```

---

## the whole route

```
                           omicstra.graph : build_omicstra_graph
                           LEVEL 0 - the only entry, the only checkpointer
                                            │
                                        discover
                      which arm answers THIS question, per task not per cohort
                                            │
             ┌──────────────────────────────┼───────────────────────────┐
             │ ask                          │ compute                   │ refuse
             ▼                              ▼                           ▼
     graphs/route.py                 graphs/eda.py                graphs/route.py
     JUDGE #1 - routing              the admissibility gate       reports the reason
     the only node that may          proceed / caution / stop     as a decision: the
     emit a SelectionRecord                  │                    contract declares no
             │                        carry_counts                method, so there is
             │                        counts the inventory        nothing to compute
             │                        already measured
             │                               │
             │                        graphs/encode.py  ──── H&E, one shard per section
             │                               │
             │                        graphs/st.py      ──── molecular, same shards
             │                               │
             │                        graphs/align.py   ──── the grid. RESOLVES; it has
             │                               │                never trained through here
             └───────────────┬───────────────┘
                             ▼
                    graphs/evaluate.py
                    JUDGE #2's gate. a verdict per hypothesis over the
                    artifacts the evidence pack cites
                             │
                            END

                           graphs/promote.py
                           the pack's gate. computed proposal in, prose out.
                           entered by `omicstra promote`, not by the chain -
                           curating a pack is a separate act from running one
```

**evaluate is terminal on BOTH arms, and the first wiring of it was wrong.** it
reads the artifacts the cohort's evidence pack cites, and a cohort WITH a pack
takes the ask arm - so hanging it off the compute arm alone put the one stage
that needs a pack on the one arm that is taken when there is none. on the ask arm
it states the verdict; on the compute arm it refuses and names the missing pack,
which is a true answer rather than a gap.

**the chain stops at the first stage that leaves work in flight.** `submitted` is
the usual reason: an encode with `compute=True` returns at submission, so the
embeddings the next stage reads are not on disk when the next node would run. the
run handle is how it continues - the shards land, and a caller re-enters at the
stage that needs them. with `compute=False` the whole chain walks and reports what
WOULD happen at every stage, which is what makes "ask the three hypotheses"
answerable in one call.

`graph.py:STOP_AFTER` is that list. `graph.py:COMPUTE_CHAIN` is the order.

`omicstra chain` is the caller that holds the whole thing. `--answer gate=option`
pre-answers a gate so the walk does not pause on it; without an answer it stops
at the gate and prints the options, because the system does not pick.

---

## the two judges, which are not the same thing

```
judge #1  ROUTING      graphs/route.py:judge        built, deterministic
                       maps a question onto a task family and resolves the
                       cohort's recorded evidence. the ONLY node permitted to
                       emit a SelectionRecord

judge #2  HYPOTHESIS   graphs/evaluate.py           the gate
                       protocols/evaluate.py        the maths: 6 guards, the
                                                    H1-H3 gates, 3 evaluate gates
                       no model is called on either path. `modelInsideServer:
                       false` is a claim about both
```

a verdict here is `biological` | `diagnostic` | `unguarded` | `not_run` per
hypothesis. the fourth state exists because *"no guard failed"* was true of a
hypothesis where no guard ran at all.

---

## every graph module, and the gate it holds

| module | level | its gate | refuses rather than asks |
|:--|:--|:--|:--|
| `graph.py` | 0 | none - it routes and owns the checkpointer | a task the contract does not declare |
| `graphs/eda.py` | 1 | the admissibility verdict | - |
| `graphs/route.py` | 1 | a contraindicated method | a cohort with no evidence pack |
| `graphs/encode.py` | 1 | weights, fit, floor, scope, capacity | a gate reached still open |
| `graphs/st.py` | 1 | the same set, molecular | an undeclared edge scale or gene axis |
| `graphs/align.py` | 1 | admissibility, verdict disagreement, grid scope | a cohort whose EDA gate never ran |
| `graphs/evaluate.py` | 1 | scope, artifact pinning, guard failure | no pack; a pin that does not match |
| `graphs/promote.py` | 1 | the pack's prose | an edited number |
| `graphs/submit.py` | - | not a gate: the shared dispatch tail | a write into a declared-read-only path |

`tests/test_rule_lints.py` asserts every row of this table exists as a module and
every module appears in this table. a new graph that forgets the line fails a
test rather than being discoverable only by reading the tree.

---

## the one key that means two things

`arm` is shared by name between level 0 and every subgraph, and the meanings
differ:

```
level 0      ask | compute | refuse          which arm answers the question
a subgraph   he | st | align | evaluate      which arm the RUN HANDLE belongs to
```

a subgraph wired under level 0 therefore receives `compute` in the field the MCP
reads as a handle arm. `mcp/server.py:_runner_for` refuses an arm it does not
know rather than defaulting to the H&E graph, which would resolve a tile
geometry for a cohort nobody asked about one. both names are published, so this
is named rather than renamed.

---

## where a record goes

```
every arm   ->  state["records"]  ->  _handle -> _persist
                                      runs/{project_id}/records/decisions.jsonl
                                      ONE ledger, idempotent on record_id

runs_record(run_id)   the RUN's checkpoint: what did this run decide
runs_trace(run_id)    the same records as a TIMELINE: what did it cost
report.py:gather()    the COHORT's ledger: what has this cohort decided
```

two views, one store.

### and every node is timed, at the point it is REGISTERED

`records.timed_node` wraps each `add_node`, not each call site. three rules fall
out of where it sits, and each one was a wrong answer first:

```
only the tail      a node returns the WHOLE accumulated record list, so
                   stamping all of it would attribute every earlier step's
                   cost to this one
never overwrite    a node that timed its own work keeps its number. the
                   measuring functions time the maths, which is tighter than
                   a node's wall clock - the wrapper fills a gap
started_at too     a node's raw dict is not a Record yet, and the default only
                   fires in `coerce` on the way into the ledger. without this
                   the checkpoint and the ledger disagreed about whether a step
                   had a time, and they are two views of ONE store
```

at the registration point rather than inside each node, because a node that
forgets to stop its own stopwatch reports a plausible wrong number, which is
worse than reporting none. `tests/test_one_ledger.py` asserts every `add_node`
in `graphs/` is wrapped, so a new node is timed without anyone remembering.

**an untimed step took an UNKNOWN time**, which is a different statement from
took no time. `runs_trace` names them rather than averaging a zero into a total.

---

## what is not on this page

`protocols/` is level 2 and holds no gate: order, applicability, authority, no
interrupt. `measures/` is the maths, callable from a notebook. neither appears
above because neither can ask anybody, which is the whole of the level rule.
