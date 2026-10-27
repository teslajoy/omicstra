# changelog

## v1.3.0a2 (2026-10-27)

alignment chain integration. dispatch is configuration-driven: each run config
declares its method and `run_align` routes by method string, not by experiment ID
prefix. classical baselines (B1-B3) run as a four-step LCEL chain reusing steps
from the contrastive chain. `save_test_artifacts` moved to
`adapters/alignment_data` so the classical chain runs without torch.

## v1.3.0a1 (2026-10-02)

adds the worker that serves the encode queue, a verdict per hypothesis behind its
own gates, one decision ledger that reaches the report, and the whole chain wired
at level 0. the server holds no model, and that is now a test rather than a
claim: it imports no model SDK, no extra can install one, and both judges resolve
from lookup tables. it has been driven end to end from a Claude-on-Bedrock
client - from a workstation and from a scheduled cluster node - where three runs
took three different routes through the surface and every server answer was
byte-identical to the same tool called with no model on the path.

## v1.2.0 (2026-10-02)

makes both modality agents reachable by a client, with a run handle that survives
the process, and runs a cohort the package had never seen through them. the agents
are separate on purpose: the H&E arm asks about gated weights and a tile geometry,
the ST arm about the encoder's prototype floor and the edge scale, and one tool
would hide what each exists to ask. a cohort with no recorded evaluation of its
own is not routable, and the server says so rather than reusing the seed cohort's
winner. the second cohort's own evidence pack is 1.3.

## v1.1.0 (2026-09-11)

`omicstra run`: embeddings extracted here rather than resolved, the niche join,
H1/H2/H3 as a chain, and a promotion step where a person approves the pack, with
the report rendered from a cohort-agnostic template. laptop-only for both.

that report is a template, not this cohort's report. the renderer takes a pack and
a record ledger and knows nothing about tnbc-92 - sections appear because a record
exists, so a cohort that never ran H3 has no H3 section and says why.

## v1.0.0 (2026-08-14)

ships the read path (inventory, EDA, admissibility gate, routing) over stdio and
HTTP, the alignment and evaluation stages verified bit-identical against the
published grid, and the seed cohort's evidence pack in the repo, so a fresh clone
can route without training anything. the package itself ships no cohort:
`pip install omicstra` gets the contracts, and you point it at a cohort directory.