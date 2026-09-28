"""LEVEL 1: the ST gate. its own graph, not a switch on the H&E one.

a generic `run_encode(modality=...)` was rejected for a reason that is visible
here: the two arms ask genuinely different questions. the H&E arm asks about
gated weights and a tile geometry; this one asks whether a section clears the
encoder's prototype floor, and refuses outright when the edge scale or the gene
axis is undeclared. one tool would hide the thing each agent exists to ask.

what is SHARED is the shape, and only where sharing hides nothing:

  gates / ask / halt   from `graphs.encode`. the node is generic - it calls
                       preflight with this arm's encoder, and preflight decides
                       applicability from the contract. the gate SET differs
                       because the encoder differs, not because the node does
  submit.dispatch      the executor choice, the handover and the ledger record.
                       both arms make the same two promises about recovery, and a
                       fix to one executor must not miss the other arm

what is NOT shared is this module's compute node, which resolves an edge scale
and a gene axis - neither of which exists on the H&E side.

two REFUSALS rather than gates, and the distinction is deliberate. a gate is a
question with a closed set of answers and a person to ask. an undeclared
`scale_to_microns` and an undeclared `gene_axis` are not questions: there is no
option list, only a value nobody has stated, and the only defaults available are
Novae's 1.0 (pixels read as microns) and "guess the identifier space". Both would
produce vectors that look fine and are not the cohort's.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from omicstra.graphs.encode import (
    EncodeState,
    _route_answer,
    _route_gates,
    ask,
    gates,
    halt,
)
from omicstra.protocols.encode import ComputeRefused, assert_clear, preflight


def st(state: EncodeState) -> dict:
    """the ST compute. refuses if anything upstream left a gate open.

    the refusal is belt and braces on purpose: the routing should make it
    unreachable, and a compute path that trusts its own routing is one edit away
    from embedding a cohort at the wrong edge scale.
    """
    import json

    from omicstra.adapters.canonical import list_samples
    from omicstra.graphs.submit import dispatch
    from omicstra.models.encoders import spec
    from omicstra.protocols.encode import (
        GeneAxis,
        StGraph,
        encode_st_cohort,
        st_shard_body,
        st_shards,
    )
    from omicstra.settings import settings

    encoder = state.get("encoder", "novae")
    reqs = preflight(state.get("cohort", {}), encoder,
                     unit_counts=state.get("unit_counts"),
                     min_scope=state.get("min_scope"),
                     answers=state.get("answers"))
    answered = state.get("answers", {})
    still_open = [r for r in reqs if r.open]
    if still_open:
        raise ComputeRefused(f"reached compute with {[r.id for r in still_open]} open")
    if not answered:
        assert_clear(reqs)

    es = spec(encoder)
    if es.gene_axis is None:
        return {"report": {"status": "refused", "encoder": encoder,
                           "why": f"{encoder} declares no gene axis, so it is not an ST "
                                  "encoder. the H&E arm is he_encode."}}

    root = settings.project_root(state.get("project_id"))
    plat = json.loads((root / "platform.json").read_text())
    pname = state.get("platform") or next(iter(plat.get("platforms", {})), None)
    if pname is None:
        return {"report": {"status": "refused",
                           "why": "no platform declared - the spatial graph is per platform"}}

    # the two refusals. they are reported rather than raised because a caller
    # asking "can this cohort run the ST arm" deserves the reason, not a traceback.
    try:
        graph = StGraph.from_platform(plat, pname)
    except ValueError as e:
        return {"report": {"status": "refused", "platform": pname, "why": str(e),
                           "undeclared": f"platforms.{pname}.st_graph"}}
    try:
        axis = GeneAxis.from_cohort(state.get("cohort", {}), want=es.gene_axis, root=root)
    except ValueError as e:
        return {"report": {"status": "refused", "platform": pname, "why": str(e),
                           "undeclared": "gene_axis"}}

    by_sample = plat.get("samples", {})
    wanted = set(state.get("samples") or [])
    triples, skipped = [], []
    for smp in list_samples(root):
        if by_sample and by_sample.get(smp.sample_id) != pname:
            continue
        if wanted and smp.sample_id not in wanted:
            continue
        if not (smp.counts and smp.spots):
            skipped.append({"sample": smp.sample_id,
                            "why": "no counts" if not smp.counts else "no coordinates"})
            continue
        triples.append((smp.sample_id, smp.counts, smp.spots))

    if not triples:
        return {"report": {"status": "nothing_to_run", "platform": pname,
                           "skipped": skipped,
                           "why": "no ingested sample on this platform carries both a "
                                  "counts matrix and a coordinates table"}}

    out = root / "data" / "embeddings" / f"{encoder}_spot"
    declared = {"platform": pname, **graph.params(), **axis.params(),
                "n_shards": len(triples), "skipped": skipped}

    if not state.get("compute"):
        return {"report": {"status": "ready", **declared,
                           "encoder": encoder,
                           "shards": [t[0] for t in triples],
                           "out_dir": str(out), "answers": answered,
                           "plan": state.get("plan", {}),
                           "note": "gates cleared and the shard list is resolved. pass "
                                   "compute=True to dispatch."}}

    shards = st_shards(triples, out, graph, axis, encoder)
    return dispatch(state, shards, out, st_shard_body, fn_name="encode_st",
                    cohort_run=lambda pending: encode_st_cohort(
                        [(s.id, s.params["counts"], s.params["coords"]) for s in pending],
                        out, graph, axis, encoder=encoder),
                    report_extra=declared)


def build_st_graph(checkpointer=None):
    """gates -> (ask) -> st, with halt on refusal. the H&E shape, its own compute."""
    g = StateGraph(EncodeState)
    g.add_node("gates", gates)
    g.add_node("ask", ask)
    g.add_node("st", st)
    g.add_node("halt", halt)

    g.add_edge(START, "gates")
    g.add_conditional_edges("gates", _route_gates, {"ask": "ask", "encode": "st"})
    g.add_conditional_edges("ask", _route_answer, {"encode": "st", "halt": "halt"})
    g.add_edge("st", END)
    g.add_edge("halt", END)
    return g.compile(checkpointer=checkpointer)


__all__ = ["build_st_graph", "st"]
