"""LEVEL 1: the ST gate. its own graph, not a switch on the H&E one.

the arms ask different questions - gated weights and a tile geometry against a
prototype floor and an edge scale - so one tool would hide what each agent
exists to ask. shared: `gates`/`ask`/`halt` (generic; preflight decides
applicability from the contract) and `submit.dispatch` (both arms make the same
recovery promises). not shared: this compute node.

the edge scale and the gene axis are REFUSALS, not gates. a gate has a closed
option set and someone to ask; an unstated value has neither, and its only
defaults are novae's scale_to_microns=1.0 (pixels read as microns) and a guessed
identifier space. both embed noise that looks like signal.
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

    belt and braces: the routing makes it unreachable, and a compute path that
    trusts its own routing is one edit from the wrong edge scale.
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

    from omicstra.adapters.canonical import observed_gene_id

    by_sample = plat.get("samples", {})
    wanted = set(state.get("samples") or [])
    triples, skipped, graphs, axes = [], [], {}, {}
    for smp in list_samples(root):
        sid = smp.sample_id
        if by_sample and by_sample.get(sid) != pname:
            continue
        if wanted and sid not in wanted:
            continue
        if not (smp.counts and smp.spots):
            skipped.append({"sample": sid,
                            "why": "no counts" if not smp.counts else "no coordinates"})
            continue
        # both per sample: the edge scale is measured per section on one cohort,
        # the gene axis is observed per section on a pooled one. a section the
        # declarations cannot reach is skipped naming what is missing - never
        # borrowing another section's value.
        try:
            graphs[sid] = StGraph.from_platform(plat, pname, sid)
        except ValueError as e:
            skipped.append({"sample": sid, "why": str(e),
                            "undeclared": f"platforms.{pname}.st_graph"})
            continue
        try:
            axes[sid] = GeneAxis.resolve(state.get("cohort", {}), want=es.gene_axis,
                                         observed=observed_gene_id(root, sid), root=root)
        except ValueError as e:
            graphs.pop(sid, None)
            skipped.append({"sample": sid, "why": str(e), "undeclared": "gene_mapping"})
            continue
        triples.append((sid, smp.counts, smp.spots))

    if not triples:
        why = ("no ingested sample on this platform carries both a counts matrix and "
               "a coordinates table, a declared edge scale, and a gene axis this "
               "cohort's declarations can map onto the encoder's")
        undeclared = sorted({d["undeclared"] for d in skipped if d.get("undeclared")})
        return {"report": {"status": "refused" if undeclared else "nothing_to_run",
                           "platform": pname, "skipped": skipped, "why": why,
                           **({"undeclared": undeclared} if undeclared else {})}}

    out = root / "data" / "embeddings" / f"{encoder}_spot"
    first = triples[0][0]
    scales = sorted({g.scale_to_microns for g in graphs.values()})
    routes = sorted({f"{a.have}->{a.want}" for a in axes.values()})
    one = graphs[first]
    declared = {"platform": pname, "st_graph": one.method,
                "radius_cap_px": one.radius_cap_px,
                # sets: a per-sample cohort has no single value, and naming one
                # would name an arbitrary section's.
                "scale_to_microns": scales[0] if len(scales) == 1 else scales,
                "scale_is_per_sample": len(scales) > 1,
                "gene_axis_route": routes[0] if len(routes) == 1 else routes,
                "gene_axis_is_per_sample": len(routes) > 1,
                **axes[first].params(),
                "n_shards": len(triples), "skipped": skipped}

    if not state.get("compute"):
        return {"report": {"status": "ready", **declared,
                           "encoder": encoder,
                           "shards": [t[0] for t in triples],
                           "out_dir": str(out), "answers": answered,
                           "plan": state.get("plan", {}),
                           "note": "gates cleared and the shard list is resolved. pass "
                                   "compute=True to dispatch."}}

    shards = st_shards(triples, out, graphs, axes, encoder)
    return dispatch(state, shards, out, st_shard_body, fn_name="encode_st",
                    cohort_run=lambda pending: encode_st_cohort(
                        [(s.id, s.params["counts"], s.params["coords"]) for s in pending],
                        out, graphs, axes, encoder=encoder),
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
