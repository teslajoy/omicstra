"""the molecular assay's two pathway chains, and the aggregate they share.

LEVEL 2. a fixed order, transcribed from what produced the published artifacts,
not designed here. `design/analysis_pipeline_steps.md` §5 documents the variable
-gene chain; `scripts/niche_aucell.py` is the other one, and §5 does not mention
it - so a port of "§5" alone would have silently dropped a chain still in use.

    NICHE_AGGREGATE_STEPS   MT filter · k from the declaration · assemble the
                            niche. the only part the two genuinely share

    VARIABLE_GENE_STEPS     per-spot log1p(CPM 1e6) -> MAD/mean per gene ACROSS
                            the spots -> top percentile of the non-zero -> a
                            gene SET per niche. that set is gpath2vec's input;
                            enrichment, graph and embedding are the encoder's

    AUCELL_STEPS            SUM the spots -> normalize_total(1e4) -> log1p ->
                            AUCell over a declared universe

the two normalisations are not interchangeable and the difference is the point.
the variable-gene chain normalises PER SPOT before aggregating, because MAD/mean
needs the spots as several observations of one niche. the AUCell chain sums the
spots first and normalises the result. same counts, different unit of
normalisation, different meaning - and `enrichment_method` on the encoder was
never the real fork, which sits here, upstream.

every threshold is DECLARED. `top_percentile`, the minimum foreground and
background sizes, the UMI floors - absent is a refusal with a reason, never a
default, because a number chosen on one cohort is a number chosen on one cohort.
"""
from __future__ import annotations

from typing import Any

from omicstra.protocols import Ctx, Step
from omicstra.records import DiagnosticRecord


class PathwayInputMissing(RuntimeError):
    """a declared input is absent, so the step cannot run and will not guess."""


def _rec(step_id: str, status: str, result: str, decision: str,
         observed: dict[str, Any], method: str = "", criterion: str = "",
         **kw) -> DiagnosticRecord:
    return DiagnosticRecord(step_id=step_id, status=status, method=method,
                            criterion=criterion, result=result, decision=decision,
                            observed=observed, **kw)


def _declared(ctx: Ctx, field: str):
    """a step's declared parameter, or None. never a default.

    `ctx["params"]` is already this step's own, narrowed to its `params_space`
    by `_link` - so a step cannot read a parameter it did not declare, and
    whatever else the cohort's file happened to carry never reaches it.

    the chain's numbers were chosen on one cohort looking at one tissue. a
    default here would carry that choice silently onto the next cohort, which
    is the one thing `EVALUATION.md` §8 says a threshold of this kind must not
    do.
    """
    v = (ctx.get("params") or {}).get(field)
    return v if v not in ("", None) else None


def _missing(step: str, field: str, why: str) -> DiagnosticRecord:
    return _rec(step, "not_run", f"{field} is not declared",
                f"declare {field} for this cohort - {why}",
                {"missing": field})


# ------------------------------------------------- shared · niche aggregate --
def mito_filter(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """drop mitochondrial genes BEFORE aggregating. standard spatial QC.

    before, not after: a niche's MT fraction is a property of its spots, and
    summing first then filtering changes the library size every later
    normalisation divides by.
    """
    pattern = _declared(ctx, "pattern")
    if pattern is None:
        return _missing("mito_filter", "pattern",
                        "the prefix that marks mitochondrial genes is an "
                        "annotation convention, not a constant"), {}
    genes = ctx.get("genes")
    if genes is None:
        raise PathwayInputMissing("mito_filter needs `genes` on the context")

    import re

    rx = re.compile(pattern)
    keep = [i for i, g in enumerate(genes) if not rx.match(str(g))]
    dropped = len(genes) - len(keep)
    return _rec("mito_filter", "pass",
                f"{dropped} of {len(genes)} genes dropped",
                "filtered before aggregation, so library size is computed on "
                "what remains",
                {"n_in": len(genes), "n_out": len(keep), "reason": "mitochondrial",
                 "pattern": pattern},
                method=f"regex {pattern} over the gene axis",
                criterion="mitochondrial genes excluded before any normalisation",
                ), {"gene_keep": keep}


def niche_assemble(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """the unit: a centre spot and its k neighbours.

    k comes from the cohort DECLARATION. "7 spots" is one platform's number -
    k=6 plus a centre - and MODALITY_TEMPLATE.md is explicit that a new cohort
    must not have to edit code to change it.
    """
    k = _declared(ctx, "k_neighbors")
    if k is None:
        return _missing("niche_assemble", "k_neighbors",
                        "the niche width is a property of the platform's "
                        "lattice, measured per cohort"), {}
    members = ctx.get("niche_members")
    if members is None:
        raise PathwayInputMissing("niche_assemble needs `niche_members`")

    width = int(k) + 1
    full = sum(1 for m in members.values() if len(m) == width)
    return _rec("niche_assemble", "pass",
                f"{full} of {len(members)} niches have the full {width} spots",
                "an incomplete niche is kept and counted, not dropped here - "
                "the qc step decides on declared thresholds",
                {"n_in": len(members), "n_out": len(members),
                 "k_neighbors": int(k), "spots_per_niche": width,
                 "n_full": full, "n_partial": len(members) - full},
                method="centre spot plus its k declared spatial neighbours",
                criterion="k read from the cohort, never assumed",
                ), {"spots_per_niche": width}


NICHE_AGGREGATE_STEPS: list[Step] = [
    Step(id="mito_filter", fn=mito_filter, produces=frozenset({"gene_keep"}),
         params_space=("pattern",)),
    Step(id="niche_assemble", fn=niche_assemble,
         produces=frozenset({"niche"}), requires=frozenset({"gene_keep"}),
         params_space=("k_neighbors",)),
]
