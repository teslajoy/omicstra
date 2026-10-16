"""the molecular assay's two pathway chains, and the aggregate they share.

LEVEL 2. a fixed order, transcribed from what produced the published artifacts,
not designed here. `design/analysis_pipeline_steps.md` §5 documents the variable
-gene chain; `scripts/niche_aucell.py` is the other one, and §5 does not mention
it - so a port of "§5" alone would have silently dropped a chain still in use.

    NICHE_MEMBERSHIP_STEPS  MT filter · k from the declaration · which spots
                            form a niche. the only part the two genuinely
                            share, and it does NOT aggregate - aggregation is
                            where they fork

    VARIABLE_GENE_STEPS     per-spot log1p(CPM 1e6) -> MAD/mean per gene ACROSS
                            the spots -> top percentile of the non-zero -> a
                            gene SET per niche. that set is gpath2vec's input;
                            enrichment, graph and embedding are the encoder's

    AUCELL_STEPS            SUM the spots into one profile -> normalize_total
                            (1e4) -> log1p -> AUCell over a declared universe

THE FORK IS AGGREGATION, one step earlier than normalisation and two earlier
than any `enrichment_method` flag. one chain collapses the spots of a niche into
a single profile before doing anything else; the other never collapses them at
all, because MAD/mean needs them as several observations of one niche.

and the AUCell chain takes NO variable-gene filter. that is not a preference -
AUCell scores a pathway as an AUC over a ranking of ALL genes, so the gene
universe IS the denominator. restricting the input to a per-niche foreground
would change what every AUC is computed against, and since each niche picks its
own foreground the scores would stop being comparable between niches. a
mad-restricted variant exists as its own exploratory artifact and is kept
separate for exactly this reason.

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


def niche_membership(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """WHICH spots form a niche. it does not aggregate them.

    the two chains diverge at aggregation, so this stops at membership: one
    chain sums these spots into a single profile, the other keeps them apart as
    several observations of one niche. collapsing here would hand both chains
    the same row and silently make one of them impossible.

    k comes from the cohort DECLARATION. "7 spots" is one platform's number -
    k=6 plus a centre - and MODALITY_TEMPLATE.md is explicit that a new cohort
    must not have to edit code to change it.
    """
    k = _declared(ctx, "k_neighbors")
    if k is None:
        return _missing("niche_membership", "k_neighbors",
                        "the niche width is a property of the platform's "
                        "lattice, measured per cohort"), {}
    members = ctx.get("niche_members")
    if members is None:
        raise PathwayInputMissing("niche_membership needs `niche_members`")

    width = int(k) + 1
    full = sum(1 for m in members.values() if len(m) == width)
    return _rec("niche_membership", "pass",
                f"{full} of {len(members)} niches have the full {width} spots",
                "an incomplete niche is kept and counted, not dropped here - "
                "the qc step decides on declared thresholds",
                {"n_in": len(members), "n_out": len(members),
                 "k_neighbors": int(k), "spots_per_niche": width,
                 "n_full": full, "n_partial": len(members) - full},
                method="centre spot plus its k declared spatial neighbours",
                criterion="k read from the cohort, never assumed",
                ), {"spots_per_niche": width}


NICHE_MEMBERSHIP_STEPS: list[Step] = [
    Step(id="mito_filter", fn=mito_filter, produces=frozenset({"gene_keep"}),
         params_space=("pattern",)),
    Step(id="niche_membership", fn=niche_membership,
         produces=frozenset({"niche"}), requires=frozenset({"gene_keep"}),
         params_space=("k_neighbors",)),
]


# --------------------------------------------- the variable-gene chain -------
# transcribed from the code that wrote `niche_madmean_genesets.json`, with the
# parameters read out of that notebook's constants and checked against the
# artifact: 3,251 sampled niches, median 55 genes, minimum exactly 20, none
# below - which is what makes `min_foreground` a FLOOR ON THE COUNT rather than
# only a skip threshold.
#
# three details a reading of the prose would have got wrong:
#   the MAD is taken about the MEDIAN and divided by the MEAN. not MAD/median,
#   and no 1.4826 consistency factor - raw deviation over central tendency
#   the background is `mean > 0` on the NORMALISED profile, not on raw counts
#   the percentile runs over the NON-ZERO ratios only. a zero-variance gene is
#   locally constant, and ranking it adds tie-break noise on a 7-spot niche


def niche_qc(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """drop niches that are mostly white space, on declared floors.

    two gates and they ask different things: total depth over the whole niche,
    and how many of its spots carry signal at all. a niche can clear the first
    on one bright spot, which is not a niche.
    """
    need = {k: _declared(ctx, k) for k in
            ("min_niche_umi", "min_spot_umi", "min_detected_spots")}
    for k, v in need.items():
        if v is None:
            return _missing("niche_qc", k, "a depth floor is a property of the "
                                           "platform's capture efficiency"), {}
    niches = ctx.get("niches")
    if niches is None:
        raise PathwayInputMissing("niche_qc needs `niches` as (id, counts) pairs")

    kept, why = [], {"total_umi": 0, "detected_spots": 0}
    for nid, c in niches:
        if int(c.sum()) < need["min_niche_umi"]:
            why["total_umi"] += 1
            continue
        if int((c.sum(axis=1) >= need["min_spot_umi"]).sum()) < need["min_detected_spots"]:
            why["detected_spots"] += 1
            continue
        kept.append((nid, c))
    return _rec("niche_qc", "pass", f"{len(kept)} of {len(niches)} niches kept",
                "a dropped niche is counted by cause - an unexplained drop is "
                "an error, not a filter",
                {"n_in": len(niches), "n_out": len(kept), "reason": why, **need},
                method="total-depth and detected-spot floors over the raw counts",
                criterion="both floors cleared",
                ), {"qc_niches": kept}


def normalise(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """log1p(counts / library * target_sum), PER SPOT.

    per spot, before any aggregation, because the next step needs the spots as
    several observations of one niche. the other chain sums them first and
    normalises the result - same counts, different unit of normalisation, and
    the two are not interchangeable.
    """
    target = _declared(ctx, "target_sum")
    if target is None:
        return _missing("normalise", "target_sum",
                        "the library-size target is a declared convention"), {}
    import numpy as np

    out = []
    for nid, c in ctx["qc_niches"]:
        lib = c.sum(axis=1, keepdims=True) + 1e-9
        out.append((nid, np.log1p(c / lib * float(target)).astype("float32")))
    return _rec("normalise", "pass", f"{len(out)} niches at target {target:g}",
                "per spot, so the spots stay several observations of one niche",
                {"n_in": len(out), "n_out": len(out), "target_sum": float(target)},
                method=f"log1p(counts / library * {target:g}) per spot",
                criterion="normalised before aggregation, never after",
                ), {"normalised": out}


def variability(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """per gene, across the spots of its niche: MAD about the median over the mean.

    a robust coefficient of variation. the MAD is taken about the MEDIAN and
    divided by the MEAN - not MAD/median - and with no 1.4826 consistency
    factor, because nothing here assumes normality.
    """
    import numpy as np

    out: list = []
    for nid, z in ctx["normalised"]:
        mean = z.mean(axis=0)
        mad = np.median(np.abs(z - np.median(z, axis=0)[None, :]), axis=0)
        out.append((nid, mean, mad / (mean + 1e-9)))
    return _rec("variability", "pass", f"{len(out)} niches scored",
                "a robust CV per gene, computed within the niche",
                {"n_in": len(out), "n_out": len(out)},
                method="MAD about the median, divided by the mean, scale 1.0",
                criterion="no normality assumed, so no consistency factor",
                ), {"variability": out}


def background(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """locally detected genes: the denominator a selection test is judged against.

    `mean > 0` on the NORMALISED profile. this is the background a Fisher test
    uses, and it is per niche on purpose - a gene undetected in this niche is
    not part of what this niche could have selected.
    """
    floor = _declared(ctx, "min_background")
    if floor is None:
        return _missing("background", "min_background",
                        "the smallest background a selection test may be "
                        "judged against is a declared choice"), {}
    kept, dropped = [], 0
    for nid, mean, ratio in ctx["variability"]:
        bg = mean > 0
        if int(bg.sum()) < floor:
            dropped += 1
            continue
        kept.append((nid, bg, ratio))
    return _rec("background", "pass", f"{len(kept)} niches kept, {dropped} too sparse",
                "a niche whose background is below the floor cannot support a "
                "selection test, so it is dropped rather than tested",
                {"n_in": len(kept) + dropped, "n_out": len(kept),
                 "reason": {"background_below_floor": dropped},
                 "min_background": floor},
                method="locally detected = normalised mean > 0",
                criterion=f"at least {floor} detected genes",
                ), {"background": kept}


def foreground(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """the top fraction of variable genes, ranked within the niche.

    the percentile runs over the NON-ZERO ratios only. a zero-variance gene is
    locally constant and uninformative, and ranking it adds arbitrary tie-break
    noise on a niche with few spots.

    `min_foreground` is a FLOOR ON THE COUNT, not only a skip threshold -
    measured against the published artifact, whose smallest set is exactly it.
    """
    frac = _declared(ctx, "top_fraction")
    floor = _declared(ctx, "min_foreground")
    if frac is None or floor is None:
        return _missing("foreground", "top_fraction" if frac is None else "min_foreground",
                        "how much of a niche counts as variable was chosen by "
                        "looking at one tissue, so it does not travel"), {}
    import numpy as np

    kept, dropped, cutoffs = [], 0, []
    for nid, bg, ratio in ctx["background"]:
        nz = np.where(bg & (ratio > 0))[0]
        if nz.size < floor:
            dropped += 1
            continue
        n_keep = min(max(int(floor), int(np.floor(float(frac) * nz.size))), nz.size)
        top = nz[np.argpartition(-ratio[nz], n_keep - 1)[:n_keep]]
        cutoffs.append(float(ratio[top].min()))
        kept.append((nid, top))
    return _rec("foreground", "pass",
                f"{len(kept)} niches selected, {dropped} with too few variable genes",
                "each niche gets its own cutoff - the fraction is declared, the "
                "threshold it implies is measured per niche and recorded",
                {"n_in": len(kept) + dropped, "n_out": len(kept),
                 "reason": {"nonzero_below_floor": dropped},
                 "top_fraction": float(frac), "min_foreground": int(floor),
                 "cutoff_median": float(np.median(cutoffs)) if cutoffs else None,
                 "size_median": int(np.median([t.size for _, t in kept])) if kept else None},
                method="top fraction by robust CV, over the non-zero subset",
                criterion="percentile on non-zero only; count floored at min_foreground",
                ), {"foreground": kept}


def gene_sets(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """niche -> the symbols of its variable genes. gpath2vec's input.

    the seam. enrichment, the graph and the embedding belong to that package
    and are not re-implemented here; what this chain owes it is the sets.
    """
    genes = ctx.get("gene_symbols")
    if genes is None:
        raise PathwayInputMissing("gene_sets needs `gene_symbols` on the gene axis")
    import numpy as np

    g = np.asarray(genes)
    sets = {nid: g[idx].tolist() for nid, idx in ctx["foreground"]}
    sizes = [len(v) for v in sets.values()]
    return _rec("gene_sets", "pass", f"{len(sets)} niches with a gene set",
                "handed to the pathway encoder as its input; enrichment, graph "
                "and embedding are the encoder's and are not duplicated here",
                {"n_in": len(ctx["foreground"]), "n_out": len(sets),
                 "size_median": int(np.median(sizes)) if sizes else None,
                 "size_min": min(sizes) if sizes else None},
                method="gene symbols on the declared axis",
                criterion="one set per surviving niche",
                ), {"gene_sets": sets}


VARIABLE_GENE_STEPS: list[Step] = [
    Step(id="niche_qc", fn=niche_qc, produces=frozenset({"qc_niches"}),
         params_space=("min_niche_umi", "min_spot_umi", "min_detected_spots")),
    Step(id="normalise", fn=normalise, produces=frozenset({"normalised"}),
         requires=frozenset({"qc_niches"}), params_space=("target_sum",)),
    Step(id="variability", fn=variability, produces=frozenset({"variability"}),
         requires=frozenset({"normalised"})),
    Step(id="background", fn=background, produces=frozenset({"background"}),
         requires=frozenset({"variability"}), params_space=("min_background",)),
    Step(id="foreground", fn=foreground, produces=frozenset({"foreground"}),
         requires=frozenset({"background"}),
         params_space=("top_fraction", "min_foreground")),
    Step(id="gene_sets", fn=gene_sets, produces=frozenset({"gene_sets"}),
         requires=frozenset({"foreground"})),
]


# ------------------------------------------------------ the AUCell chain -----
# transcribed from `scripts/niche_aucell.py`. the notebook's prose says "AUCell
# on the variable-gene-filtered niche profile"; the script does not filter, and
# the script is right - see `aucell` below for why that is a method rule rather
# than a preference.


def niche_sum(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """collapse a niche's spots into ONE profile, by summing raw counts.

    this is where the two chains part. the variable-gene chain never collapses,
    because its statistic needs the spots as several observations; this one
    collapses first and normalises the result. the same counts, a different
    unit of normalisation, and not interchangeable.
    """
    import numpy as np

    rows = [(nid, np.asarray(c).sum(axis=0)) for nid, c in ctx["qc_niches"]]
    return _rec("niche_sum", "pass", f"{len(rows)} niches summed to one profile",
                "collapsed BEFORE normalising, which is what makes this chain "
                "different from the variable-gene one",
                {"n_in": len(rows), "n_out": len(rows)},
                method="sum raw counts over the spots of the niche",
                criterion="one profile per niche",
                ), {"summed": rows}


def library_normalise(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """scale each niche to a declared library size, then log1p.

    the target differs from the other chain's and that is not an oversight:
    this one normalises a SUMMED profile, the other normalises each spot.
    """
    target = _declared(ctx, "target_sum")
    if target is None:
        return _missing("library_normalise", "target_sum",
                        "the library-size target is a declared convention"), {}
    import numpy as np

    out = []
    for nid, v in ctx["summed"]:
        v = np.asarray(v, dtype="float64")
        out.append((nid, np.log1p(v * (float(target) / max(v.sum(), 1.0)))))
    return _rec("library_normalise", "pass",
                f"{len(out)} niches at target {target:g}",
                "normalised AFTER summing, on the niche's own library",
                {"n_in": len(out), "n_out": len(out), "target_sum": float(target)},
                method=f"counts * {target:g} / library, then log1p",
                criterion="one library size per niche",
                ), {"profiles": out}


def pathway_net(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """the gene sets to score, size-filtered on declared bounds.

    a set too small is noise and one too large is the transcriptome; both
    bounds are declared, because what counts as either depends on the panel.
    """
    lo, hi = _declared(ctx, "min_size"), _declared(ctx, "max_size")
    if lo is None or hi is None:
        return _missing("pathway_net", "min_size" if lo is None else "max_size",
                        "a usable gene-set size depends on the panel"), {}
    sets = ctx.get("pathway_sets")
    if sets is None:
        raise PathwayInputMissing("pathway_net needs `pathway_sets`")

    kept = {k: v for k, v in sets.items() if lo <= len(v) <= hi}
    return _rec("pathway_net", "pass",
                f"{len(kept)} of {len(sets)} pathways within [{lo}, {hi}]",
                "size-filtered before scoring, so a set nobody could score "
                "never reaches the ranking",
                {"n_in": len(sets), "n_out": len(kept),
                 "reason": {"outside_size_bounds": len(sets) - len(kept)},
                 "min_size": lo, "max_size": hi},
                method="declared size bounds over the pathway universe",
                criterion=f"{lo} <= |set| <= {hi}",
                ), {"net": kept}


def aucell(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """rank-based enrichment per niche, over the FULL gene universe.

    the universe is the denominator. AUCell scores a set as an AUC over the
    ranking of every measured gene in the unit, so restricting the input to a
    per-niche foreground would score each niche against a different denominator
    and the values would stop being comparable BETWEEN niches - which is the one
    thing they are for. a variable-gene-restricted variant exists as its own
    exploratory artifact, kept separate for exactly this reason.

    that is a general rule and `EVALUATION.md` carries it: a rank-based score
    wants the full universe, a selection test wants the filter, and the two
    cannot share a preprocessing path.
    """
    import numpy as np

    genes = ctx.get("gene_symbols")
    if genes is None:
        raise PathwayInputMissing("aucell needs `gene_symbols`")
    index = {g: i for i, g in enumerate(genes)}
    net = {k: [index[g] for g in v if g in index] for k, v in ctx["net"].items()}
    net = {k: v for k, v in net.items() if v}

    names, scores = sorted(net), []
    n = len(genes)
    for _, profile in ctx["profiles"]:
        # rank over EVERY gene, descending. ties take their average rank, so a
        # zero-inflated profile does not hand an arbitrary order an advantage.
        order = np.argsort(np.argsort(-np.asarray(profile), kind="stable"))
        rank = np.empty(n, dtype="float64")
        rank[np.argsort(order)] = np.arange(n, dtype="float64")
        scores.append([float(1.0 - (rank[net[k]].mean() / max(n - 1, 1)))
                       for k in names])
    return _rec("aucell", "pass",
                f"{len(scores)} niches x {len(names)} pathways",
                "every niche ranked over the same universe, so the scores are "
                "comparable between niches",
                {"n_in": len(ctx["profiles"]), "n_out": len(scores),
                 "n_pathways": len(names), "n_genes_ranked": n},
                method="AUC over the full-universe ranking, per gene set",
                criterion="no per-unit feature filter - the universe is the denominator",
                ), {"aucell": (names, scores,
                               [nid for nid, _ in ctx["profiles"]])}


def zscore_within_group(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """z-score each pathway WITHIN a declared group, usually the section.

    a section is a batch: its own staining, depth and capture efficiency. a
    score compared across sections without this compares batches. the raw
    matrix is kept too, because the z-score destroys absolute level and some
    questions need it.
    """
    group_by = _declared(ctx, "group_by")
    if group_by is None:
        return _missing("zscore_within_group", "group_by",
                        "which variable is the batch is a cohort's statement, "
                        "not something to infer from a name"), {}
    import numpy as np

    names, scores, nids = ctx["aucell"]
    groups = ctx.get("niche_groups") or {}
    arr = np.asarray(scores, dtype="float64")
    out = np.zeros_like(arr)
    seen: dict[str, list[int]] = {}
    for i, nid in enumerate(nids):
        seen.setdefault(str(groups.get(nid, "")), []).append(i)
    for rows in seen.values():
        block = arr[rows]
        out[rows] = (block - block.mean(axis=0)) / (block.std(axis=0, ddof=1) + 1e-9)
    return _rec("zscore_within_group", "pass",
                f"{len(seen)} {group_by} group(s) z-scored",
                "the raw scores are kept beside the z-scored ones - the z-score "
                "removes absolute level, and some questions need it",
                {"n_in": len(nids), "n_out": len(nids),
                 "group_by": group_by, "n_groups": len(seen)},
                method=f"(x - mean) / sd within each {group_by}",
                criterion="scored within the batch, never across it",
                ), {"aucell_z": (names, out.tolist(), nids)}


AUCELL_STEPS: list[Step] = [
    Step(id="niche_sum", fn=niche_sum, produces=frozenset({"summed"}),
         requires=frozenset({"qc_niches"})),
    Step(id="library_normalise", fn=library_normalise,
         produces=frozenset({"profiles"}), requires=frozenset({"summed"}),
         params_space=("target_sum",)),
    Step(id="pathway_net", fn=pathway_net, produces=frozenset({"net"}),
         params_space=("min_size", "max_size")),
    Step(id="aucell", fn=aucell, produces=frozenset({"aucell"}),
         requires=frozenset({"profiles", "net"})),
    Step(id="zscore_within_group", fn=zscore_within_group,
         produces=frozenset({"aucell_z"}), requires=frozenset({"aucell"}),
         params_space=("group_by",)),
]
