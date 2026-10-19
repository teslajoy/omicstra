"""the niche join - "make one row per unit of analysis", and account for every
unit that does not survive.

a PORT. the body is `scripts/build_niche_join.py`, the code that produced the
259 parquets every published run reads. this file gives it declared semantics, a
funnel and two guards. it changes no arithmetic.

what the join actually decides, and what it does not
----------------------------------------------------
it joins EMBEDDINGS, not counts. it never sees a gene, so the gene-universe
question - the three-fold spread from 10,571 to 33,047 genes per sample - lives
upstream in the pathway build and is not settled here. calling this step's
choices `gene_handling` would name the wrong thing in the manifest.

what it does decide is which UNITS survive, and there are four such decisions,
plus a fifth - `neighbour_lists` - that decides which spots a surviving niche
pools over on each side. each is ported as built and declared, for the reason
every other as-built declaration in this project exists: the ten-arm
comparison is fair only while every arm saw the same rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omicstra.protocols import Ctx, Step
from omicstra.records import DiagnosticRecord


class ConfounderBiasedFunnel(RuntimeError):
    """a funnel stage eliminated a whole group of the confounder axis.

    a drop that removes a patient entirely does not thin the cohort, it changes
    which cohort was studied - and every downstream number is then computed on a
    different population than the one described. that is an error; a merely
    UNEVEN drop is a caution, because it is common, often unavoidable, and
    reportable rather than fatal.
    """


class CircularSupervision(RuntimeError):
    """a signal is about to be both an input feature and the target it is scored
    against.

    this is guard G6, and it is a hard error rather than a caution because the
    result it produces looks excellent. `mc_weights` is the supervision for two
    arms of the grid and a 14-d vector sitting in the same table as the inputs;
    one careless column list turns a retrieval score into a measurement of
    whether a vector can find itself.
    """


# the five as-built decisions. each names the alternative it forecloses, so the
# road not taken is recorded rather than rediscovered.
SEMANTICS = {
    "spot_set": {
        "as_built": "intersection(virchow2_meta, novae_parquet)",
        "why": "a spot needs both a morphology vector and an expression vector to be a "
               "cross-modal unit at all. a union would create rows with one side missing.",
        "declared_alternative": None,
    },
    "neighbour_padding": {
        "as_built": "repeat_self",
        "why": "a niche whose neighbour is absent from the virchow2 meta is padded by "
               "repeating the centre spot, so the 7-token stack stays fixed-size. "
               "dropping the niche instead would remove boundary tissue, which is real.",
        "declared_alternative": {"value": "drop_incomplete_niche", "run": False,
                                 "question": "does excluding boundary niches change H1?"},
    },
    "neighbour_lists": {
        "as_built": "computed_per_modality",
        "why": "the H&E niche is pooled at encode time over a KDTree on the image-side "
               "spot set, over whichever neighbours exist. the ST niche and the 7-token "
               "stack are pooled over the ST encoder output's neighbor_spot_ids, "
               "filtered to spots with an H&E vector and padded with the centre. the "
               "two lists coincide when both modalities keep the same spots - on the "
               "seed cohort all 260 subarrays with both do, so the published join is "
               "unaffected - and diverge wherever one modality drops a spot the other "
               "keeps: a QC filter, a failed tile, an encoder floor.",
        "declared_alternative": {
            "value": "one_neighbour_list_for_both", "run": False,
            "question": "on a cohort whose modalities keep different spots, does pooling "
                        "both sides over the same seven spots change H1?"},
    },
    "pathway_coverage": {
        "as_built": "drop_niche_on_nan_gpath2vec",
        "why": "a niche with no significantly enriched pathway has no gpath2vec vector. "
               "align.py would drop these at training time anyway; doing it here gives "
               "smaller parquets and one intersection semantic instead of two.",
        "declared_alternative": {
            "value": "union_with_zero_fill", "run": False,
            "question": "zero-filling an absent pathway vector and measuring one are "
                        "different statements. does keeping those 67,131 niches, marked, "
                        "change H2 or H3?"},
    },
    "novae_normalisation": {
        "as_built": "zscore_per_subarray_after_intersection",
        "why": "the standard deviation reflects the niches that actually feed alignment. "
               "normalising before the filter would scale by rows that were then dropped.",
        "declared_alternative": {"value": "zscore_before_intersection", "run": False,
                                 "question": "does the filter's effect on the scale matter?"},
    },
}

# columns that are SUPERVISION, never input. the join puts them in the same table
# as the features because a training script needs both, which is exactly why the
# separation has to be asserted rather than remembered.
SUPERVISION_ONLY = frozenset({"mc_weights_niche", "mc_megacluster", "archetype",
                              "compartment", "tls"})

FEATURE_COLUMNS = frozenset({"virchow2_niche", "virchow2_cell_tokens",
                             "novae_niche", "gpath2vec_niche"})


@dataclass
class FunnelStage:
    """one row of the funnel: how many units, and why any were lost.

    `reason` is required. a stage that drops units without one is the defect the
    funnel exists to make impossible - the EDA contract has carried
    `itemised_funnel` as an absent advisory since April for exactly this.
    """
    name: str
    n_in: int
    n_out: int
    reason: str = ""

    @property
    def dropped(self) -> int:
        return self.n_in - self.n_out

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.name, "n_in": self.n_in, "n_out": self.n_out,
                "dropped": self.dropped, "reason": self.reason}


@dataclass
class Funnel:
    """every unit accounted for, from enumeration to the final table."""
    stages: list[FunnelStage] = field(default_factory=list)

    def add(self, name: str, n_in: int, n_out: int, reason: str = "") -> None:
        if n_out < n_in and not reason:
            raise ValueError(
                f"funnel stage {name!r} dropped {n_in - n_out} unit(s) with no reason. "
                "an unexplained drop is what the funnel exists to prevent.")
        self.stages.append(FunnelStage(name, n_in, n_out, reason))

    def as_dicts(self) -> list[dict[str, Any]]:
        return [s.as_dict() for s in self.stages]

    def record(self, step_id: str = "niche_join") -> DiagnosticRecord:
        first, last = self.stages[0], self.stages[-1]
        # a FRACTION, formatted as a percent at the point of display. the literal
        # 100.0 was the one number in this module, and a percent conversion sitting
        # next to real thresholds is indistinguishable from one to a reader or a lint.
        kept = last.n_out / first.n_in if first.n_in else 0.0
        return DiagnosticRecord(
            step_id=step_id, status="pass",
            method="itemised funnel - unit count at each stage with a reason for every drop",
            scope=f"{len(self.stages)} stage(s)",
            observed={"stages": self.as_dicts()},
            criterion="every dropped unit is attributable to a named stage",
            result=f"{first.n_in:,} -> {last.n_out:,} ({kept:.1%} retained)",
            decision="; ".join(f"{s.name}: -{s.dropped:,} ({s.reason})"
                               for s in self.stages if s.dropped))


def drop_distribution(groups: dict[str, tuple[int, int]], axis: str = "patient_id") -> dict:
    """is a stage's drop even across the confounder axis, or concentrated?

    `groups` is {group: (n_dropped, n_before)}.

    "even" here means: whether a unit is dropped does not depend on which group
    it came from. it does NOT mean every group loses the same fraction - chance
    alone spreads those - so the test is whether the observed spread exceeds what
    chance predicts, not whether it is zero.

    a 24% drop is harmless only if it falls evenly. if it concentrates in a few
    patients, the surviving table is a biased sample and every metric computed on
    it carries that bias silently - the same reasoning as NMI(label, subject),
    applied to the funnel instead of to a label.

    the test is against what UNIFORM dropping would actually predict rather than
    an eyeballed threshold: under a single pooled rate each group is
    Binomial(n_g, p), so the ratio of observed chi-square to its degrees of
    freedom is 1 when the drop is even and grows with concentration. reporting
    the ratio rather than only a p-value matters because at this many units a
    p-value is significant for a spread far too small to care about.
    """
    import numpy as np

    d = np.array([v[0] for v in groups.values()], float)
    n = np.array([v[1] for v in groups.values()], float)
    if n.sum() == 0 or len(d) < 2:
        return {"axis": axis, "n_groups": len(d), "testable": False,
                "note": "too few groups to say anything about evenness"}

    p_pooled = float(d.sum() / n.sum())
    rates = np.divide(d, n, out=np.zeros_like(d), where=n > 0)
    eliminated = sorted(g for g, r in zip(groups, rates) if r >= 1.0)

    denom = n * p_pooled * (1 - p_pooled)
    chi2 = float((((d - n * p_pooled) ** 2) / np.where(denom > 0, denom, np.inf)).sum())
    dof = len(d) - 1
    phi = chi2 / dof if dof else 0.0

    return {"axis": axis, "n_groups": len(d), "testable": True,
            "pooled_rate": round(p_pooled, 4),
            "min_rate": round(float(rates.min()), 4),
            "max_rate": round(float(rates.max()), 4),
            "sd_rate": round(float(rates.std()), 4),
            "overdispersion": round(phi, 1),
            # NOT "uniform". this says the spread is not detectably wider than
            # chance AT THIS BAR, which is absence of evidence - a small cohort
            # can fail to show concentration that is really there. naming the
            # field `uniform` claimed the stronger thing.
            "concentration_detected": bool(phi > OVERDISPERSION_CAUTION),
            "eliminated_groups": eliminated,
            "min_retained_units": int((n - d).min()),
            "note": ("overdispersion is observed spread / spread expected if the drop "
                     "were uniform. 1 is even; larger is concentrated.")}


# a drop twice as variable as chance is worth reporting. it is a CAUTION bar and
# not a failure bar: an uneven drop is common and often unavoidable, and the
# thing that must never pass silently is an eliminated group, which is separate.
OVERDISPERSION_CAUTION = 2.0


def assert_funnel_not_confounded(dist: dict) -> list[str]:
    """raise on an eliminated group, return cautions for an uneven one."""
    if dist.get("eliminated_groups"):
        raise ConfounderBiasedFunnel(
            f"{len(dist['eliminated_groups'])} group(s) on {dist['axis']} lost every "
            f"unit: {dist['eliminated_groups'][:5]}. the surviving table describes a "
            "different cohort than the one enumerated.")
    if not dist.get("testable") or not dist.get("concentration_detected"):
        return []
    return [(f"the drop is uneven across {dist['axis']}: overdispersion "
             f"{dist['overdispersion']}x against a uniform drop, rates "
             f"{dist['min_rate']:.3f}-{dist['max_rate']:.3f}. every downstream metric is "
             f"computed on a sample whose density varies with {dist['axis']}, which "
             "belongs in the writeup rather than in a footnote.")]


def assert_no_circular_supervision(feature_columns, supervision: str) -> None:
    """guard G6. the supervision signal must not appear in the input vector.

    a static assertion on column names, deliberately: by the time it is a tensor
    the question is unanswerable, and the failure mode is a number that looks
    like a triumph.
    """
    cols = set(feature_columns)
    if supervision in cols:
        raise CircularSupervision(
            f"{supervision!r} is both an input feature and the supervision signal. "
            "the score this produces measures whether a vector can find itself.")
    if leaked := cols & SUPERVISION_ONLY:
        raise CircularSupervision(
            f"{sorted(leaked)} are supervision/evaluation columns and must not be input "
            f"features. declared inputs: {sorted(cols)}")


def semantics_record(step_id: str = "niche_join_semantics") -> DiagnosticRecord:
    """the five as-built decisions, as a record, so they reach the ledger."""
    alts = {k: v["declared_alternative"] for k, v in SEMANTICS.items()
            if v["declared_alternative"]}
    return DiagnosticRecord(
        step_id=step_id, status="pass",
        method="join semantics, ported as built",
        scope=f"{len(SEMANTICS)} decisions, {len(alts)} with a declared alternative",
        observed={k: v["as_built"] for k, v in SEMANTICS.items()},
        criterion="every unit-affecting choice is declared, and its alternative recorded",
        result=f"{len(alts)} alternative(s) declared and not run",
        decision="ported as built; changing any of these is a new arm, not a fix - "
                 "the ten-arm comparison is fair only while every arm saw the same rows")


# ============================================================ JOIN_STEPS ====
# the join as a chain. ported from `scripts/build_niche_join.py` against
# `script_map/build_niche_join.md`; every step below names the map rows it
# mirrors, and the map's bottom section lists what no step covers and why.
#
# the script is one cohort's chain and names its three encoders in code. this
# does not: an assay is whatever the cohort DECLARES, the anchor is declared,
# the neighbour source is declared, and the group a z-score is taken within is
# declared. nothing here knows what tissue it is looking at.
#
# the ctx this chain reads, all of it per ONE sample - a sample is a shard, and
# `dispatch/` already owns the loop over shards:
#
#   ctx["sample"]           the sample key being joined
#   ctx["assays"]           {assay: frame indexed by unit id}  - the vectors
#   ctx["neighbours"]       {unit id: [unit id, ...]}          - as declared
#   ctx["unit_labels"]      {label: frame indexed by unit id}
#   ctx["subject_of"]       {sample key: subject id}
#   ctx["subject_labels"]   {subject id: {label: value}}
#
# four corrections the map forced, each one a place a transcription would have
# carried a defect forward:
#
#   the neighbour filter tests ONE assay's index in the script (map row 220), so
#   a neighbour absent from another assay still entered the pool and the mean ran
#   over a smaller denominator with no counter. here `present` is the unit set
#   every pooled assay agrees on, and the realised denominator is recorded
#
#   the intersection records its n_out and never its n_in (map row 201). every
#   stage here states both ends
#
#   a subject with no label became `None` in two columns, uncounted (map row 1 of
#   section 5). absence is counted and marked, or the step refuses
#
#   the provenance column stored the PRE-filter neighbour list (map row 291), so
#   recomputing the pool from it did not reproduce the row. here it stores the
#   members actually pooled


class JoinInputMissing(RuntimeError):
    """a declared input is absent from the context, so no step will guess at it."""


def _jrec(step_id: str, status: str, result: str, decision: str,
          observed: dict[str, Any], method: str = "", criterion: str = "",
          **kw) -> DiagnosticRecord:
    return DiagnosticRecord(step_id=step_id, status=status, method=method,
                            criterion=criterion, result=result, decision=decision,
                            observed=observed, **kw)


def _jdeclared(ctx: Ctx, field: str):
    """this step's declared parameter, or None. never a default.

    `ctx["params"]` is already narrowed to the step's own `params_space` by
    `_link`, so a step cannot read a parameter it did not declare.
    """
    v = (ctx.get("params") or {}).get(field)
    return v if v not in ("", None) else None


def _jmissing(step: str, field: str, why: str) -> DiagnosticRecord:
    return _jrec(step, "not_run", f"{field} is not declared",
                 f"declare {field} for this cohort - {why}", {"missing": field})


def _need(ctx: Ctx, key: str, step: str):
    if key not in ctx:
        raise JoinInputMissing(f"{step} needs {key!r} on the context")
    return ctx[key]


# ----------------------------------------------------------- 1 · the units --
def unit_inventory(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """which units are units at all: the intersection over the declared assays.

    mirrors map row 201, and adds the end it was missing. the script records
    `n_niches_pre_intersection`, which IS the size of this intersection - but
    the size it intersected, one count per assay, appears nowhere. a funnel
    stage needs both ends or it cannot say what it removed.

    a required assay with no units for this sample makes the whole sample
    `not_applicable` rather than an empty table: there is nothing to join, which
    is a different statement from joining and getting nothing.
    """
    required = _jdeclared(ctx, "required_assays")
    if required is None:
        return _jmissing("unit_inventory", "required_assays",
                         "which assays a unit must appear in is the cohort's "
                         "definition of a unit, not this code's"), {}
    assays = _need(ctx, "assays", "unit_inventory")
    required = list(required)

    undeclared = [a for a in required if a not in assays]
    if undeclared:
        return _jrec("unit_inventory", "not_run",
                     f"declared assay(s) {undeclared} are not on the context",
                     "a declaration that names an absent assay is refused, not "
                     "silently narrowed to the assays that happen to be present",
                     {"required": required, "present": sorted(assays)}), {}

    per_assay = {a: list(assays[a].index) for a in required}
    sizes = {a: len(v) for a, v in per_assay.items()}
    empty = sorted(a for a, n in sizes.items() if n == 0)
    if empty:
        return _jrec("unit_inventory", "not_applicable",
                     f"assay(s) {empty} contribute no units to this sample",
                     "the sample is not joinable - this is absence of the "
                     "modality, not a join that produced nothing",
                     {"per_assay": sizes, "n_out": 0, "reason": f"no units in {empty}",
                      "sample": ctx.get("sample")}), {}

    first, *rest = required
    units = [u for u in per_assay[first]
             if all(u in set(per_assay[a]) for a in rest)]
    n_in = max(sizes.values())
    only_one = n_in - len(units)
    return _jrec("unit_inventory", "pass",
                 f"{len(units)} of {n_in} units appear in all {len(required)} assays",
                 "a unit needs a vector in every declared assay to be a "
                 "cross-modal unit at all; a union would make rows with a side "
                 "missing",
                 {"n_in": n_in, "n_out": len(units),
                  "reason": "present in fewer than all declared assays",
                  "per_assay": sizes, "n_in_one_assay_only": only_one,
                  "required_assays": required},
                 method="intersection over the declared assays, order preserved",
                 criterion="every assay's unit count is stated, not only the result",
                 ), {"units": units, "unit_counts": sizes}


# -------------------------------------------------------- 2 · the subject ---
def subject_resolution(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """the sample's subject, and the subject-level labels, with absence counted.

    mirrors map rows 128 and 179-181. in the script both lookups are `.get` ->
    `None`: a subject with no label, or a sample whose key is absent from the
    subject map, still emits every row with `None` in two columns and nothing
    counted anywhere. a subject-level label is the axis every split and every
    confounder check is taken on, so losing it silently is not a cosmetic gap.

    `on_absent` is declared because the right answer differs by cohort: `refuse`
    when the label is load-bearing for the split, `mark` when it is descriptive
    and a present/absent flag is enough.
    """
    labels = _jdeclared(ctx, "subject_labels")
    policy = _jdeclared(ctx, "on_absent")
    if labels is None:
        return _jmissing("subject_resolution", "subject_labels",
                         "which labels live at subject level rather than unit "
                         "level is a property of the cohort's tables"), {}
    if policy not in ("refuse", "mark"):
        return _jmissing("subject_resolution", "on_absent",
                         "declare `refuse` when a subject label is load-bearing "
                         "for the split, `mark` when it is descriptive"), {}

    sample = _need(ctx, "sample", "subject_resolution")
    subject_of = _need(ctx, "subject_of", "subject_resolution")
    subject_labels = ctx.get("subject_labels") or {}
    labels = list(labels)

    subject = subject_of.get(sample)
    if subject is None:
        rec = _jrec("subject_resolution",
                    "fail" if policy == "refuse" else "pass",
                    f"sample {sample!r} resolves to no subject",
                    ("refused - a unit with no subject cannot be placed on either "
                     "side of a by-subject split"
                     if policy == "refuse" else
                     "kept and marked absent; the split must exclude it explicitly"),
                    {"sample": sample, "subject": None,
                     "reason": "sample absent from the subject map",
                     "n_absent_labels": len(labels)})
        return rec, {"subject": None, "subject_values": {},
                     "subject_absent": list(labels)}

    have = subject_labels.get(subject) or {}
    values = {k: have[k] for k in labels if k in have and have[k] is not None}
    absent = [k for k in labels if k not in values]
    status = "fail" if (absent and policy == "refuse") else "pass"
    return _jrec("subject_resolution", status,
                 f"subject {subject!r}: {len(values)} of {len(labels)} "
                 f"declared label(s) present",
                 ("refused - a declared subject label is absent and this cohort "
                  "declares it load-bearing"
                  if status == "fail" else
                  "present labels attach to every unit; absent ones are marked, "
                  "never written as a bare null"),
                 {"sample": sample, "subject": subject,
                  "present": sorted(values), "absent": absent,
                  "reason": "subject label absent" if absent else "",
                  "n_in": len(labels), "n_out": len(values)},
                 method="sample -> subject -> declared subject-level labels",
                 criterion="an absent subject label is counted and marked, not nulled",
                 ), {"subject": subject, "subject_values": values,
                     "subject_absent": absent}


# --------------------------------------------------- 3 · the neighbourhood ---
def neighbourhood(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """which units each unit pools over, on a presence set every assay agrees on.

    mirrors map rows 218, 220 and 221-224, and corrects the index the filter
    runs against. the script filters the declared neighbour list against ONE
    assay's index, so a neighbour present there but absent from another pooled
    assay passed the filter, reached the pool as a missing row, and shrank that
    assay's denominator with no counter and no reason. `present` here is the
    intersection from step 1, which is the only set every pooled assay agrees
    on, so the filter and the pool cannot disagree.

    padding is declared rather than assumed. repeating the centre says "this
    niche is smaller than k+1", which is true; a zero vector would be a
    statement about the tissue that nothing measured.
    """
    k = _jdeclared(ctx, "k_neighbors")
    source = _jdeclared(ctx, "neighbour_source")
    padding = _jdeclared(ctx, "padding")
    if k is None:
        return _jmissing("neighbourhood", "k_neighbors",
                         "the niche width is a property of the platform's "
                         "lattice, measured per cohort"), {}
    if source is None:
        return _jmissing("neighbourhood", "neighbour_source",
                         "which assay's neighbour lists are authoritative is a "
                         "declaration - two assays can disagree"), {}
    if padding != "repeat_centre":
        return _jmissing("neighbourhood", "padding",
                         "`repeat_centre` is the only policy implemented. "
                         "dropping incomplete niches is declared in SEMANTICS "
                         "and not run, and a zero vector is not an option"), {}

    from omicstra.measures import niche_members

    units = _need(ctx, "units", "neighbourhood")
    declared = _need(ctx, "neighbours", "neighbourhood")
    k = int(k)
    present = set(units)

    members, padded, oversize = {}, 0, 0
    for u in units:
        raw = list(declared.get(u) or [])
        if len(raw) > k:
            oversize += 1
        kept = [n for n in raw if n in present]
        if len(kept) < k:
            padded += 1
        members[u] = niche_members(u, raw, k, present)

    width = k + 1
    return _jrec("neighbourhood", "pass",
                 f"{len(units) - padded} of {len(units)} units have {width} "
                 f"distinct members; {padded} padded with their own centre",
                 "every member is a unit that survived step 1, so the filter "
                 "and every pooled assay see the same set",
                 {"n_in": len(units), "n_out": len(units),
                  "reason": "", "k_neighbors": k, "members_per_unit": width,
                  "n_padded": padded, "n_declared_lists_longer_than_k": oversize,
                  "neighbour_source": source, "padding": padding},
                 method=f"declared neighbour list from {source}, filtered to the "
                        f"step-1 intersection, padded to {width} by repeating the centre",
                 criterion="the presence set is the one every pooled assay agrees on",
                 ), {"members": members}


# ------------------------------------------------------- 4 · the pooling -----
def pool_continuous(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """pool each declared assay over a unit's members, and record the denominator.

    mirrors map rows 231, 241-251 and the `niche_mean_pool` helper at 153-166.
    the mean is nan-safe, which is right - a member missing one value should
    contribute its others - but it means the denominator varies, and the script
    never recorded it. a vector pooled over four members and one pooled over
    seven are not the same measurement, and nothing downstream could tell them
    apart.

    `support` below is that denominator: per assay, how many of a unit's members
    actually had a row. step 3 makes it k+1 for every unit on a well-formed
    cohort, so a number below that is a finding rather than a routine fact.
    """
    pooled_assays = _jdeclared(ctx, "pooled_assays")
    data_columns = _jdeclared(ctx, "data_columns")
    if pooled_assays is None:
        return _jmissing("pool_continuous", "pooled_assays",
                         "which assays are pooled over a neighbourhood, and "
                         "which are already per-unit, is a cohort declaration"), {}
    if data_columns is None:
        return _jmissing("pool_continuous", "data_columns",
                         "which of an assay's columns carry the measurement. a "
                         "frame also holds coordinates, ids and provenance, and "
                         "selecting by dtype pulls those into the vector"), {}

    from omicstra.measures import pool_rows

    assays = _need(ctx, "assays", "pool_continuous")
    members = _need(ctx, "members", "pool_continuous")
    units = _need(ctx, "units", "pool_continuous")
    pooled_assays = list(pooled_assays)

    undeclared = [a for a in pooled_assays if a not in assays]
    if undeclared:
        return _jrec("pool_continuous", "not_run",
                     f"declared assay(s) {undeclared} are not on the context",
                     "refused rather than skipped - a pooled assay that is "
                     "silently absent is a column of zeros nobody ordered",
                     {"declared": pooled_assays, "present": sorted(assays)}), {}

    undeclared_cols = [a for a in pooled_assays if a not in data_columns]
    if undeclared_cols:
        return _jrec("pool_continuous", "not_run",
                     f"assay(s) {undeclared_cols} declare no data columns",
                     "refused - an assay whose measurement columns are not "
                     "declared would have them inferred, and the script that "
                     "produced this chain's artifacts had to name five "
                     "non-data columns by hand for exactly that reason",
                     {"pooled_assays": pooled_assays,
                      "declared_for": sorted(data_columns)}), {}

    pooled: dict[str, dict] = {a: {} for a in pooled_assays}
    support: dict[str, dict] = {a: {} for a in pooled_assays}
    for a in pooled_assays:
        frame = assays[a]
        cols = list(data_columns[a])
        if missing_cols := [c for c in cols if c not in frame.columns]:
            return _jrec("pool_continuous", "not_run",
                         f"assay {a!r} declares column(s) {missing_cols} it does "
                         "not have",
                         "refused - a declaration that does not match the data is "
                         "a defect in one of the two, and guessing which hides it",
                         {"assay": a, "declared": cols,
                          "present": list(frame.columns)}), {}
        idx = set(frame.index)
        for u in units:
            mem = members[u]
            pooled[a][u] = pool_rows(frame, mem, cols)
            support[a][u] = sum(1 for m in mem if m in idx)

    width = len(members[units[0]]) if units else 0
    short = {a: sum(1 for n in s.values() if n < width) for a, s in support.items()}
    underpooled = sorted(a for a, n in short.items() if n)
    return _jrec("pool_continuous", "pass" if not underpooled else "fail",
                 f"{len(units)} unit(s) pooled over {len(pooled_assays)} assay(s); "
                 + (f"assay(s) {underpooled} pooled some units over fewer than "
                    f"{width} members" if underpooled else
                    f"every unit pooled over all {width} members in every assay"),
                 ("refused - a pooled vector whose denominator differs between "
                  "units is not comparable between them, which is the defect the "
                  "script's single-index filter hid"
                  if underpooled else
                  "the denominator is uniform, so pooled vectors are comparable "
                  "across units"),
                 {"n_in": len(units), "n_out": len(units), "reason": "",
                  "members_per_unit": width, "pooled_assays": pooled_assays,
                  "n_units_under_full_support": short,
                  "data_columns": {a: list(data_columns[a]) for a in pooled_assays}},
                 method="nan-safe mean over a unit's members, per assay",
                 criterion="every unit pooled over the same number of members",
                 ), {"pooled": pooled, "support": support}


# ------------------------------------------------- 5 · the centre labels -----
def centre_labels(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """discrete labels, from the CENTRE unit only, with absence as its own column.

    mirrors map rows 260-265 and 267-276. taking a class label from the centre
    while pooling the vectors over k+1 members is deliberate - a discrete label
    does not mean anything averaged - but it means a row's label and its vectors
    describe different footprints, which belongs in the record rather than in a
    reader's head.

    absence is marked in a companion `{label}__present` column, never written as
    a sentinel value. the script used -1 and downstream code filters `< 0`; a
    sentinel inside the value space of a label is one renumbering away from
    being a class, and on a cohort whose classes are strings there is no
    negative number to reach for.
    """
    declared = _jdeclared(ctx, "centre_only_labels")
    if declared is None:
        return _jmissing("centre_labels", "centre_only_labels",
                         "which labels are discrete, and so may not be pooled, "
                         "is a property of the cohort's label tables"), {}

    tables = _need(ctx, "unit_labels", "centre_labels")
    units = _need(ctx, "units", "centre_labels")
    declared = list(declared)

    undeclared = [n for n in declared if n not in tables]
    if undeclared:
        return _jrec("centre_labels", "not_run",
                     f"declared label(s) {undeclared} have no table on the context",
                     "refused - a declared label with no table is a missing "
                     "input, not an absent value",
                     {"declared": declared, "present": sorted(tables)}), {}

    values: dict[str, dict] = {}
    presence: dict[str, dict] = {}
    for name in declared:
        frame = tables[name]
        col = next(iter(frame.columns), None)
        if col is None:
            return _jrec("centre_labels", "not_run",
                         f"label table {name!r} has no column to read",
                         "refused rather than read by position",
                         {"label": name}), {}
        idx = set(frame.index)
        values[name] = {u: (frame.at[u, col] if u in idx else None) for u in units}
        presence[name] = {u: (u in idx) for u in units}

    missing = {n: sum(1 for p in presence[n].values() if not p) for n in declared}
    return _jrec("centre_labels", "pass",
                 f"{len(declared)} label(s) read from the centre unit; absent "
                 f"counts {missing}",
                 "absence is a column, so a downstream filter selects on it "
                 "rather than on a magic value inside the label's own range",
                 {"n_in": len(units), "n_out": len(units), "reason": "",
                  "labels": declared, "n_absent_per_label": missing,
                  "footprint": "centre unit only; pooled vectors span the "
                               "whole neighbourhood"},
                 method="centre unit's value per declared discrete label",
                 criterion="no sentinel inside a label's value space",
                 ), {"label_values": values, "label_present": presence}


# ----------------------------------------------------------- 6 · assemble ----
def assemble(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """one row per unit, and a provenance column that reproduces the row.

    mirrors map rows 278-292. the script's `neighbor_spot_ids` stores the RAW
    declared list, before its own filter and before padding, so a reader who
    recomputes the pool from that column does not get the row back. here the
    column is the members that were actually pooled.
    """
    provenance = _jdeclared(ctx, "provenance")
    if provenance != "members_pooled":
        return _jmissing("assemble", "provenance",
                         "`members_pooled` is the only policy implemented - the "
                         "column must be the list the pooling actually used"), {}

    units = _need(ctx, "units", "assemble")
    members = _need(ctx, "members", "assemble")
    pooled = _need(ctx, "pooled", "assemble")
    values = ctx.get("label_values") or {}
    present = ctx.get("label_present") or {}
    subject_values = ctx.get("subject_values") or {}
    subject_absent = ctx.get("subject_absent") or []

    rows = []
    for u in units:
        row: dict[str, Any] = {"unit_id": u, "sample": ctx.get("sample"),
                               "subject": ctx.get("subject"),
                               "members_pooled": list(members[u])}
        for a, by_unit in pooled.items():
            row[a] = by_unit[u]
        for name, by_unit in values.items():
            row[name] = by_unit[u]
            row[f"{name}__present"] = present[name][u]
        for name, v in subject_values.items():
            row[name] = v
        for name in subject_absent:
            row[f"{name}__present"] = False
        rows.append(row)

    return _jrec("assemble", "pass",
                 f"{len(rows)} row(s), {len(rows[0]) if rows else 0} column(s)",
                 "the provenance column is the member list the pooling used, so "
                 "the row can be recomputed from the row",
                 {"n_in": len(units), "n_out": len(rows), "reason": "",
                  "columns": sorted(rows[0]) if rows else [],
                  "provenance": provenance},
                 method="fixed-width row per unit",
                 criterion="provenance reproduces the pooling",
                 ), {"rows": rows}


# ---------------------------------------------- 7 · the anchor's coverage ----
def anchor_coverage(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """drop units with no vector in the declared anchor assay, and say so.

    mirrors map rows 234-238 and 301-305 - the big drop, and the one the script
    already counted. two things are declared here that the script wrote in code:
    WHICH assay is the anchor, and what happens to a unit the anchor does not
    cover. the script hardcoded a 512-wide nan row at row 238 while taking the
    real width from the data at rows 390 and 461; a width written in one place
    and measured in another is a width that disagrees on the next cohort.

    `mark` keeps the unit with an explicit coverage column, which is the
    `union_with_zero_fill` alternative SEMANTICS declares and does not run -
    except that it marks rather than zero-fills, because a zero vector and an
    absent one are different statements.

    a sample left with no units REFUSES. an empty table is not a result: the
    next stage would read it, find nothing, and report a cohort rather than a
    gap.
    """
    anchor = _jdeclared(ctx, "anchor_assay")
    on_absent = _jdeclared(ctx, "on_absent")
    if anchor is None:
        return _jmissing("anchor_coverage", "anchor_assay",
                         "which assay a unit must be covered by is the cohort's "
                         "choice of anchor, and it changes which units exist"), {}
    if on_absent not in ("drop", "mark"):
        return _jmissing("anchor_coverage", "on_absent",
                         "declare `drop` to require anchor coverage, `mark` to "
                         "keep uncovered units with a coverage column"), {}

    assays = _need(ctx, "assays", "anchor_coverage")
    rows = _need(ctx, "rows", "anchor_coverage")
    if anchor not in assays:
        return _jrec("anchor_coverage", "not_run",
                     f"the declared anchor assay {anchor!r} is not on the context",
                     "refused - without the anchor there is no coverage question "
                     "to answer, and answering it as `everything is covered` "
                     "would pass silently",
                     {"anchor": anchor, "present": sorted(assays)}), {}

    covered = set(assays[anchor].index)
    kept = [r for r in rows if r["unit_id"] in covered]
    n_dropped = len(rows) - len(kept)

    if on_absent == "mark":
        for r in rows:
            r[f"{anchor}__covered"] = r["unit_id"] in covered
        kept = rows
        n_dropped = 0

    if not kept:
        return _jrec("anchor_coverage", "fail",
                     f"every one of {len(rows)} unit(s) lacks {anchor!r} coverage",
                     "refused - an empty table would be read downstream as a "
                     "sample that was measured and found empty",
                     {"n_in": len(rows), "n_out": 0, "anchor": anchor,
                      "reason": f"no unit covered by {anchor}",
                      "sample": ctx.get("sample")}), {}

    return _jrec("anchor_coverage", "pass",
                 f"{len(kept)} of {len(rows)} unit(s) retained",
                 (f"units without {anchor!r} coverage are dropped here rather "
                  "than at training time, so one intersection semantic holds "
                  "for every arm"
                  if on_absent == "drop" else
                  f"uncovered units are kept and marked in {anchor}__covered; "
                  "measuring an absence and zero-filling it are different claims"),
                 {"n_in": len(rows), "n_out": len(kept),
                  "reason": f"no vector in the anchor assay {anchor!r}"
                            if n_dropped else "",
                  "anchor": anchor, "on_absent": on_absent,
                  "n_dropped": n_dropped},
                 method=f"coverage by the declared anchor assay {anchor!r}",
                 criterion="the anchor is declared, and its width is never written down",
                 ), {"table": kept}


# -------------------------------------------------- 8 · the standardisation --
def standardise(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """z-score the declared assays within the declared group, AFTER the drop.

    mirrors map rows 307-315. the order is the method: the mean and the standard
    deviation are taken over the units that survived step 7, so step 7 changes
    every value this step emits. `requires={"table"}` makes that a BUILD-time
    assertion - a chain with this step before the drop raises
    `ProtocolOrderError` before any data is touched, rather than producing a
    table whose numbers are quietly different.

    `ddof` is declared and checked rather than accepted: `measures.zscore_per_group`
    computes the population deviation, which is what `numpy.std` does by default
    and so what the script did. a cohort that declares the sample deviation gets
    a refusal naming the measure, not a number computed by a different formula
    than the one declared.
    """
    assays = _jdeclared(ctx, "standardise_assays")
    group_by = _jdeclared(ctx, "group_by")
    eps = _jdeclared(ctx, "eps")
    ddof = _jdeclared(ctx, "ddof")
    if assays is None:
        return _jmissing("standardise", "standardise_assays",
                         "which assays are standardised, and which arrive on a "
                         "comparable scale already, is a cohort declaration"), {}
    if group_by is None:
        return _jmissing("standardise", "group_by",
                         "the group a z-score is taken WITHIN decides what it "
                         "removes - sample, subject or batch are different claims"), {}
    if eps is None:
        return _jmissing("standardise", "eps",
                         "the floor below which a deviation is treated as zero "
                         "depends on the assay's scale"), {}

    from omicstra.measures import zscore_per_group

    # the measure's formula, named rather than reimplemented. the funnel rule's
    # sibling: a number computed by one formula and declared as another is the
    # same defect as a number that disagrees with its artifact.
    implemented_ddof = 0
    if ddof is None:
        return _jmissing("standardise", "ddof",
                         "the divisor is part of the definition and the published "
                         "chain used the population form; declare it rather than "
                         "inheriting whichever one a library defaults to"), {}
    if int(ddof) != implemented_ddof:
        return _jrec("standardise", "not_run",
                     f"ddof={int(ddof)} declared; measures.zscore_per_group "
                     f"computes ddof={implemented_ddof}",
                     "refused - the alternative belongs in the measure, where a "
                     "notebook can check it, not in a step that quietly uses the "
                     "other formula",
                     {"declared_ddof": int(ddof),
                      "implemented_ddof": implemented_ddof}), {}

    table = _need(ctx, "table", "standardise")
    present = _need(ctx, "assays", "standardise")
    assays = list(assays)
    undeclared = [a for a in assays if a not in present]
    if undeclared:
        return _jrec("standardise", "not_run",
                     f"declared assay(s) {undeclared} are not on the context",
                     "refused rather than skipped", {"declared": assays}), {}

    import numpy as np

    groups = sorted({str(r.get(group_by)) for r in table})
    for a in assays:
        for g in groups:
            rows_g = [r for r in table if str(r.get(group_by)) == g]
            z = zscore_per_group(np.stack([r[a] for r in rows_g]), eps=float(eps))
            for r, zr in zip(rows_g, z):
                r[a] = zr

    return _jrec("standardise", "pass",
                 f"{len(assays)} assay(s) standardised within {len(groups)} "
                 f"{group_by} group(s) over {len(table)} retained unit(s)",
                 "taken after the coverage drop, so the deviation reflects the "
                 "units that feed the next stage rather than units already gone",
                 {"n_in": len(table), "n_out": len(table), "reason": "",
                  "standardise_assays": assays, "group_by": group_by,
                  "n_groups": len(groups), "eps": float(eps),
                  "ddof": implemented_ddof},
                 method=f"per-{group_by} z-score, population deviation, "
                        f"zero-variance columns left alone",
                 criterion="the standardisation runs after every unit-removing stage",
                 ), {"table": table}


# ------------------------------------------------------------ 9 · the funnel --
def join_funnel(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """every unit accounted for, assembled from what the steps recorded.

    mirrors map rows 318-331. it reads `n_in`, `n_out` and `reason` out of the
    earlier steps' own records rather than recounting, which makes the funnel a
    check on the chain instead of a second opinion about it: a step that dropped
    units without recording a reason fails here, in `Funnel.add`, and a step
    that recorded no counts at all is named.

    then the two guards that already existed: is the drop concentrated on the
    confounder axis, and is the supervision signal also an input feature.
    """
    axis = _jdeclared(ctx, "confounder_axis")
    features = _jdeclared(ctx, "feature_columns")
    supervision = _jdeclared(ctx, "supervision")
    if axis is None:
        return _jmissing("join_funnel", "confounder_axis",
                         "the axis a drop must not concentrate on is the axis "
                         "the split is taken by"), {}

    records = ctx.get("_records") or {}
    funnel = Funnel()
    uncounted = []
    for step_id, rec in records.items():
        obs = (rec or {}).get("observed") or {}
        if "n_in" not in obs or "n_out" not in obs:
            if (rec or {}).get("status") == "pass":
                uncounted.append(step_id)
            continue
        funnel.add(step_id, int(obs["n_in"]), int(obs["n_out"]),
                   reason=str(obs.get("reason") or ""))

    if uncounted:
        return _jrec("join_funnel", "fail",
                     f"step(s) {uncounted} passed without recording n_in/n_out",
                     "refused - a stage with no counts is the unexplained drop "
                     "the funnel exists to make impossible, whether or not it "
                     "dropped anything",
                     {"uncounted": uncounted, "n_stages": len(funnel.stages)}), {}
    if not funnel.stages:
        return _jrec("join_funnel", "not_run",
                     "no step recorded a unit count",
                     "a funnel over nothing is not a funnel",
                     {"steps_seen": sorted(records)}), {}

    # the confounder question cannot be answered HERE, and pretending otherwise
    # was the first version of this step. one shard is one sample, so the axis a
    # split is taken on has exactly one group in it - `drop_distribution` would
    # report `testable: False` on every shard, and a per-group `n_before` does
    # not exist inside a shard at all. the honest output is the per-group pair
    # this shard contributes; `drop_distribution` and
    # `assert_funnel_not_confounded` then run ONCE over every shard's pairs,
    # which is the only level at which "is the drop concentrated" has an answer.
    table = ctx.get("table") or []
    first, last = funnel.stages[0], funnel.stages[-1]
    group = str(table[0].get(axis)) if table and axis in table[0] else None
    contribution = {group: (first.n_in - last.n_out, first.n_in)} if group else {}
    dist = {"axis": axis, "testable": False, "scope": "one shard",
            "note": ("a shard holds one group of this axis. aggregate "
                     "`funnel_contribution` over every shard and pass it to "
                     "drop_distribution(), which is where the question is "
                     "answerable")}

    if features is not None and supervision is not None:
        # the guard raises for a direct caller, where an exception is the only
        # way to stop a notebook. inside the chain that would land as `error`,
        # which says the step broke - and this step did not break, the cohort
        # failed a check. the distinction is the whole of the status vocabulary.
        try:
            assert_no_circular_supervision(features, str(supervision))
        except CircularSupervision as e:
            return _jrec("join_funnel", "fail", str(e),
                         "refused - the score this table would produce measures "
                         "whether a vector can find itself",
                         {"feature_columns": sorted(features),
                          "supervision": str(supervision),
                          "guard": "G6"}), {}

    rec = funnel.record(step_id="join_funnel")
    return _jrec("join_funnel", "pass",
                 rec.result,
                 rec.decision or "every dropped unit is attributable to a stage",
                 {"n_in": first.n_in, "n_out": last.n_out, "reason": "",
                  "stages": funnel.as_dicts(), "drop_distribution": dist,
                  "funnel_contribution": {k: list(v) for k, v in contribution.items()},
                  "circular_supervision_checked": bool(
                      features is not None and supervision is not None)},
                 method=rec.method,
                 criterion="every stage states both ends, and every drop a reason",
                 ), {"funnel": funnel.as_dicts(),
                     "funnel_contribution": contribution}


# the order IS the method. `requires` is asserted at BUILD time, so a chain that
# standardises before the coverage drop raises before any data is touched.
JOIN_STEPS: list[Step] = [
    Step(id="unit_inventory", fn=unit_inventory,
         produces=frozenset({"units"}),
         params_space=("required_assays",)),
    Step(id="subject_resolution", fn=subject_resolution,
         produces=frozenset({"subject"}), requires=frozenset({"units"}),
         params_space=("subject_labels", "on_absent")),
    Step(id="neighbourhood", fn=neighbourhood,
         produces=frozenset({"members"}), requires=frozenset({"units"}),
         params_space=("k_neighbors", "neighbour_source", "padding")),
    Step(id="pool_continuous", fn=pool_continuous,
         produces=frozenset({"pooled"}), requires=frozenset({"members"}),
         params_space=("pooled_assays", "data_columns")),
    Step(id="centre_labels", fn=centre_labels,
         produces=frozenset({"label_values"}), requires=frozenset({"units"}),
         params_space=("centre_only_labels",)),
    Step(id="assemble", fn=assemble,
         produces=frozenset({"rows"}),
         requires=frozenset({"pooled", "label_values", "subject"}),
         params_space=("provenance",)),
    Step(id="anchor_coverage", fn=anchor_coverage,
         produces=frozenset({"table"}), requires=frozenset({"rows"}),
         params_space=("anchor_assay", "on_absent")),
    Step(id="standardise", fn=standardise,
         produces=frozenset({"standardised"}), requires=frozenset({"table"}),
         params_space=("standardise_assays", "group_by", "eps", "ddof")),
    Step(id="join_funnel", fn=join_funnel,
         requires=frozenset({"standardised"}),
         params_space=("confounder_axis", "feature_columns", "supervision")),
]
