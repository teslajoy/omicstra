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

what it does decide is which UNITS survive, and there are four such decisions.
each is ported as built and declared, for the reason every other as-built
declaration in this project exists: the ten-arm comparison is fair only while
every arm saw the same rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


# the four as-built decisions. each names the alternative it forecloses, so the
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
        kept = 100.0 * last.n_out / first.n_in if first.n_in else 0.0
        return DiagnosticRecord(
            step_id=step_id, status="pass",
            method="itemised funnel - unit count at each stage with a reason for every drop",
            scope=f"{len(self.stages)} stage(s)",
            observed={"stages": self.as_dicts()},
            criterion="every dropped unit is attributable to a named stage",
            result=f"{first.n_in:,} -> {last.n_out:,} ({kept:.1f}% retained)",
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
    """the four as-built decisions, as a record, so they reach the ledger."""
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
