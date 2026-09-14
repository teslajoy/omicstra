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
