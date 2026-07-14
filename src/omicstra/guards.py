"""the guard library - the generalizable construct-validity IP.

each guard is a dataset-agnostic check; the tnbc-92 value is just the example
that proved it fires. eval refuses to report a number whose guard fails.
"""
from __future__ import annotations

from pydantic import BaseModel


class GuardResult(BaseModel):
    name: str
    passed: bool
    value: float | None = None
    note: str = ""


def nmi_label_confounder(label, confounder, tau: float = 0.5) -> GuardResult:
    # G1: NMI(candidate_label, batch/patient) > tau -> label carries the confounder.
    # tnbc-92: archetype vs patient_id = 0.89 -> demote to diagnostic.
    raise NotImplementedError


def held_out_honesty(within_score: float, cross_score: float) -> GuardResult:
    # G2: eval on the confounder-held-out split only.
    # tnbc-92: within-subarray 0.667 -> cross-subarray 0.000.
    raise NotImplementedError


def bio_vs_confounder_ratio(bio_z: float, confounder_z: float) -> GuardResult:
    # G3: ratio < 1 -> anti-helpful. tnbc-92: B1 CCA 0.071.
    raise NotImplementedError


def specificity_matrix(pathway_axes) -> GuardResult:
    # G4: pair magnitude z with off-diagonal cosine (aliasing). tnbc-92: R4 0.75.
    raise NotImplementedError


def noise_floor(embedding) -> GuardResult:
    # G5: permutation noise-floor before promotion.
    # same/diff cos within eps + ratio < 1 = collapse (v2 gpath2vec signature).
    raise NotImplementedError


def circular_supervision(st_features: list[str], supervision: str) -> GuardResult:
    # G6: the supervision signal must not appear in the input vector. static assert.
    raise NotImplementedError