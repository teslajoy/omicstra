"""the relation between what an encoder was built for and what a cohort supplies.

a frozen encoder has a unit it was trained on. a cohort has a unit it actually
hands over. nothing else in the package states the two side by side, and without
that a reader has to hold a model card and a platform declaration in their head
at once to know whether a number means anything.

this decides nothing. it reports a relation and names the evidence on both
sides, because the decision is the `encoder_compatibility` gate's - and that gate
is a MEASURED probe, not a tolerance check, for the reason this module makes
visible: an encoder can be out of distribution by CLASS, where no margin is small
enough.

cohort-free, in the way the rest of the package is. it reads the caller's own
declarations and the package's encoder metadata. it does not know a reference
cohort, and comparing two cohorts is two calls rather than a hardcoded seed -
which is also the only shape that works when a server is bounded to one cohort.
"""
from __future__ import annotations

from typing import Any

from omicstra.models.encoders import EncoderUnavailable, spec


def _supplied(platform: dict) -> dict:
    """what this platform hands an encoder, from its own declaration."""
    tile = (platform.get("he_tile") or {})
    return {
        "resolution_class": platform.get("resolution_class"),
        "spot_diameter_um": platform.get("spot_diameter_um"),
        "spot_pitch_um": platform.get("spot_pitch_um"),
        "cells_per_unit": platform.get("cells_per_spot"),
        "tile_um": tile.get("tile_um"),
        "transcriptome_scope": platform.get("transcriptome_scope"),
    }


def _fit_he(exp: Any, sup: dict) -> dict:
    """a tile encoder against a tile the cohort cuts.

    the comparison is a ratio rather than a verdict. 'in distribution' is not a
    threshold this module is entitled to set - it states how far off the unit is
    and leaves the line to the gate.
    """
    have, want = sup.get("tile_um"), exp.footprint_um
    if have is None:
        return {"relation": "not_stated",
                "why": "this platform declares no tile footprint, so there is nothing to "
                       "compare. declare he_tile.tile_um to get an answer."}
    if want is None:
        return {"relation": "encoder_states_none",
                "why": "the encoder declares no fixed footprint, so any tile is as "
                       "in-distribution as any other by this measure."}
    ratio = float(have) / float(want)
    return {
        "relation": "comparable",
        "supplied_um": float(have), "expected_um": float(want),
        "ratio": round(ratio, 3),
        "shortfall_pct": round((1 - ratio) * 100, 1) if ratio < 1 else 0.0,
        "why": (f"the cohort hands {have} um of tissue per tile where the weights were "
                f"built on {want} um. same kind of unit, different span - a margin, not "
                f"a class mismatch."),
    }


def _fit_molecular(exp: Any, sup: dict, spec_: Any) -> dict:
    """a subcellular encoder against a spot platform is a CLASS mismatch.

    this is the case a tolerance cannot express. Novae's unit is a cell; a 100 um
    spot is ~200 cells of averaged expression. there is no footprint ratio that
    makes those the same kind of thing, which is exactly why the compatibility
    gate probes rather than compares.
    """
    have, want = sup.get("resolution_class"), exp.resolution_class
    if have is None:
        return {"relation": "not_stated",
                "why": "this platform declares no resolution_class."}
    if have in want:
        return {"relation": "in_class", "supplied": have, "expected": list(want),
                "why": "the cohort's unit is the kind of unit the weights were built on."}
    out = {
        "relation": "out_of_class", "supplied": have, "expected": list(want),
        "why": (f"the weights were built on {' or '.join(want)} data and this cohort "
                f"supplies {have}. no footprint tolerance expresses this - it is a "
                f"different kind of unit, so compatibility is measured by probe."),
    }
    if sup.get("cells_per_unit"):
        out["aggregation"] = (
            f"~{sup['cells_per_unit']} cells averaged into one unit, against weights "
            f"trained on single cells")
    return out


def encoder_fit(cfg: Any, platforms: dict, platform_by_sample: dict | None = None) -> dict:
    """for each encoder this cohort declares: built-for vs supplied.

    `cfg` is the cohort's ProjectConfig - the encoders come from its declaration,
    so a cohort that names an encoder the package does not know is told that,
    with the list of names it could have meant.
    """
    counts: dict[str, int] = {}
    for p in (platform_by_sample or {}).values():
        counts[p] = counts.get(p, 0) + 1

    rows = []
    for enc in cfg.encoders:
        try:
            s = spec(enc.name)
        except EncoderUnavailable as e:
            rows.append({"encoder": enc.name, "relation": "undeclared_encoder", "why": str(e)})
            continue
        if s.expects is None:
            rows.append({"encoder": enc.name, "relation": "encoder_states_none",
                         "why": "this encoder declares no unit expectation."})
            continue

        per_platform = []
        for pname, pdef in platforms.items():
            sup = _supplied(pdef)
            fit = (_fit_he(s.expects, sup) if s.modality == "he"
                   else _fit_molecular(s.expects, sup, s))
            row = {"platform": pname, "n_samples": counts.get(pname), "supplied": sup, **fit}
            if s.min_units is not None:
                row["min_units"] = s.min_units
                row["min_units_note"] = (
                    "a floor on units per sample, not a footprint. the platform_floor gate "
                    "reads it; samples below it are the cohort's problem to answer, not the "
                    "encoder's to tolerate.")
            per_platform.append(row)

        rows.append({
            "encoder": s.name, "role": s.role, "modality": s.modality, "dim": s.dim,
            "unit": s.unit,
            "built_for": {"resolution_class": list(s.expects.resolution_class),
                          "footprint_um": s.expects.footprint_um,
                          "declared_not_measured": s.expects.declared,
                          "source": s.expects.source},
            "trained_on": s.trained_on,
            "per_platform": per_platform,
        })

    return {
        "project_id": getattr(cfg, "project_id", None),
        "encoders": rows,
        "decides": "nothing. this states a relation and names the evidence on both sides; "
                   "the encoder_compatibility gate decides, by probe",
        "comparing_cohorts": "call this once per cohort. the package holds no reference "
                             "cohort, and a server is bounded to one",
    }
