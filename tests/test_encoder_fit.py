"""what an encoder was built for, against what a cohort supplies.

the relation this states is the one a stranger with their own data needs first,
and it is the only place the package puts a model card and a platform
declaration side by side. it decides nothing - the encoder_compatibility gate
decides, by probe - so what is under test is that the relation is stated
honestly and that absence is reported rather than guessed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from omicstra.models.fit import encoder_fit

ROOT = Path(__file__).resolve().parents[1]


class _Cfg:
    """a cohort declaration, without needing a cohort on disk."""
    def __init__(self, encoders, project_id="synthetic"):
        self.encoders = [type("E", (), e)() for e in encoders]
        self.project_id = project_id


def _spot(**over):
    d = {"resolution_class": "spot", "spot_diameter_um": 100.0, "spot_pitch_um": 150.0,
         "cells_per_spot": 200, "he_tile": {"tile_um": 95.9}}
    d.update(over)
    return d


def test_a_tile_encoder_reports_a_ratio_not_a_verdict():
    """same kind of unit, different span - that is a margin, and the module's job
    is to state its size, not to decide whether it is acceptable."""
    r = encoder_fit(_Cfg([{"name": "virchow2"}]), {"p": _spot()})
    fit = r["encoders"][0]["per_platform"][0]
    assert fit["relation"] == "comparable"
    assert fit["expected_um"] == 112.0 and fit["supplied_um"] == 95.9
    assert fit["shortfall_pct"] == pytest.approx(14.4, abs=0.1)
    assert "relation" in r["decides"] and "gate decides" in r["decides"]


def test_a_subcellular_encoder_on_a_spot_platform_is_a_class_mismatch():
    """the case a tolerance cannot express.

    Novae's unit is a cell; a 100 um spot is ~200 cells of averaged expression.
    no footprint ratio makes those the same kind of thing, which is exactly why
    compatibility is probed rather than compared.
    """
    r = encoder_fit(_Cfg([{"name": "novae"}]), {"p": _spot()})
    fit = r["encoders"][0]["per_platform"][0]
    assert fit["relation"] == "out_of_class"
    assert "200 cells" in fit["aggregation"]
    assert fit["min_units"] == 512


def test_one_encoder_can_be_in_and_out_of_class_in_the_same_cohort():
    """platform is a property of a SAMPLE, so the answer is per platform.

    a single cohort-level verdict would hide the case this asserts, and that
    case is real: a breast cohort carrying both spot arrays and subcellular
    imaging gets two different answers for one encoder.
    """
    r = encoder_fit(_Cfg([{"name": "novae"}]),
                    {"spots": _spot(),
                     "imaging": {"resolution_class": "subcellular",
                                 "spot_diameter_um": None, "spot_pitch_um": None}},
                    platform_by_sample={"a": "spots", "b": "spots", "c": "imaging"})
    by = {p["platform"]: p for p in r["encoders"][0]["per_platform"]}
    assert by["spots"]["relation"] == "out_of_class"
    assert by["imaging"]["relation"] == "in_class"
    assert by["spots"]["n_samples"] == 2 and by["imaging"]["n_samples"] == 1


def test_an_undeclared_tile_is_reported_not_assumed():
    """a cohort that has not cut tiles yet has no footprint to compare.

    inventing one from the spot diameter would be the package guessing at cohort
    knowledge - the failure mode this whole split exists to stop.
    """
    r = encoder_fit(_Cfg([{"name": "virchow2"}]), {"p": _spot(he_tile={})})
    fit = r["encoders"][0]["per_platform"][0]
    assert fit["relation"] == "not_stated"
    assert "declare he_tile.tile_um" in fit["why"]


def test_an_unregistered_encoder_names_what_exists():
    r = encoder_fit(_Cfg([{"name": "not_a_real_encoder"}]), {"p": _spot()})
    row = r["encoders"][0]
    assert row["relation"] == "undeclared_encoder"
    assert "virchow2" in row["why"] and "novae" in row["why"]


def test_the_expectation_says_whether_it_was_measured():
    """a number from a model card and a number measured here deserve different
    trust, and a reader should not have to guess which one they are reading."""
    r = encoder_fit(_Cfg([{"name": "virchow2"}]), {"p": _spot()})
    assert r["encoders"][0]["built_for"]["declared_not_measured"] is True


def test_no_reference_cohort_is_hardcoded():
    """comparing two cohorts is two calls.

    a server is bounded to one cohort by OMICSTRA_PROJECT_DIR, so a built-in
    reference could not be read anyway - and hardcoding the seed cohort would
    put a cohort inside the package.
    """
    import inspect

    from omicstra.models import fit
    src = inspect.getsource(fit)
    assert "tnbc" not in src.lower(), "a cohort name leaked into the package"
    note = fit.encoder_fit(_Cfg([]), {})["comparing_cohorts"]
    assert "once per cohort" in note and "no reference cohort" in note
