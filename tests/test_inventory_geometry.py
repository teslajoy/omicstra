"""the lattice is measured, and a declaration that contradicts it escalates.

the seed cohort's pitch was declared 200 um and measured 150, found by hand
months after every micron-denominated figure had been scaled by the wrong
number. this step is that discovery as a measurement, per sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omicstra.protocols.inventory import (
    INVENTORY_STEPS,
    PITCH_TOLERANCE,
    _lattice,
    _pixel_size_um,
    geometry,
)

ROOT = Path(__file__).resolve().parents[1]


class _Obs:
    def __init__(self, cols=()):
        self.columns = list(cols)

    def __getitem__(self, k):
        return np.arange(4)


class _Obj:
    """the smallest thing that answers what the step asks of an object."""

    def __init__(self, xy, diameter_px=None, cols=()):
        self.obsm = {"spatial": np.asarray(xy, dtype=float)}
        self.obs = _Obs(cols)
        self.uns = ({"spatial": {"lib": {"scalefactors": {
            "spot_diameter_fullres": diameter_px}}}} if diameter_px else {})


def _grid(nx=6, ny=6, step=200.0):
    return np.array([[i * step, j * step] for i in range(nx) for j in range(ny)], dtype=float)


def _ctx(obj, *, pitch=None, diameter=None, sid="S1", platform="p"):
    from omicstra.protocols import inventory as inv

    inv._CACHE["/fake/S1.h5ad"] = obj
    return {"project_dir": str(ROOT), "project_id": None, "params": {},
            "paths_by_sid": {sid: "/fake/S1.h5ad"},
            "platform_by_sample": {sid: platform},
            "platform_defs": {platform: {"spot_pitch_um": pitch,
                                         "spot_diameter_um": diameter}}}


# --- the step is registered, after what it needs ---------------------------
def test_geometry_runs_after_coords_and_platform():
    step = next(s for s in INVENTORY_STEPS if s.id == "geometry")
    assert {"coords", "platform"} <= step.requires
    assert [s.id for s in INVENTORY_STEPS].index("geometry") > \
           [s.id for s in INVENTORY_STEPS].index("coords")


# --- the maths --------------------------------------------------------------
def test_spacing_is_the_median_first_neighbour_distance():
    assert _lattice(_grid(step=200.0))["nn"] == pytest.approx(200.0)
    assert _lattice(np.zeros((2, 2)))["nn"] is None      # too few to have a lattice


def test_pixel_size_comes_from_the_object_or_says_why_not():
    um, how = _pixel_size_um(_Obj(_grid(), diameter_px=100.0), 50.0)
    assert um == pytest.approx(0.5) and how == "object scalefactors"
    assert _pixel_size_um(_Obj(_grid()), 50.0) == (None, "object carries no scalefactors")
    assert _pixel_size_um(_Obj(_grid(), diameter_px=100.0), None)[1].endswith("spot diameter")


# --- the escalation ---------------------------------------------------------
def test_a_lattice_that_agrees_with_the_declaration_passes():
    # 200 px spacing, 100 px spot for a 100 um spot -> 1 um/px -> 200 um pitch
    obj = _Obj(_grid(step=200.0), diameter_px=100.0)
    rec, _ = geometry(_ctx(obj, pitch=200.0, diameter=100.0))
    row = rec.observed["rows"][0]
    assert row["um_per_px"] == pytest.approx(1.0)
    assert row["pitch_measured_um"] == pytest.approx(200.0)
    assert row["agrees"] is True and not rec.caveats


def test_a_declaration_the_lattice_contradicts_escalates_without_halting():
    """the seed cohort's defect, in miniature: declared 200, lattice says 150."""
    obj = _Obj(_grid(step=150.0), diameter_px=100.0)
    rec, _ = geometry(_ctx(obj, pitch=200.0, diameter=100.0))
    row = rec.observed["rows"][0]
    assert row["pitch_measured_um"] == pytest.approx(150.0)
    assert row["agrees"] is False
    assert row["pitch_rel_delta"] == pytest.approx(0.25)
    assert rec.status == "pass", "a readable cohort must not be halted by an open question"
    assert rec.caveats and "micron-denominated" in rec.caveats[0]
    assert "person resolves" in rec.decision


def test_a_disagreement_inside_the_tolerance_does_not_escalate():
    obj = _Obj(_grid(step=200.0 * (1 - PITCH_TOLERANCE / 2)), diameter_px=100.0)
    rec, _ = geometry(_ctx(obj, pitch=200.0, diameter=100.0))
    assert rec.observed["rows"][0]["agrees"] is True and not rec.caveats


# --- what it refuses to do --------------------------------------------------
def test_a_pitch_is_never_derived_from_a_declared_pixel_size():
    """deriving it from a declared um/px that was itself computed from a pitch
    would test the declaration against itself - the shape of the original error."""
    obj = _Obj(_grid(step=158.6))                        # no scalefactors
    rec, _ = geometry(_ctx(obj, pitch=150.0, diameter=100.0))
    row = rec.observed["rows"][0]
    assert row["nn_units"] == pytest.approx(158.6)
    assert row["pitch_measured_um"] is None
    assert "against itself" in row["why_not_derived"]
    assert not rec.caveats, "an unmeasurable pitch cannot contradict anything"


def test_a_platform_with_no_spot_diameter_is_not_measurable():
    """an imaging platform binned into a grid has spacing but no pitch."""
    obj = _Obj(_grid(step=470.0), diameter_px=100.0, cols=("array_row", "array_col"))
    rec, _ = geometry(_ctx(obj, pitch=None, diameter=None))
    row = rec.observed["rows"][0]
    assert row["pitch_measured_um"] is None
    assert row["pixel_size_from"] == "platform declares no spot diameter"
    assert row["grid"] == [4, 4]


# --- the seed cohort, where its data is present -----------------------------
def test_the_seed_cohort_reproduces_its_recorded_spacing():
    rec = ROOT / "projects" / "tnbc-92" / "inventory.json"
    if not rec.is_file():
        pytest.skip("no inventory record on this machine")
    d = json.loads(rec.read_text()).get("geometry")
    if not d:
        pytest.skip("record predates the geometry step")
    rows = d["observed"]["rows"]
    nn = [r["nn_units"] for r in rows if r.get("nn_units")]
    assert len(rows) >= 280
    assert float(np.median(nn)) == pytest.approx(158.6, abs=0.5), "the measured lattice moved"
    assert all(r["pitch_measured_um"] is None for r in rows), (
        "this cohort's objects carry no pixel size; a derived pitch would be circular")
