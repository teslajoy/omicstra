"""the ST graph declaration, and the one field in it that moves an embedding.

Novae builds its neighbour topology from coordinates, which does not care about
units, but it feeds every edge length to the GAT scaled by `scale_to_microns`.
these tests hold three things in place: the scale cannot default, the seed
cohort declares the value its cache was actually built with, and the wrapper
applies a scale for one call without leaking it into the next.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from omicstra.protocols.encode import StGraph

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "projects" / "tnbc-92" / "platform.json"


def _platform(st_graph: dict) -> dict:
    return {"platforms": {"p": {"st_graph": st_graph}}}


def test_an_undeclared_scale_is_refused_not_defaulted():
    with pytest.raises(ValueError, match="scale_to_microns"):
        StGraph.from_platform(_platform({"method": "delaunay", "radius_cap_px": None}), "p")


def test_a_platform_with_no_st_graph_is_refused():
    with pytest.raises(ValueError, match="method, scale_to_microns"):
        StGraph.from_platform({"platforms": {"p": {}}}, "p")


def test_the_scale_reaches_the_record():
    g = StGraph.from_platform(
        _platform({"method": "delaunay", "radius_cap_px": None, "scale_to_microns": 0.5}), "p")
    assert g.params() == {"st_graph": "delaunay", "radius_cap_px": None,
                          "scale_to_microns": 0.5}


def test_the_seed_cohort_declares_the_scale_its_cache_was_built_at():
    """the declaration and the run manifest are two records of one run.

    the manifest is not in CI, so the comparison runs where the cache exists; the
    declaration itself is checked everywhere.
    """
    platform = json.loads(PLATFORM.read_text())
    g = StGraph.from_platform(platform, "original_st")
    decl = platform["platforms"]["original_st"]["st_graph"]
    assert decl["scale_as_built"] is True
    assert decl["identity_field"] == "scale_to_microns"

    alt = decl["scale_declared_alternative"]
    assert alt["run"] is False and alt["question"]

    manifest = ROOT / "data" / "embeddings" / "novae_niche_full" / "run_manifest.json"
    if not manifest.is_file():
        pytest.skip("novae cache absent on this machine")
    built = json.loads(manifest.read_text())["um_per_pixel_spot"]
    assert g.scale_to_microns == pytest.approx(built, rel=0, abs=1e-12), (
        f"platform.json declares {g.scale_to_microns}; the cache was built at {built}")


class _FakeModel:
    def __init__(self, settings):
        self.settings, self.seen = settings, None

    def compute_representations(self, adata, zero_shot):
        self.seen = self.settings.scale_to_microns
        adata.obsm["novae_latent"] = "latent"


@pytest.fixture
def fake_novae(monkeypatch):
    settings = types.SimpleNamespace(scale_to_microns=1.0)
    monkeypatch.setitem(sys.modules, "novae", types.SimpleNamespace(settings=settings))
    return settings


def test_novae_applies_the_declared_scale_for_the_call_and_restores_it(fake_novae):
    from omicstra.models.encoders import _Novae

    model = _FakeModel(fake_novae)
    adata = types.SimpleNamespace(obsm={})
    assert _Novae(model).embed(adata, scale_to_microns=1.2195121951219512) == "latent"
    assert model.seen == 1.2195121951219512
    assert fake_novae.scale_to_microns == 1.0, "one cohort's scale leaked into the process"


def test_novae_restores_the_scale_when_the_call_fails(fake_novae):
    from omicstra.models.encoders import _Novae

    class Boom(_FakeModel):
        def compute_representations(self, adata, zero_shot):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _Novae(Boom(fake_novae)).embed(types.SimpleNamespace(obsm={}), scale_to_microns=0.9)
    assert fake_novae.scale_to_microns == 1.0


def test_novae_has_no_scale_default(fake_novae):
    from omicstra.models.encoders import _Novae

    with pytest.raises(TypeError):
        _Novae(_FakeModel(fake_novae)).embed(types.SimpleNamespace(obsm={}))
    with pytest.raises(ValueError):
        _Novae(_FakeModel(fake_novae)).embed(types.SimpleNamespace(obsm={}), scale_to_microns=0)
