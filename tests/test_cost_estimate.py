"""what a run will cost, from measurements - or a refusal.

the plan could say how many shards there were and how much disk they needed,
and nothing else. the resource that has actually stopped a run here is memory,
and an estimate invented from another device would be worse than none.
"""
from __future__ import annotations

import pytest

from omicstra.models.encoders import EncoderCost, EncoderSpec, spec
from omicstra.protocols.encode import estimate, machine

LAPTOP = {"memory_gb": 16.0, "free_disk_gb": 500.0, "cpus": 8}
SMALL = {"memory_gb": 2.0, "free_disk_gb": 1.0, "cpus": 2}


def test_the_machine_reports_what_it_can_measure():
    m = machine()
    assert m["cpus"] and m["cpus"] > 0
    for k in ("memory_gb", "free_disk_gb"):
        assert k in m, "a resource that cannot be measured is reported as None, not omitted"


def test_an_estimate_is_time_disk_and_memory_from_a_measurement():
    e = estimate("virchow2", 286250, device="mps", pool=LAPTOP)
    assert e["estimated"] is True
    assert e["hours"] == pytest.approx(5.8, abs=0.1), "the published extraction took 5.8 h"
    assert e["unit"] == "tile", "the estimate multiplies the ENCODER's unit, not samples"
    assert e["output_gb"] > 0 and e["peak_memory_gb"] > 0
    assert e["measured_on"], "a number with no provenance is a guess"


def test_the_same_encoder_costs_differently_per_device():
    mps = estimate("virchow2", 10_000, device="mps", pool=LAPTOP)["hours"]
    cpu = estimate("virchow2", 10_000, device="cpu", pool=LAPTOP)["hours"]
    assert cpu > mps * 2, "a laptop cpu is not a card, and the plan should say so"


def test_an_unmeasured_device_refuses_rather_than_extrapolating():
    e = estimate("virchow2", 100, device="cuda", pool=LAPTOP)
    assert e["estimated"] is False and e["verdict"] == "unknown"
    assert "no measurement" in e["why_not"] and "mps" in e["why_not"]
    assert "hours" not in e


def test_a_machine_too_small_says_which_resource_and_does_not_hedge():
    e = estimate("virchow2", 1_000_000, device="cpu", pool=SMALL)
    assert e["verdict"] == "does_not_fit"
    assert set(e["blocked_by"]) == {"disk", "memory"}
    assert "run it somewhere with more" in e["advice"]


def test_a_declared_walltime_is_checked_when_one_exists():
    """a laptop has no walltime; a scheduler declares one, and a job that cannot
    finish inside it should be refused before it is queued."""
    pool = dict(LAPTOP, walltime_limit_s=3600)
    e = estimate("virchow2", 286250, device="mps", pool=pool)
    wall = next(c for c in e["checks"] if c["resource"] == "walltime")
    assert wall["fits"] is False and wall["need_hours"] == pytest.approx(5.8, abs=0.1)
    assert "walltime" in e["blocked_by"]


def test_the_pool_says_whether_it_was_declared_or_measured():
    assert estimate("virchow2", 10, device="mps", pool=LAPTOP)["pool"] == "declared"
    assert estimate("virchow2", 10, device="mps")["pool"].startswith("measured")


# --- the registry carries the measurement, not the estimator ---------------
def test_the_measured_encoders_carry_their_provenance():
    for name in ("virchow2", "novae"):
        for c in spec(name).cost:
            assert c.seconds_per_unit > 0 and c.bytes_per_unit > 0
            assert c.peak_memory_gb > 0 and c.measured_on


def test_an_encoder_with_no_measurement_is_a_valid_encoder():
    """most encoders will arrive unmeasured; that must not break planning."""
    s = EncoderSpec(name="x", dim=1, role="st", modality="molecular", unit="spot",
                    trained_on="nothing")
    assert s.cost == () and s.cost_on("cpu") is None


def test_a_cost_is_looked_up_by_device():
    s = EncoderSpec(name="x", dim=1, role="st", modality="molecular", unit="spot",
                    trained_on="nothing",
                    cost=(EncoderCost(device="cpu", seconds_per_unit=1.0, bytes_per_unit=8,
                                      peak_memory_gb=1.0, measured_on="a note"),))
    assert s.cost_on("cpu").seconds_per_unit == 1.0
    assert s.cost_on("mps") is None
