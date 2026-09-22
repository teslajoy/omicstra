"""what a run will cost, from measurements - or a refusal.

the plan could say how many shards there were and how much disk they needed,
and nothing else. the resource that has actually stopped a run here is memory,
and an estimate invented from another device would be worse than none.
"""
from __future__ import annotations

import pytest

from omicstra.models.encoders import EncoderCost, EncoderSpec, spec
from omicstra.protocols.encode import (
    ComputeRefused,
    capacity_plan,
    estimate,
    machine,
    pool_resources,
)

# the field names are the data contract's pool vocabulary, not names of this
# test's choosing - that is the property being protected.
LAPTOP = {"memory_gb_per_task": 16.0, "free_disk_gb": 500.0, "cpus_per_task": 8}
SMALL = {"memory_gb_per_task": 2.0, "free_disk_gb": 1.0, "cpus_per_task": 2}


def test_the_machine_reports_what_it_can_measure():
    m = machine()
    assert m["cpus_per_task"] and m["cpus_per_task"] > 0
    for k in ("memory_gb_per_task", "free_disk_gb"):
        assert k in m, "a resource that cannot be measured is reported as None, not omitted"


def test_a_measured_machine_speaks_the_declared_pools_vocabulary():
    """a declared pool and a measured one are compared field by field. if the
    names diverge, a declared limit is read by nothing and enforced by nothing."""
    import json

    from omicstra.settings import settings
    fields = set(json.loads(
        (settings.resolve(settings.configs_dir) / "data_contract.json").read_text()
    )["pool"]["fields_in_1_2"])
    measured = set(machine()) - {"measured_at"}
    assert measured <= fields, f"machine() invents {measured - fields}, which no pool may declare"


def test_free_space_is_read_at_the_output_location():
    """the question is whether the OUTPUT volume holds it, not the current one."""
    m = machine(at="/")
    assert m["free_disk_gb"] is not None and m["measured_at"] == "/"


def test_an_output_directory_that_does_not_exist_yet_still_has_a_volume():
    m = machine(at="/tmp/omicstra-not-created-yet/deeper/still")
    assert m["free_disk_gb"] is not None, "planning must work before the run makes its dirs"


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


def test_a_resource_with_no_value_is_reported_unchecked_not_passed():
    """silence is the failure mode: a pool that declares no disk must not read as
    a pool whose disk was checked and found sufficient."""
    e = estimate("virchow2", 1000, device="mps", pool={"memory_gb_per_task": 64.0})
    assert e["verdict"] == "fits" and e["not_checked"] == ["disk"]
    assert "never checked" in e["advice"]


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


# --- the cohort's own plan -------------------------------------------------
def test_a_pool_declared_by_name_only_is_not_mistaken_for_a_checked_one():
    res, note = pool_resources({"pool": "mac"})
    assert res is None and "name only" in note


def test_a_pool_declaring_a_field_no_gate_reads_is_refused():
    """the limit that is declared and never compared against is the one that
    fails at the end of the run it should have prevented."""
    with pytest.raises(ComputeRefused, match="which no gate reads"):
        pool_resources({"pool": {"id": "x", "memory_gb": 16}})


def test_a_plan_needs_counts_and_a_device_and_says_which_is_missing():
    cohort = {"pool": "mac", "output_dir": "data/embeddings"}
    assert "no unit counts" in capacity_plan(cohort, "virchow2")["why_not"]
    no_dev = capacity_plan(cohort, "virchow2", unit_counts={"a": 10})
    assert no_dev["verdict"] == "unknown" and "no device" in no_dev["why_not"]


def test_a_plan_multiplies_the_cohorts_own_counts_by_the_encoders_measurement():
    cohort = {"pool": "mac", "output_dir": "data/embeddings"}
    plan = capacity_plan(cohort, "virchow2", unit_counts={"a": 286250}, device="mps")
    assert plan["hours"] == pytest.approx(5.8, abs=0.1)
    assert plan["n_units"] == 286250 and plan["output_dir"] == "data/embeddings"


def test_a_device_declared_on_the_pool_is_enough_to_plan_with():
    cohort = {"pool": {"id": "mac", "device": "mps", "memory_gb_per_task": 64.0,
                       "free_disk_gb": 500.0}}
    plan = capacity_plan(cohort, "virchow2", unit_counts={"a": 1000})
    assert plan["estimated"] is True and plan["device"] == "mps"
