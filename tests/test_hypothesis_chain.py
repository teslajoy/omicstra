"""H2 and H3 get the fixture-pinned acceptance H1 already had, plus their gates.

H1 was pinned with a sha256 so its acceptance did not depend on a gitignored
runs/ tree. H2 and H3 are pinned the same way, and each carries the gate its
hypothesis needs: H2 scores clustering against a label, so the LABEL is gated
(G1); H3 claims cross-patient transfer, so the SPLIT is checked (G2).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from omicstra.protocols.evaluate import coverage, gate_h2, gate_h3

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "tests" / "fixtures"
GRID = ["B1_v3", "B2_v3", "B3_v3", "B4_v3", "R1_v3",
        "R2_v3", "R3_v3", "R4_v3", "R5_v3", "R6_v3"]


def _load(h):
    return (json.loads((F / f"{h}_summary_v3.json").read_text()),
            json.loads((F / f"{h}_summary_v3.meta.json").read_text()))


# --- pinned, and intact -----------------------------------------------------
@pytest.mark.parametrize("h", ["h1", "h2", "h3"])
def test_each_pinned_summary_is_intact(h):
    """against the sha recorded at pinning, so an edit to a fixture fails here
    rather than shifting a number the acceptance tests then accept."""
    meta = json.loads((F / f"{h}_summary_v3.meta.json").read_text())
    got = hashlib.sha256((F / f"{h}_summary_v3.json").read_bytes()).hexdigest()
    assert got == meta["sha256"]


@pytest.mark.parametrize("h", ["h2", "h3"])
def test_the_fixture_matches_what_eval_wrote(h):
    """content, not bytes: key order and whitespace may differ from the source."""
    src = ROOT / "runs" / "tnbc-92_v3" / "eval" / h.upper() / "summary.json"
    if not src.is_file():
        pytest.skip("run grid not present on this machine")
    pinned, _ = _load(h)
    assert json.loads(src.read_text()) == pinned


# --- coverage is declared, not discovered -----------------------------------
@pytest.mark.parametrize("h,missing", [("h1", []), ("h2", ["B4_v3"]), ("h3", ["R5_v3"])])
def test_each_rollup_declares_the_runs_it_did_not_score(h, missing):
    """each rollup drops a different run. a missing run is a gap in a comparison,
    so the meta says so and this test holds the meta to the data."""
    summary = json.loads((F / f"{h}_summary_v3.json").read_text())
    cov = coverage(summary, GRID)
    assert cov["missing"] == missing
    if h != "h1":
        _, meta = _load(h)
        assert meta["missing_from_grid"] == missing


# --- H2: the label is gated before the score --------------------------------
def test_h2_rollup_is_scored_against_a_label_that_fails_g1():
    """archetype is assigned per patient, so NMI vs patient_id is 0.645 on the
    niche join. every ARI in this rollup is therefore diagnostic."""
    summary, meta = _load("h2")
    assert meta["clusters_against"] == "archetype"
    g = gate_h2(summary, meta["label_gate"]["nmi"], "archetype")
    assert g["claims_biology"] is False
    assert g["reference_status"].startswith("diagnostic")


def test_the_label_gate_is_measured_not_trusted():
    """the NMI in the meta is recomputed from the join where the join exists."""
    join = ROOT / "data" / "embeddings" / "niches_v3"
    if not join.is_dir():
        pytest.skip("niche join not present on this machine")
    pa = pytest.importorskip("pyarrow.parquet")
    import pandas as pd

    from omicstra.protocols.evaluate import nmi_label_confounder

    files = sorted(join.glob("*.parquet"))
    df = pd.concat([pa.read_table(f, columns=["patient_id", "archetype"]).to_pandas()
                    for f in files], ignore_index=True)
    r = nmi_label_confounder(df["archetype"].astype(str), df["patient_id"])
    _, meta = _load("h2")
    assert r.value == pytest.approx(meta["label_gate"]["nmi"], abs=5e-4)
    assert r.passed is meta["label_gate"]["passed"]


def test_h2_rollup_reference_is_not_the_routing_packs_reference():
    """0.295 here, against archetype; 0.164 in the pack, against mc_megacluster.
    same field name, different quantity - they must not be compared."""
    summary, _ = _load("h2")
    ev = ROOT / "projects" / "tnbc-92" / "routing_evidence.json"
    rollup_ref = summary["raw_reference"]["raw_he"]["ari"]
    assert rollup_ref == pytest.approx(0.2949, abs=1e-3)
    if ev.is_file():
        assert "0.164" in ev.read_text(), "the pack's reference moved - revisit this test"


def test_a_label_that_passes_g1_is_allowed_to_claim_biology():
    summary, _ = _load("h2")
    assert gate_h2(summary, 0.437, "mc_megacluster")["claims_biology"] is True


# --- H3: cross-patient is checked -------------------------------------------
def test_h3_every_run_uses_the_same_held_out_patients():
    summary, meta = _load("h3")
    g = gate_h3(summary, meta["split"]["test_patients"])
    assert g["cross_patient"] is True and g["test_patients"] == [14]
    assert next(x for x in g["guards"] if x["guard"] == "G2_held_out_honesty")["status"] == "pass"


def test_h3_a_split_that_varies_across_runs_fails():
    summary, _ = _load("h3")
    broken = json.loads(json.dumps(summary))
    next(iter(broken["per_run"].values()))["n_patients"] = 3
    assert gate_h3(broken, 14)["cross_patient"] is False


def test_h3_reports_patient_identity_as_a_confounder_strength_not_a_verdict():
    """0.94 for R4 against 0.071 chance: the denominator a biology claim beats."""
    summary, meta = _load("h3")
    probe = gate_h3(summary, meta["split"]["test_patients"])["patient_identity_in_aligned_space"]
    assert probe["R4_v3"] == pytest.approx(0.94, abs=0.01)


def test_h3_fixture_says_it_is_not_the_published_headline():
    _, meta = _load("h3")
    assert "eval_h3_pathway_cca" in meta["not_the_headline"]


# --- no guard verdict is invented -------------------------------------------
@pytest.mark.parametrize("gate,args", [(gate_h2, ("archetype",)), (gate_h3, ())])
def test_guards_without_inputs_are_not_run_and_say_what_is_missing(gate, args):
    """a guard applied to a number it was never given is a fabricated verdict."""
    h = "h2" if gate is gate_h2 else "h3"
    summary, meta = _load(h)
    out = gate(summary, 0.645, *args) if gate is gate_h2 else gate(summary, meta["split"]["test_patients"])
    skipped = [g for g in out["guards"] if g.get("status") == "not_run"]
    assert skipped, "every rollup lacks inputs for at least one guard"
    for g in skipped:
        assert "needs " in g["note"]
