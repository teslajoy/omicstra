"""H2 and H3 get the fixture-pinned acceptance H1 already had, plus their gates.

H1 was pinned with a sha256 so its acceptance did not depend on a gitignored
runs/ tree. H2 and H3 are pinned the same way, and each carries the gate its
hypothesis needs: H2 scores clustering against a label, so the LABEL is gated
(G1); H3 claims cross-patient transfer, so the SPLIT is checked (G2).

for H2 and H3 the eval.py rollups are DIAGNOSTIC: no routed number is read from
them. acceptance is held to the artifacts the evidence pack cites -
metrics_h2_ci.json (H2-C) and per_pathway_cca.parquet (H3) - pinned below.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from omicstra.protocols.evaluate import (
    coverage,
    gate_h2,
    gate_h2c,
    gate_h3,
    gate_h3_pathway,
)

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


# --- the acceptance pins: the artifacts the evidence pack cites -------------
# the rollups above stay pinned as diagnostics. these are what a routed number
# is read from, so these are what acceptance holds to.
PACK = ROOT / "projects" / "tnbc-92" / "routing_evidence.json"
RUNS = ROOT / "runs" / "tnbc-92_v3"


def _cited(name):
    return (json.loads((F / f"{name}.json").read_text()),
            json.loads((F / f"{name}.meta.json").read_text()))


def _st_features():
    return {rid: json.loads((ROOT / "configs" / "v3" / f"{rid}.json").read_text())["st_features"]
            if (ROOT / "configs" / "v3" / f"{rid}.json").is_file()
            else ["novae_niche", "gpath2vec_niche"]          # B1-B4: align_classical.ST_FEATURES
            for rid in GRID}


@pytest.mark.parametrize("name", ["h2c_metrics_ci_v3", "h3_per_pathway_cca_v3"])
def test_each_cited_pin_is_intact(name):
    meta = json.loads((F / f"{name}.meta.json").read_text())
    assert hashlib.sha256((F / f"{name}.json").read_bytes()).hexdigest() == meta["sha256"]


def test_the_h2c_pin_is_the_artifact_byte_for_byte():
    src = RUNS / "eval" / "H2" / "metrics_h2_ci.json"
    if not src.is_file():
        pytest.skip("run grid not present on this machine")
    assert src.read_bytes() == (F / "h2c_metrics_ci_v3.json").read_bytes()
    _, meta = _cited("h2c_metrics_ci_v3")
    for rid, sha in meta["upstream_sha256"].items():
        got = hashlib.sha256((RUNS / rid / "eval" / "biology.parquet").read_bytes()).hexdigest()
        assert got == sha, f"{rid}'s biology.parquet moved under the pinned ratio"


def test_the_h3_pin_matches_the_parquet_and_its_null():
    src = RUNS / "eval" / "H3" / "pathway_cca_gpath2vec_v3"
    if not src.is_dir():
        pytest.skip("run grid not present on this machine")
    _, meta = _cited("h3_per_pathway_cca_v3")
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    assert sha(src / "per_pathway_cca.parquet") == meta["source_sha256"]
    assert sha(src / "perm_nulls.parquet") == meta["perm_nulls_sha256"]
    assert json.loads((src / "provenance.json").read_text()) == meta["provenance"]


def test_every_h2c_number_the_pack_routes_is_in_the_pin():
    if not PACK.is_file():
        pytest.skip("evidence pack absent")
    ci, _ = _cited("h2c_metrics_ci_v3")
    task = json.loads(PACK.read_text())["tasks"]["subject_identity_suppression"]
    assert "metrics_h2_ci.json" in task["source"]
    for cand in task["candidates"]:
        r = ci["h2c"][cand["id"]]
        assert r["ratio"] == pytest.approx(cand["value"], abs=5e-4)
        assert [r["ratio_ci_low"], r["ratio_ci_high"]] == pytest.approx(cand["ci"], abs=5e-4)


def test_every_h3_number_the_pack_routes_is_in_the_pin():
    if not PACK.is_file():
        pytest.skip("evidence pack absent")
    rows, _ = _cited("h3_per_pathway_cca_v3")
    task = json.loads(PACK.read_text())["tasks"]["pathway_transfer"]
    assert "per_pathway_cca.parquet" in task["source"]
    g = gate_h3_pathway(rows, _st_features())
    for cand in task["candidates"]:
        assert g["runs"][cand["id"]]["n_significant"] == cand["value"], cand["id"]
    z = g["runs"]["R4_v3"]["z"]
    assert (round(z["Immune_System"], 1), round(z["ECM_Organization"], 1),
            round(z["Cell_Cycle"], 1), round(z["Programmed_Cell_Death"], 1)) == (25.4, 16.0, 16.5, 11.2)


def test_the_pins_declare_what_they_do_not_cover():
    ci, m2 = _cited("h2c_metrics_ci_v3")
    assert m2["h2c_missing_from_grid"] == sorted(set(GRID) - set(ci["h2c"])) == ["B2_v3", "B3_v3", "B4_v3"]
    rows, m3 = _cited("h3_per_pathway_cca_v3")
    assert m3["missing_from_grid"] == ["B4_v3"]
    g = gate_h3_pathway(rows, _st_features())
    assert g["runs"]["B3_v3"]["n_significant"] == 4 and "B3_v3" in m3["pack_does_not_list"]


# --- the gates on the cited artifacts ---------------------------------------
def test_h2c_gate_ranks_as_the_pack_does_and_records_g1_by_design():
    ci, _ = _cited("h2c_metrics_ci_v3")
    g = gate_h2c(ci, floor=0.137, floor_source="docs/v2/tnbc92_results_summary.md H2 Part C")
    assert g["ranking"][:3] == ["R6_v3", "R5_v3", "R1_v3"] and g["ranking"][-1] == "B1_v3"
    g1 = next(x for x in g["guards"] if x["guard"] == "G1_label_granularity")
    assert g1["status"] == "satisfied_by_design" and "cross-patient" in g1["note"]


def test_h2c_g3_reports_that_no_run_suppresses_patient_below_biology():
    """every ratio is under 1: in every run the patient z exceeds the biology z.
    the ranking is still meaningful; the guard says what the ranking is not."""
    ci, _ = _cited("h2c_metrics_ci_v3")
    g = gate_h2c(ci)
    assert not any(r["G3"]["passed"] for r in g["runs"].values())
    assert g["runs"]["B1_v3"]["ratio"] < 0.1


def test_h3_gate_is_cross_patient_and_excludes_the_untestable_pathway():
    rows, _ = _cited("h3_per_pathway_cca_v3")
    g = gate_h3_pathway(rows, _st_features())
    g2 = next(x for x in g["guards"] if x["guard"] == "G2_held_out_honesty")
    assert g2["status"] == "pass"
    assert all(r["excluded_untestable"] == ["TGF-beta_Signaling"] for r in g["runs"].values())


def test_h3_g6_derives_circularity_from_st_features_not_the_artifact_flag():
    """the script hardcodes R1-R4. B1-B3 and R5 also carry gpath2vec, so their
    z_st and z_mean rows are marked clean and are circular. the headline view is not."""
    rows, _ = _cited("h3_per_pathway_cca_v3")
    g6 = next(x for x in gate_h3_pathway(rows, _st_features())["guards"]
              if x["guard"] == "G6_circular_supervision")
    assert g6["status"] == "flag_disagreement"
    flagged = {(d["run"], d["view"]) for d in g6["disagreements"]}
    assert flagged == {(r, v) for r in ("B1_v3", "B2_v3", "B3_v3", "R5_v3") for v in ("z_st", "z_mean")}
    assert "not affected" in g6["note"]
