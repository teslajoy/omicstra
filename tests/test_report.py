"""the report is a template: same code, any cohort, sections driven by inputs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from omicstra.report import gather, render

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "projects" / "tnbc-92"

SYNTHETIC = {
    "project": {"project_id": "two-method", "platform": "a platform", "k_neighbors": 4,
                "encoders": [{"name": "enc_a", "dim": 8, "role": "primary"}]},
    "cohort": {"subject_id_column": "donor", "compute_backend": "local"},
    "pack": {"evidence_version": 1, "runs_root": "runs/x", "families_not_declared": ["z"],
             "tasks": {"one_task": {
                 "task_id": "one_task", "outcome": "recommend", "winner": "m1",
                 "metric": "a score", "source": "an artifact", "margin": 0.2,
                 "floor": {"id": "chance", "value": 0.0},
                 "candidates": [{"id": "m1", "value": 0.9}, {"id": "m2", "value": 0.7}],
                 "guards": [], "note": "a person wrote this"}}},
    "records": [{"step_id": "align", "status": "pass", "actor": "deterministic",
                 "params": {"resolves_only": False}, "duration_s": 1.5}],
}


def test_a_two_method_cohort_renders_through_the_same_code():
    md = render(SYNTHETIC, "two-method report")
    for heading in ("1 · cohort", "4 · per question", "5 · the matrix",
                    "6 · caveats and notes", "7 · provenance"):
        assert heading in md
    assert "m1" in md and "0.9" in md
    assert "a person wrote this" in md


def test_the_seed_cohort_renders_through_the_same_code():
    if not (PROJ / "routing_evidence.json").is_file():
        pytest.skip("cohort absent")
    md = render(gather(PROJ))
    assert md.startswith("# ") and len(md) > 1000
    assert "cross_modal_retrieval" in md


def test_a_section_with_no_input_says_so_and_does_not_vanish():
    bare = {"pack": {"tasks": {}}}
    md = render(bare)
    assert "2 · admissibility" in md and "not applicable" in md
    assert "3 · what was computed" in md


def test_a_family_with_no_evidence_is_one_line_not_a_gap():
    b = dict(SYNTHETIC)
    b["pack"] = dict(b["pack"], tasks=dict(b["pack"]["tasks"],
                     absent_task={"task_id": "absent_task", "outcome": "not_available",
                                  "why": "never scored on this cohort", "candidates": []}))
    md = render(b)
    assert "never scored on this cohort" in md
    assert md.count("absent_task") >= 2          # its block and its matrix row


def test_a_pack_without_outcomes_is_not_given_one():
    """rendering must not derive a decision - that is promote's job, and a second
    source for one decision is how two reports disagree."""
    b = dict(SYNTHETIC)
    t = {k: v for k, v in b["pack"]["tasks"]["one_task"].items()
         if k not in ("outcome", "winner")}
    b["pack"] = dict(b["pack"], tasks={"one_task": t})
    md = render(b)
    assert "not_derived" in md and "run promote" in md


def test_the_renderer_names_no_cohort():
    src = (ROOT / "src" / "omicstra" / "report.py").read_text().lower()
    for token in ("tnbc", "wang", "virchow", "novae", "patient_id", "archetype"):
        assert token not in src, f"{token!r} is a cohort fact, not a template"


def test_gather_skips_what_a_cohort_does_not_have(tmp_path):
    (tmp_path / "cohort.json").write_text('{"subject_id_column": "donor"}')
    b = gather(tmp_path)
    assert "cohort" in b and "pack" not in b and "platform" not in b
    assert render(b)                                   # still renders


def test_gather_reports_an_unreadable_file_rather_than_dying(tmp_path):
    (tmp_path / "cohort.json").write_text("{not json")
    b = gather(tmp_path)
    assert b["unreadable"][0]["file"] == "cohort.json"


def test_human_fields_are_marked_as_authored():
    md = render(SYNTHETIC)
    i = md.index("6 · caveats")
    assert "authored" in md[i:i + 400]
