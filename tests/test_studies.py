"""a study groups projects, the way GDC/Gen3 nests project under program.

`program` is deliberately NOT the word: `routing_contract.json` already spends
it on a biological abstraction level - pathway questions are `level: program`,
above cohort and structure - and the router's vocabulary should not have to
share it with a directory.

a project belongs to exactly ONE study, which is Gen3's rule and what a path can
express. a project in two studies would be a declaration, not a directory.
"""
from __future__ import annotations

import json

import pytest

from omicstra.contracts import declarations as D
from omicstra.settings import settings


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """a studies root and a flat projects root side by side, as a migration
    would actually look: some cohorts moved, some not yet."""
    monkeypatch.setattr(settings, "repo_root", tmp_path)
    monkeypatch.setattr(settings, "project_dir", None)

    flat = tmp_path / "projects" / "oldcohort"
    flat.mkdir(parents=True)
    (flat / "project.json").write_text(json.dumps(
        {"project_id": "oldcohort", "platform": "p"}))

    s = tmp_path / "studies" / "bforepc" / "projects" / "tnbc"
    s.mkdir(parents=True)
    (s / "project.json").write_text(json.dumps(
        {"project_id": "tnbc", "k_neighbors": 4}))
    (tmp_path / "studies" / "bforepc" / "study.json").write_text(json.dumps(
        {"platform": "study platform", "k_neighbors": 6, "seed": 42}))
    return tmp_path


# --- resolution -------------------------------------------------------------
def test_a_flat_project_still_resolves(tree):
    """nothing on disk has to move for the study level to exist."""
    assert settings.project_root("oldcohort") == tree / "projects" / "oldcohort"
    assert settings.study_root("oldcohort") is None


def test_a_study_qualified_id_resolves(tree):
    assert settings.project_root("bforepc/tnbc") == \
        tree / "studies" / "bforepc" / "projects" / "tnbc"
    assert settings.study_root("bforepc/tnbc") == tree / "studies" / "bforepc"


def test_a_flat_id_finds_a_project_that_has_moved_under_a_study(tree):
    """so an id in a committed record keeps resolving after a migration."""
    assert settings.project_root("tnbc") == \
        tree / "studies" / "bforepc" / "projects" / "tnbc"


def test_the_same_project_name_under_two_studies_REFUSES(tree):
    """serving one of them would be the 'another cohort's data' failure the
    boundary exists to stop."""
    other = tree / "studies" / "htan" / "projects" / "tnbc"
    other.mkdir(parents=True)
    (other / "project.json").write_text(json.dumps({"project_id": "tnbc"}))
    with pytest.raises(ValueError, match="2 studies"):
        settings.project_root("tnbc")
    # qualified, it is unambiguous again
    assert settings.project_root("htan/tnbc") == other


# --- inheritance ------------------------------------------------------------
def test_a_project_inherits_what_it_does_not_declare(tree):
    v, origin = D.resolve("seed", "bforepc/tnbc")
    assert v == 42 and origin == "study.json#seed"


def test_a_project_overrides_what_it_does_declare(tree):
    v, origin = D.resolve("k_neighbors", "bforepc/tnbc")
    assert v == 4, "the project's own value wins"
    assert origin == "project.json#k_neighbors"


def test_absent_at_both_levels_resolves_to_nothing_rather_than_a_default(tree):
    """the package's first rule: declared, not assumed. absent is a refusal the
    CALLER makes, and this reports absence rather than inventing a value."""
    assert D.resolve("nonesuch", "bforepc/tnbc") == (None, None)


def test_the_origin_distinguishes_a_study_act_from_a_project_act(tree):
    """a study-level declaration binds every project under it, which is a bigger
    act than a project-level one. an actor ledger that could not tell them apart
    would under-report what was decided."""
    p = D.provenance("bforepc/tnbc")
    assert p["seed"] == "study.json"
    assert p["k_neighbors"] == "project.json"


def test_a_flat_project_consults_no_study_file(tree):
    assert [n for n, _ in D.sources("oldcohort")] == ["cohort.json", "project.json"]
    assert [n for n, _ in D.sources("bforepc/tnbc")] == \
        ["study.json", "cohort.json", "project.json"]


# --- and the work root follows ---------------------------------------------
def test_the_work_root_follows_a_study_qualified_project(tree):
    assert settings.work_root("bforepc/tnbc") == \
        settings.resolve(settings.runs_dir) / "bforepc/tnbc"


def test_a_declared_work_dir_still_wins_under_a_study(tree):
    proj = tree / "studies" / "bforepc" / "projects" / "tnbc"
    d = json.loads((proj / "project.json").read_text())
    d["work_dir"] = "work"
    (proj / "project.json").write_text(json.dumps(d))
    assert settings.work_root("bforepc/tnbc") == (proj / "work").resolve()
