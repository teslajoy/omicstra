"""a cohort needs a declared place to be WRITTEN, separate from where it is read.

`settings.runs_dir` resolved under the package repo root. right for a source
checkout, where the repo IS the root, and wrong for an installed server with no
checkout: it puts a cohort's ledger beside site-packages.

the obvious fix would be a bug. `project.json#runs_dir` points at the PUBLISHED
GRID - the seed cohort declares `../../runs/tnbc-92_v3`, which `read_only_inputs`
lists under `refuse_writes` - so aiming the ledger there would point the one
thing the server writes at the one thing it must never write. the two meanings
were conflated because on a checkout they sit side by side.
"""
from __future__ import annotations

import json

import pytest

from omicstra.settings import settings


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    root = tmp_path / "acohort"
    (root / "data").mkdir(parents=True)
    # `platform` is required by ProjectConfig. without it the load raises and
    # `read_only_inputs` returns [] - so the oracle guard would protect nothing
    # and this file's refusal test would pass for the wrong reason.
    (root / "project.json").write_text(
        json.dumps({"project_id": "acohort", "platform": "synthetic"}))
    monkeypatch.setattr(settings, "project_dir", root)
    return root


def _declare(root, **fields):
    d = json.loads((root / "project.json").read_text())
    d.update(fields)
    (root / "project.json").write_text(json.dumps(d))


# --- the fallback moves nothing --------------------------------------------
def test_an_undeclared_cohort_keeps_the_historical_path(cohort):
    """declaring nothing must move nothing, or every existing ledger orphans."""
    assert settings.work_root(None) == settings.resolve(settings.runs_dir) / "acohort"


def test_the_seed_cohort_is_unmoved(monkeypatch):
    monkeypatch.setattr(settings, "project_dir", None)
    if not (settings.resolve(settings.projects_dir) / "tnbc-92").is_dir():
        pytest.skip("seed cohort absent")
    assert settings.work_root("tnbc-92") == \
        settings.resolve(settings.runs_dir) / "tnbc-92"


# --- a declaration redirects it --------------------------------------------
def test_a_declared_work_dir_is_relative_to_the_cohort(cohort):
    _declare(cohort, work_dir="work")
    assert settings.work_root(None) == (cohort / "work").resolve()


def test_an_absolute_work_dir_is_taken_as_given(cohort, tmp_path):
    elsewhere = tmp_path / "scratch" / "acohort"
    _declare(cohort, work_dir=str(elsewhere))
    assert settings.work_root(None) == elsewhere


def test_work_dir_is_not_runs_dir(cohort):
    """the distinction this whole change exists for: runs_dir is where the
    finished grid is READ from and is usually declared read-only."""
    _declare(cohort, runs_dir="../../runs/acohort_v3", work_dir="work")
    assert settings.work_root(None) == (cohort / "work").resolve()
    assert "acohort_v3" not in str(settings.work_root(None))


# --- and it is refused if it lands in an oracle -----------------------------
def test_a_work_root_inside_a_read_only_path_is_refused(cohort):
    """a declared path is one a person wrote, so it can point anywhere -
    including at the grid the ports are diffed against. writing there makes the
    next diff compare a port against its own output and pass.
    """
    from omicstra.dispatch import WouldOverwriteOracle, assert_work_root_writable

    (cohort / "grid").mkdir()
    _declare(cohort, work_dir="grid",
             read_only_inputs={"policy": "refuse_writes", "paths": ["grid"]})
    with pytest.raises(WouldOverwriteOracle) as e:
        assert_work_root_writable(None)
    assert "read-only" in str(e.value) and "work_dir" in str(e.value)


def test_a_work_root_outside_every_oracle_passes(cohort):
    from omicstra.dispatch import assert_work_root_writable

    (cohort / "grid").mkdir()
    _declare(cohort, work_dir="work",
             read_only_inputs={"policy": "refuse_writes", "paths": ["grid"]})
    assert assert_work_root_writable(None) == (cohort / "work").resolve()


# --- every writer goes through it ------------------------------------------
def test_the_ledger_follows_the_declaration(cohort):
    from omicstra.contracts.routing import ledger

    _declare(cohort, work_dir="work")
    assert ledger(None).path == (cohort / "work" / "records" / "decisions.jsonl")


def test_no_writer_resolves_runs_dir_directly_any_more():
    """four call sites read `settings.resolve(settings.runs_dir)`. one resolver
    now, or the next writer added will reach for the wrong one."""
    import re
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[1] / "src" / "omicstra"
    bad = [f"{f.relative_to(pkg)}:{i}"
           for f in pkg.rglob("*.py")
           for i, line in enumerate(f.read_text().splitlines(), 1)
           if re.search(r"resolve\(\s*(?:self\.|settings\.)runs_dir\s*\)", line)
           and f.name != "settings.py"]
    assert not bad, f"still resolving runs_dir directly: {bad}"
