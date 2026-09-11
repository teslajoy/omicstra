"""no ingest record carries a machine's home directory.

ingest.json is the cohort's provenance record and is meant to be portable by
construction - it names the source file behind every sample, so a reader on
another machine can check what was converted. an absolute path there is a silent
defect: the file looks correct until someone else opens it.

two guards, deliberately at different levels. the first runs everywhere and
tests the rule; the second runs where a record exists and tests the artifact.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# any absolute path a unix or windows machine would produce. `/Users/` and
# `/home/` are the two that actually occurred; the rest close the door.
ABSOLUTE = re.compile(r"(^|[\"\s:])(/Users/|/home/|/root/|[A-Za-z]:\\\\)")


def _load_ingest_wang():
    """import the script by path. it lives outside the package on purpose - a
    cohort's format knowledge is exactly what must not be inside the wheel."""
    path = ROOT / "scripts" / "ingest_wang.py"
    if not path.is_file():
        pytest.skip("scripts/ingest_wang.py absent")
    spec = importlib.util.spec_from_file_location("ingest_wang", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the rule ---------------------------------------------------------------
def test_rel_returns_a_repo_relative_path():
    m = _load_ingest_wang()
    got = m.rel(ROOT / "data" / "inputs" / "Images" / "imagesHD" / "TNBC1_CN1_C1.jpg")
    assert got == "data/inputs/Images/imagesHD/TNBC1_CN1_C1.jpg"
    assert not ABSOLUTE.search(got)


def test_rel_refuses_a_path_outside_the_root():
    """a fallback to the absolute path is what put /Users/ in the record. a
    source outside ROOT needs a declared base, so this raises instead."""
    m = _load_ingest_wang()
    with pytest.raises(ValueError, match="outside the repo root"):
        m.rel(Path("/tmp/somewhere/else/selection.RData"))


def test_every_recorded_path_goes_through_rel():
    """a second writer that bypasses rel() would reintroduce the defect without
    failing anything above, so assert the call sites rather than the outputs."""
    src = (ROOT / "scripts" / "ingest_wang.py").read_text()
    body = src.split("def main(", 1)[1]
    for field in ('"rdata":', '["images"][sid] ='):
        line = next(ln for ln in body.splitlines() if field in ln)
        assert "rel(" in line, f"{field} is recorded without rel(): {line.strip()}"


# --- the artifact -----------------------------------------------------------
@pytest.mark.parametrize("record", sorted(ROOT.glob("projects/*/data/canonical/ingest.json")),
                         ids=lambda p: p.parents[2].name)
def test_no_ingest_record_carries_a_home_directory(record):
    """runs on whatever records this machine has. CI has none and reports no
    tests for this parameter, which is correct - there is nothing to check."""
    text = record.read_text()
    hits = [ln.strip() for ln in text.splitlines() if ABSOLUTE.search(ln)]
    assert not hits, f"{record} carries absolute path(s):\n  " + "\n  ".join(hits)


@pytest.mark.parametrize("record", sorted(ROOT.glob("projects/*/data/canonical/ingest.json")),
                         ids=lambda p: p.parents[2].name)
def test_recorded_sources_resolve_from_the_root(record):
    """relative and WRONG is not an improvement on absolute. every recorded
    source must exist when joined to the root it declares."""
    d = json.loads(record.read_text())
    missing = [v["rdata"] for v in d.get("sources", {}).values()
               if not (ROOT / v["rdata"]).exists()]
    missing += [p for p in d.get("images", {}).values() if not (ROOT / p).exists()]
    assert not missing, f"{record} names {len(missing)} path(s) that do not exist: {missing[:3]}"
