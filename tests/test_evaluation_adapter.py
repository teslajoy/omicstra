"""tests for adapters/evaluation.py - biology_labels adapter.

the adapter is cohort-independent: it reads a declaration from project.json
and propagates patient-level labels to niche level via an explicit join key.
these tests use made-up column and label names to prove that.
"""
from __future__ import annotations

import json
import textwrap

import numpy as np
import pytest

from omicstra.adapters.evaluation import DeclarationMissing, biology_labels


@pytest.fixture()
def project_tree(tmp_path):
    """a minimal project dir with a labels file using non-tnbc names."""
    labels = tmp_path / "labels.tsv"
    labels.write_text(textwrap.dedent("""\
        subject\tcolor_class\tsize_class\tshape_class
        P1\tred\tsmall\tround
        P2\tblue\tlarge\tsquare
        P3\tred\tmedium\tround
    """))

    proj = {
        "project_id": "test-cohort",
        "biology_labels": {
            "labels_path": "labels.tsv",
            "labels_key": "subject",
            "subject_key": "subject",
            "columns": {
                "colour": "color_class",
                "size": "size_class",
            },
        },
    }
    (tmp_path / "project.json").write_text(json.dumps(proj))

    # 9 niches: 3 subarrays x 3 niches, 3 patients
    subarrays = np.array(["S1", "S1", "S1", "S2", "S2", "S2", "S3", "S3", "S3"])
    patients = np.array(["P1", "P1", "P1", "P2", "P2", "P2", "P3", "P3", "P3"])
    return tmp_path, subarrays, patients


def test_biology_labels_reads_declared_mapping(project_tree):
    d, subs, pats = project_tree
    result = biology_labels(d, subs, pats)
    assert result is not None
    labs = result["labels"]
    assert set(labs.keys()) == {"colour", "size"}
    assert list(labs["colour"][:3]) == ["red", "red", "red"]
    assert list(labs["colour"][3:6]) == ["blue", "blue", "blue"]
    assert list(labs["size"][:3]) == ["small", "small", "small"]
    assert list(labs["size"][3:6]) == ["large", "large", "large"]


def test_biology_labels_report_counts(project_tree):
    d, subs, pats = project_tree
    result = biology_labels(d, subs, pats)
    report = result["report"]
    assert report["n_units"] == 9
    assert report["n_patients"] == 3
    assert report["n_labels_file"] == 3
    assert report["aliases"] == ["colour", "size"]
    assert report["sources"] == ["color_class", "size_class"]
    assert "missing_patients" not in report


def test_biology_labels_tracks_missing_patients(tmp_path):
    labels = tmp_path / "labels.tsv"
    labels.write_text("pid\ttrait\nX1\thigh\nX2\tlow\n")

    proj = {
        "project_id": "test",
        "biology_labels": {
            "labels_path": "labels.tsv",
            "labels_key": "pid",
            "subject_key": "pid",
            "columns": {"trait_alias": "trait"},
        },
    }
    (tmp_path / "project.json").write_text(json.dumps(proj))

    subs = np.array(["A", "A", "B", "B"])
    pats = np.array(["X1", "X1", "X3", "X3"])  # X3 not in labels
    result = biology_labels(tmp_path, subs, pats)
    report = result["report"]
    assert report["n_missing"] == 1
    assert "X3" in report["missing_patients"]
    labs = result["labels"]
    assert labs["trait_alias"][0] == "high"
    assert labs["trait_alias"][2] is None


def test_biology_labels_refuses_inconsistent_subarray(tmp_path):
    labels = tmp_path / "labels.tsv"
    labels.write_text("pid\tval\nA\t1\nB\t2\n")

    proj = {
        "project_id": "test",
        "biology_labels": {
            "labels_path": "labels.tsv",
            "labels_key": "pid",
            "subject_key": "pid",
            "columns": {"v": "val"},
        },
    }
    (tmp_path / "project.json").write_text(json.dumps(proj))

    subs = np.array(["S1", "S1"])
    pats = np.array(["A", "B"])  # two patients in one subarray
    with pytest.raises(DeclarationMissing, match="multiple patients"):
        biology_labels(tmp_path, subs, pats)


def test_biology_labels_refuses_duplicate_patients(tmp_path):
    labels = tmp_path / "labels.tsv"
    labels.write_text("pid\tval\nA\t1\nA\t2\n")

    proj = {
        "project_id": "test",
        "biology_labels": {
            "labels_path": "labels.tsv",
            "labels_key": "pid",
            "subject_key": "pid",
            "columns": {"v": "val"},
        },
    }
    (tmp_path / "project.json").write_text(json.dumps(proj))

    subs = np.array(["S1"])
    pats = np.array(["A"])
    with pytest.raises(DeclarationMissing, match="duplicate"):
        biology_labels(tmp_path, subs, pats)


def test_biology_labels_refuses_missing_source_column(tmp_path):
    labels = tmp_path / "labels.tsv"
    labels.write_text("pid\tval\nA\t1\n")

    proj = {
        "project_id": "test",
        "biology_labels": {
            "labels_path": "labels.tsv",
            "labels_key": "pid",
            "subject_key": "pid",
            "columns": {"v": "nonexistent_col"},
        },
    }
    (tmp_path / "project.json").write_text(json.dumps(proj))

    subs = np.array(["S1"])
    pats = np.array(["A"])
    with pytest.raises(DeclarationMissing, match="nonexistent_col"):
        biology_labels(tmp_path, subs, pats)


def test_biology_labels_returns_none_when_undeclared(tmp_path):
    proj = {"project_id": "test"}
    (tmp_path / "project.json").write_text(json.dumps(proj))
    assert biology_labels(tmp_path, np.array(["S1"]), np.array(["P1"])) is None


def test_biology_labels_numeric_patient_id_coercion(tmp_path):
    """labels file keys on int, niche frame carries int - should still join."""
    labels = tmp_path / "labels.tsv"
    labels.write_text("pid\ttrait\n1\thigh\n2\tlow\n")

    proj = {
        "project_id": "test",
        "biology_labels": {
            "labels_path": "labels.tsv",
            "labels_key": "pid",
            "subject_key": "pid",
            "columns": {"t": "trait"},
        },
    }
    (tmp_path / "project.json").write_text(json.dumps(proj))

    subs = np.array(["S1", "S2"])
    pats = np.array([1, 2])
    result = biology_labels(tmp_path, subs, pats)
    assert result["labels"]["t"][0] == "high"
    assert result["labels"]["t"][1] == "low"