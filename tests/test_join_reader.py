"""the declared reader that makes JOIN_STEPS reachable.

tier 1: declaration-level and synthetic, so it runs anywhere. the cohort-scale
check - that the reader feeds the chain to the same parquets the script wrote -
is in test_join_slicediff.py.

the thing under test is that nothing is inferred. a loader is resolved by
declared name, a non-data column list is required rather than sniffed, and an
assay that borrows another's row index says so.
"""
from __future__ import annotations

import json

import pytest

from omicstra.adapters.join import (
    LOADERS,
    REQUIRED,
    JoinDeclarationMissing,
    declared,
)


def _project(tmp_path, join_inputs=None, **over):
    d = tmp_path / "cohort_y"
    d.mkdir(exist_ok=True)
    base = {"project_id": "cohort-y", "platform": "p"}
    if join_inputs is not None:
        base["join_inputs"] = join_inputs
    (d / "project.json").write_text(json.dumps({**base, **over}))
    return d


def _spec(**over):
    s = {
        "neighbour_source": "st",
        "neighbour_column": "neighbours",
        "assays": {"st": {"loader": "parquet_per_sample", "path": "st"}},
    }
    return {**s, **over}


# ------------------------------------------------------- the loader registry --
def test_the_four_shapes_the_seed_cohort_presents_are_all_registered():
    assert set(LOADERS) == {"npy_with_meta", "parquet_per_sample",
                            "parquet_one_file", "table_one_file"}


@pytest.mark.parametrize("loader_name", ["csv", "zarr", "", None])
def test_a_loader_this_package_does_not_have_is_refused(tmp_path, loader_name):
    spec = _spec(assays={"st": {"loader": loader_name, "path": "st"}})
    with pytest.raises(JoinDeclarationMissing) as e:
        declared(_project(tmp_path, spec))
    msg = str(e.value)
    assert "parquet_per_sample" in msg, "the refusal must list what exists"
    if loader_name:
        assert repr(loader_name) in msg or loader_name in msg


# --------------------------------------------------------- the declaration ---
def test_a_cohort_declaring_no_join_inputs_is_refused_with_the_field_list():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as t:
        p = _project(Path(t))
        with pytest.raises(JoinDeclarationMissing) as e:
            declared(p)
        msg = str(e.value)
        assert "join_inputs" in msg
        for field in REQUIRED:
            assert field in msg, f"{field} not named in the refusal"


@pytest.mark.parametrize("field", sorted(REQUIRED))
def test_each_required_field_is_refused_by_name(tmp_path, field):
    spec = {k: v for k, v in _spec().items() if k != field}
    with pytest.raises(JoinDeclarationMissing, match=field):
        declared(_project(tmp_path, spec))


def test_a_neighbour_source_that_is_not_an_assay_is_refused(tmp_path):
    """two assays can disagree about a neighbourhood, so the authoritative one
    is declared - and it has to be one that exists."""
    spec = _spec(neighbour_source="nosuch")
    with pytest.raises(JoinDeclarationMissing, match="nosuch"):
        declared(_project(tmp_path, spec))


def test_a_complete_declaration_is_accepted(tmp_path):
    out = declared(_project(tmp_path, _spec()))
    assert out["neighbour_source"] == "st"
    assert set(out["assays"]) == {"st"}


# ------------------------------------------------- what must not be inferred --
def test_the_one_file_parquet_loader_requires_its_non_data_columns(tmp_path):
    """the seed cohort's build script had to name five such columns by hand, and
    selecting by dtype pulls pixel coordinates into the vector."""
    import pandas as pd

    from omicstra.adapters.join import LOADERS as L

    d = tmp_path / "x"
    d.mkdir()
    pd.DataFrame({"subarray": ["s1"], "spot_id": ["u1"], "pixel_x": [1779.0],
                  "v0": [0.5]}).to_parquet(d / "all.parquet")
    spec = {"loader": "parquet_one_file", "path": "x/all.parquet",
            "key_column": "subarray", "unit_column": "spot_id"}
    with pytest.raises(JoinDeclarationMissing, match="non_data_columns"):
        L["parquet_one_file"](spec, "s1", tmp_path)

    ok = L["parquet_one_file"]({**spec, "non_data_columns":
                                ["subarray", "spot_id", "pixel_x"]}, "s1", tmp_path)
    assert list(ok.columns) == ["v0"], "a coordinate reached the vector"


def test_an_assay_borrowing_an_index_must_name_one_already_loaded(tmp_path):
    """280 .npy files with no meta of their own. read wrong, this silently
    permutes every vector against its unit."""
    import numpy as np

    from omicstra.adapters.join import LOADERS as L

    d = tmp_path / "cells"
    d.mkdir()
    np.save(d / "s1.npy", np.zeros((3, 4), dtype=np.float32))
    spec = {"loader": "npy_with_meta", "path": "cells",
            "index_from": "niche"}
    with pytest.raises(JoinDeclarationMissing, match="index_from"):
        L["npy_with_meta"](spec, "s1", tmp_path, assays={})


def test_a_borrowed_index_of_the_wrong_length_is_refused(tmp_path):
    import numpy as np
    import pandas as pd

    from omicstra.adapters.join import LOADERS as L

    d = tmp_path / "cells"
    d.mkdir()
    np.save(d / "s1.npy", np.zeros((3, 4), dtype=np.float32))
    other = pd.DataFrame(index=["u1", "u2"])
    with pytest.raises(JoinDeclarationMissing, match="permuted"):
        L["npy_with_meta"]({"loader": "npy_with_meta", "path": "cells",
                            "index_from": "niche"}, "s1", tmp_path,
                           assays={"niche": other})


def test_vectors_and_their_meta_must_be_of_the_same_run(tmp_path):
    import numpy as np
    import pandas as pd

    from omicstra.adapters.join import LOADERS as L

    d = tmp_path / "n"
    d.mkdir()
    np.save(d / "s1.npy", np.zeros((3, 4), dtype=np.float32))
    pd.DataFrame(index=["u1", "u2"]).to_csv(d / "s1_meta.tsv", sep="\t")
    with pytest.raises(JoinDeclarationMissing, match="same run"):
        L["npy_with_meta"]({"loader": "npy_with_meta", "path": "n"}, "s1", tmp_path)


# ------------------------------------------------------------ label roles ----
@pytest.mark.parametrize("role", ["", None, "both", "mean"])
def test_a_label_with_no_declared_role_is_refused(tmp_path, role):
    """pooling a discrete class is meaningless, so the declaration says which
    labels are averaged over the neighbourhood and which come from the centre."""
    from omicstra.adapters.join import arrays

    spec = _spec(labels={"cls": {"loader": "table_one_file", "path": "t.tsv",
                                 "key_column": "s", "unit_column": "u",
                                 "role": role}})
    p = _project(tmp_path, spec)
    with pytest.raises((JoinDeclarationMissing, FileNotFoundError)) as e:
        arrays(p, "s1")
    if isinstance(e.value, JoinDeclarationMissing) and "role" in str(e.value):
        assert "pooled" in str(e.value) and "centre" in str(e.value)
