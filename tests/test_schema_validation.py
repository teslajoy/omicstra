"""schema validation: must-reject and must-accept cases.

each must-reject case corresponds to a rule the schema declares. if linkml
accepts it, the schema is not enforcing what the description says.

the 10 cases here came from a reviewer who ran the 0.1.0 draft through
linkml-lint and gen-pydantic; 8 of 10 passed when they should have been
rejected. this test suite holds the schema to those same probes.

requires linkml (dev dependency). skipped when not installed.
"""
from __future__ import annotations

import textwrap

import pytest

from omicstra.schema import SCHEMA_PATH

linkml_validate = pytest.importorskip(
    "linkml.validator", reason="linkml not installed (dev dependency)"
).validate


def _validate(instance_yaml: str) -> list:
    """return a list of ValidationResult. empty = valid."""
    import yaml as _yaml
    from linkml.validator import validate
    from linkml.validator.report import Severity

    instance = _yaml.safe_load(instance_yaml)
    result = validate(
        instance=instance,
        schema=str(SCHEMA_PATH),
        target_class="CohortPack",
    )
    return [r for r in result.results if r.severity == Severity.ERROR]


# ---- must-reject cases --------------------------------------------------------

def test_representation_missing_derivation_is_rejected():
    """derivation is required: true on Representation."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: test
        representations:
          - repr_id: r1
            modality: morphology
            unit_level: niche
            encoder: virchow2
            dim: 1280
    """)
    errors = _validate(yaml)
    assert errors, "missing derivation should produce a validation error"


def test_representation_dim_zero_is_rejected():
    """dim minimum_value: 1. zero and negative are schema errors."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: test
        representations:
          - repr_id: r1
            modality: morphology
            unit_level: niche
            encoder: virchow2
            dim: 0
            derivation:
              encoder_name: virchow2
    """)
    errors = _validate(yaml)
    assert errors, "dim=0 should produce a validation error"


def test_representation_dim_negative_is_rejected():
    """dim minimum_value: 1. the probe from the original review."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: test
        representations:
          - repr_id: r1
            modality: morphology
            unit_level: niche
            encoder: virchow2
            dim: -5
            derivation:
              encoder_name: virchow2
    """)
    errors = _validate(yaml)
    assert errors, "dim=-5 should produce a validation error"


def test_aligned_representation_without_run_id_is_rejected():
    """rule: aligned modality requires run_id."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: test
        representations:
          - repr_id: r1
            modality: aligned
            unit_level: niche
            encoder: virchow2
            dim: 512
            derivation:
              encoder_name: virchow2
              input_repr_ids:
                - virchow2_niche_v3
    """)
    errors = _validate(yaml)
    assert errors, "aligned without run_id should produce a validation error"


def test_coordinate_frame_scale_zero_is_rejected():
    """scale_to_microns minimum_value: 0.0001. zero is refused."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: test
        assays:
          - assay_id: A001
            section_id: SA001
            platform: original_st
            modality: molecular
        coordinate_frames:
          - frame_id: F001
            assay_id: A001
            unit_label: pixels
            scale_to_microns: 0.0
    """)
    errors = _validate(yaml)
    assert errors, "scale_to_microns=0 should produce a validation error"


def test_coordinate_frame_scale_negative_is_rejected():
    """scale_to_microns minimum_value: 0.0001. negative is refused."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: test
        assays:
          - assay_id: A001
            section_id: SA001
            platform: original_st
            modality: molecular
        coordinate_frames:
          - frame_id: F001
            assay_id: A001
            unit_label: pixels
            scale_to_microns: -1.0
    """)
    errors = _validate(yaml)
    assert errors, "scale_to_microns<0 should produce a validation error"


def test_datafile_absolute_path_is_rejected():
    """path pattern ^(?!/) rejects absolute paths."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: test
        data_files:
          - path: /Users/x/data/embeddings.parquet
            format: parquet
    """)
    errors = _validate(yaml)
    assert errors, "absolute path should produce a validation error"


def test_datafile_malformed_sha256_is_rejected():
    """sha256 pattern ^[0-9a-f]{64}$ rejects short or non-hex strings."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: test
        data_files:
          - path: data/embeddings.parquet
            format: parquet
            sha256: abc
    """)
    errors = _validate(yaml)
    assert errors, "malformed sha256 should produce a validation error"


def test_datafile_uppercase_sha256_is_rejected():
    """sha256 pattern requires lowercase hex only."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: test
        data_files:
          - path: data/embeddings.parquet
            format: parquet
            sha256: ABCDEF1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF1234567890
    """)
    errors = _validate(yaml)
    assert errors, "uppercase sha256 should produce a validation error"


def test_representation_unknown_modality_is_rejected():
    """AssayPlatform and Modality are closed enums. original reviewer probe."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: test
        representations:
          - repr_id: r1
            modality: image
            unit_level: niche
            encoder: virchow2
            dim: 1280
            derivation:
              encoder_name: virchow2
    """)
    errors = _validate(yaml)
    assert errors, "unknown modality should produce a validation error"


# ---- must-accept cases --------------------------------------------------------

def test_valid_minimal_pack_is_accepted():
    """a minimal but complete cohort pack must pass validation."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: tnbc-92
        biospecimens:
          - specimen_id: BIO001
            participant_id: P001
        tissue_sections:
          - section_id: SA001
            specimen_id: BIO001
            section_index: 0
            thickness_um: 16.0
        assays:
          - assay_id: A001
            section_id: SA001
            platform: original_st
            modality: molecular
        coordinate_frames:
          - frame_id: F001
            assay_id: A001
            unit_label: pixels
            scale_to_microns: 0.928
            pitch_um: 150.0
        representations:
          - repr_id: virchow2_niche_v3
            modality: morphology
            unit_level: niche
            encoder: virchow2
            dim: 1280
            derivation:
              encoder_name: virchow2
              encoder_version: v2.0
              input_repr_ids: []
              supervision_annotation_ids: []
        annotations:
          - annotation_id: mc_megacluster
            label_type: discrete
            subject_level: false
            values_path: data/embeddings/niches_v3/mc_megacluster.parquet
            classes:
              - mc1
              - mc2
              - mc3
            coverage_fraction: 0.97
            provenance:
              source: algorithm
              method: NMF k=14 on per-spot pseudobulk expression
        data_files:
          - path: data/embeddings/virchow2_niche/SA001.npy
            format: npy
            sha256: a3f1e2d4b5c6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2
            n_rows: 208786
            n_cols: 1280
    """)
    errors = _validate(yaml)
    assert not errors, f"valid minimal pack should pass. errors: {errors}"


def test_valid_assay_platform_original_st_is_accepted():
    """original_st is a valid closed-enum value. reviewer probe that correctly accepted."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: test
        tissue_sections:
          - section_id: SA001
            specimen_id: BIO001
        assays:
          - assay_id: A001
            section_id: SA001
            platform: original_st
            modality: molecular
    """)
    errors = _validate(yaml)
    assert not errors, f"original_st platform should be accepted. errors: {errors}"


def test_valid_aligned_representation_with_run_id_is_accepted():
    """aligned representation with run_id present must pass the rule."""
    yaml = textwrap.dedent("""\
        participants:
          - participant_id: P001
            cohort: test
        representations:
          - repr_id: z_he_R4_v3
            modality: aligned
            unit_level: niche
            encoder: virchow2
            dim: 512
            run_id: R4_v3
            derivation:
              encoder_name: virchow2
              run_id: R4_v3
              input_repr_ids:
                - virchow2_niche_v3
              supervision_annotation_ids:
                - mc_weights_niche
    """)
    errors = _validate(yaml)
    assert not errors, f"aligned with run_id should be accepted. errors: {errors}"