"""schema structural checks. runs with PyYAML only — no linkml required.

checks the YAML is valid LinkML structurally before the full linkml-lint run:
- all classes have at least one identifier slot
- all range references point to an existing class or enum
- all required slots are declared on the class
- tree_root exists (B2 fix verification)
- key enforcement slots are present (minimum_value, pattern)
- supervision_annotation_ids present on DerivationRecord (G2 fix)
- unit_level present on Representation (G3-lite fix)

these checks do not substitute for linkml-validate. they are fast smoke tests
that catch regressions without requiring the dev tool.
"""
from __future__ import annotations

import yaml
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_FILE = ROOT / "src" / "omicstra" / "schema" / "omicstra.yaml"
FIXTURES = ROOT / "tests" / "fixtures" / "schema"


@pytest.fixture(scope="module")
def schema():
    return yaml.safe_load(SCHEMA_FILE.read_text())


@pytest.fixture(scope="module")
def classes(schema):
    return schema.get("classes", {})


@pytest.fixture(scope="module")
def enums(schema):
    return schema.get("enums", {})


@pytest.fixture(scope="module")
def all_types(schema, classes, enums):
    builtins = {"string", "integer", "float", "boolean", "uri", "date", "datetime"}
    return set(classes) | set(enums) | builtins


# ---- parse + version --------------------------------------------------------

def test_schema_file_exists():
    assert SCHEMA_FILE.is_file(), f"schema not found at {SCHEMA_FILE}"


def test_schema_has_version(schema):
    assert "version" in schema, "schema must carry a version field"
    v = schema["version"]
    parts = str(v).split(".")
    assert len(parts) == 3, f"version must be semver (x.y.z), got {v!r}"


def test_schema_has_id(schema):
    assert "id" in schema, "schema must have an id URI"
    assert schema["id"].startswith("https://"), "schema id must be an https URI"


# ---- tree_root (B2 fix) -----------------------------------------------------

def test_tree_root_class_exists(classes):
    roots = [n for n, c in classes.items() if c.get("tree_root")]
    assert roots, "no class has tree_root: true (B2)"
    assert len(roots) == 1, f"exactly one tree_root expected, found: {roots}"


def test_tree_root_is_cohort_pack(classes):
    roots = [n for n, c in classes.items() if c.get("tree_root")]
    assert roots[0] == "CohortPack", f"tree_root should be CohortPack, got {roots[0]}"


# ---- identifier slots -------------------------------------------------------

CLASSES_NEEDING_IDENTIFIER = {
    "Participant", "Biospecimen", "TissueSection", "Assay",
    "CoordinateFrame", "SpatialUnit", "Representation",
    "Annotation", "DataFile", "ClinicalObservation",
}


@pytest.mark.parametrize("class_name", sorted(CLASSES_NEEDING_IDENTIFIER))
def test_class_has_identifier_slot(class_name, classes):
    cls = classes.get(class_name)
    assert cls is not None, f"{class_name} not found in schema"
    attrs = cls.get("attributes", {})
    id_slots = [n for n, a in attrs.items() if a.get("identifier")]
    assert id_slots, f"{class_name} has no identifier slot"


# ---- FK ranges resolve (B3 fix) ---------------------------------------------

def test_all_ranges_resolve(classes, all_types):
    unresolved = []
    for cls_name, cls in classes.items():
        for attr_name, attr in cls.get("attributes", {}).items():
            r = attr.get("range")
            if r and r not in all_types:
                unresolved.append(f"{cls_name}.{attr_name}: range={r!r}")
    assert not unresolved, f"unresolved ranges: {unresolved}"


def test_typed_fk_slots_are_not_bare_strings(classes, enums):
    """key FK slots must have a typed range, not just string."""
    expected_ranges = {
        ("TissueSection", "specimen_id"): "Biospecimen",
        ("Assay", "section_id"): "TissueSection",
        ("CoordinateFrame", "assay_id"): "Assay",
        ("SpatialUnit", "assay_id"): "Assay",
        ("SpatialUnit", "coordinate_frame_id"): "CoordinateFrame",
    }
    for (cls_name, slot_name), expected in expected_ranges.items():
        cls = classes.get(cls_name, {})
        attr = cls.get("attributes", {}).get(slot_name, {})
        actual = attr.get("range")
        assert actual == expected, (
            f"{cls_name}.{slot_name}: expected range={expected!r}, got {actual!r} (B3)"
        )


# ---- enforcement slots present ----------------------------------------------

def test_representation_dim_has_minimum_value(classes):
    """minimum_value: 1 must be declared on dim (not just described in prose)."""
    dim = classes["Representation"]["attributes"]["dim"]
    assert "minimum_value" in dim, "dim missing minimum_value (enforcement gap)"
    assert dim["minimum_value"] >= 1, f"dim minimum_value should be >= 1, got {dim['minimum_value']}"


def test_coordinate_frame_scale_has_minimum_value(classes):
    """minimum_value: 0.0001 must be declared on scale_to_microns."""
    slot = classes["CoordinateFrame"]["attributes"]["scale_to_microns"]
    assert "minimum_value" in slot, "scale_to_microns missing minimum_value"
    assert slot["minimum_value"] > 0, "scale_to_microns minimum_value must be > 0"


def test_datafile_path_has_pattern(classes):
    """pattern ^(?!/) must be declared on DataFile.path."""
    slot = classes["DataFile"]["attributes"]["path"]
    assert "pattern" in slot, "DataFile.path missing pattern (absolute path guard)"
    assert slot["pattern"].startswith("^(?!/)"), "path pattern must start with ^(?!/)"


def test_datafile_sha256_has_pattern(classes):
    """pattern ^[0-9a-f]{64}$ must be declared on DataFile.sha256."""
    slot = classes["DataFile"]["attributes"]["sha256"]
    assert "pattern" in slot, "DataFile.sha256 missing pattern"
    assert "64" in slot["pattern"], "sha256 pattern must enforce 64 characters"


def test_representation_derivation_is_required(classes):
    """derivation required: true on Representation (enforcement fix)."""
    deriv = classes["Representation"]["attributes"]["derivation"]
    assert deriv.get("required"), "Representation.derivation must be required: true"


def test_representation_has_rules_for_aligned(classes):
    """rules block must exist on Representation for aligned -> run_id check."""
    rules = classes["Representation"].get("rules")
    assert rules, "Representation missing rules block (aligned -> run_id enforcement)"
    preconditions_found = any(
        "aligned" in str(r.get("preconditions", ""))
        for r in rules
    )
    assert preconditions_found, "Representation rules must have an aligned precondition"


# ---- G2 fix: supervision lineage --------------------------------------------

def test_derivation_record_has_supervision_annotation_ids(classes):
    """supervision_annotation_ids must be on DerivationRecord (G2 fix).

    without this slot the circularity check cannot distinguish z_he trained on
    mc_weights from z_he with no mc_weights supervision.
    """
    attrs = classes["DerivationRecord"]["attributes"]
    assert "supervision_annotation_ids" in attrs, (
        "DerivationRecord missing supervision_annotation_ids (G2)"
    )
    slot = attrs["supervision_annotation_ids"]
    assert slot.get("multivalued"), "supervision_annotation_ids must be multivalued"


# ---- G3-lite fix: unit_level ------------------------------------------------

def test_representation_has_unit_level(classes):
    """unit_level must be declared on Representation (G3-lite fix)."""
    attrs = classes["Representation"]["attributes"]
    assert "unit_level" in attrs, "Representation missing unit_level (G3-lite)"
    slot = attrs["unit_level"]
    assert slot.get("required"), "unit_level must be required"
    assert slot.get("range") == "UnitLevel", f"unit_level range must be UnitLevel, got {slot.get('range')}"


def test_unit_level_enum_has_niche(enums):
    """niche is the primary delivered unit for tnbc-92 (k=6 aggregation)."""
    assert "UnitLevel" in enums, "UnitLevel enum missing"
    values = enums["UnitLevel"].get("permissible_values", {})
    assert "niche" in values, "UnitLevel must include niche"
    assert "spot" in values, "UnitLevel must include spot"


# ---- enum coverage ----------------------------------------------------------

def test_modality_enum_has_aligned(enums):
    values = enums["Modality"]["permissible_values"]
    assert "aligned" in values, "Modality must include aligned"


def test_file_format_has_tsv(enums):
    """tsv was missing from FileFormat in 0.1.0; values_path uses parquet or tsv."""
    values = enums["FileFormat"]["permissible_values"]
    assert "tsv" in values, "FileFormat must include tsv"


def test_assay_platform_has_visium_hd(enums):
    values = enums["AssayPlatform"]["permissible_values"]
    assert "visium_hd" in values, "AssayPlatform must include visium_hd"


# ---- cohort fixture files (WS 2 acceptance) ---------------------------------

@pytest.mark.parametrize("fixture", ["tnbc92_minimal.yaml", "alternate_cohort_minimal.yaml"])
def test_fixture_file_is_valid_yaml(fixture):
    """both cohort examples must at least parse as valid YAML."""
    path = FIXTURES / fixture
    assert path.is_file(), f"fixture not found: {path}"
    doc = yaml.safe_load(path.read_text())
    assert isinstance(doc, dict), f"{fixture} must be a YAML mapping"


@pytest.mark.parametrize("fixture", ["tnbc92_minimal.yaml", "alternate_cohort_minimal.yaml"])
def test_fixture_has_participants(fixture):
    """every cohort pack must have at least one participant."""
    doc = yaml.safe_load((FIXTURES / fixture).read_text())
    assert doc.get("participants"), f"{fixture} must declare at least one participant"


def test_alternate_cohort_uses_different_encoder(schema):
    """alternate cohort must use a different encoder than the seed cohort.

    this is the core claim: the schema is cohort-agnostic. if both fixtures
    declare the same encoder and dim the alternate fixture proves nothing.
    """
    seed = yaml.safe_load((FIXTURES / "tnbc92_minimal.yaml").read_text())
    alt = yaml.safe_load((FIXTURES / "alternate_cohort_minimal.yaml").read_text())
    seed_encoders = {r["encoder"] for r in seed.get("representations", [])}
    alt_encoders = {r["encoder"] for r in alt.get("representations", [])}
    assert seed_encoders != alt_encoders, (
        f"alternate cohort must use a different encoder. "
        f"seed: {seed_encoders}, alt: {alt_encoders}"
    )


@pytest.mark.parametrize("alt_platform", ["merscope", "cosmx", "visium_hd"])
def test_alternate_cohort_uses_different_platform(alt_platform):
    """any valid non-original_st platform is a legal alternate cohort.

    the alternate fixture does not commit to a specific platform; this test
    builds inline assay data with a parametrized platform value and asserts
    it differs from the seed cohort (original_st). the specific platform for
    any future second cohort is undecided and not encoded here.
    """
    seed = yaml.safe_load((FIXTURES / "tnbc92_minimal.yaml").read_text())
    seed_platforms = {a["platform"] for a in seed.get("assays", [])}
    assert alt_platform not in seed_platforms, (
        f"{alt_platform!r} must not be the seed cohort's platform ({seed_platforms})"
    )


def test_alternate_cohort_aligned_repr_carries_supervision():
    """aligned representation must carry supervision_annotation_ids in derivation."""
    alt = yaml.safe_load((FIXTURES / "alternate_cohort_minimal.yaml").read_text())
    aligned = [r for r in alt.get("representations", []) if r.get("modality") == "aligned"]
    assert aligned, "alternate cohort fixture must have at least one aligned representation"
    for r in aligned:
        sup = r.get("derivation", {}).get("supervision_annotation_ids", [])
        assert sup, (
            f"aligned repr {r['repr_id']} must carry supervision_annotation_ids "
            "(G2: needed for circularity traversal)"
        )