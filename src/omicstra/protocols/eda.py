"""the eda protocol - "is this data usable". ids match configs/eda_contract.json.

not all of it is "signal" - two steps ask whether biology is present
(markers, spatial autocorrelation) and four ask whether the data is trustworthy
(counts, batch structure, segmentation, section alignment). the question the
chain answers is usability, which is what EDA means.

distinct from `omicstra.eda`, which is the gate that JUDGES these records. this
file measures; that one decides. the two merge when the old module is absorbed.

the contract declares WHICH checks exist and what each is judged against. this
file declares the ORDER and the applicability. where the contract has a check
and no implementation yet, the step is registered anyway with no `fn` body -
so it reports `not_run` and appears in the record, rather than being silently
absent. a missing check that says nothing is the defect this closes.
"""
from __future__ import annotations

from omicstra.protocols import eda_steps
from omicstra.protocols import Step
from omicstra.records import DiagnosticRecord

Ctx = dict


def _adata(ctx: Ctx):
    import anndata as ad
    return ad.read_h5ad(ctx["adata_path"])


def _unimplemented(step_id: str):
    def fn(ctx: Ctx) -> DiagnosticRecord:
        return DiagnosticRecord(
            step_id=step_id, status="not_run",
            result="declared in the contract, no implementation registered",
            decision="implement it, or declare it not applicable for this cohort",
        )
    return fn


# --- applicability predicates: measured, never declared ---------------------
def _has_spatial(ctx: Ctx) -> bool:
    import anndata as ad
    return "spatial" in ad.read_h5ad(ctx["adata_path"]).obsm


def _multi_sample(ctx: Ctx) -> bool:
    import anndata as ad
    a = ad.read_h5ad(ctx["adata_path"])
    k = ctx.get("params", {}).get("sample_key", "sample")
    return k in a.obs and a.obs[k].nunique() > 1


def _has_segmentation(ctx: Ctx) -> bool:
    import anndata as ad
    obs = ad.read_h5ad(ctx["adata_path"]).obs
    return any(c in obs for c in ("cell_id", "segmentation", "nucleus_id"))


def _multi_section(ctx: Ctx) -> bool:
    import anndata as ad
    obs = ad.read_h5ad(ctx["adata_path"]).obs
    return "section" in obs and obs["section"].nunique() > 1


def _project(ctx: Ctx) -> dict:
    """the cohort's own declarations. absent is not an error - it is unresolved."""
    import json
    from omicstra.settings import settings
    out = {}
    for name in ("project.json", "cohort.json"):
        f = settings.project_root(ctx.get("project_id")) / name
        if f.exists():
            out.update(json.loads(f.read_text()))
    return out


def _multi_modal(ctx: Ctx) -> bool:
    encs = _project(ctx).get("encoders") or []
    return len({e.get("role") for e in encs if isinstance(e, dict)}) > 1


def _uses_foundation_models(ctx: Ctx) -> bool:
    return bool([e for e in (_project(ctx).get("encoders") or [])
                 if isinstance(e, dict) and e.get("name")])


def _declared_field(field: str, unresolved=("", None)):
    """a check over a DECLARATION rather than the data.

    resolved-or-not is a property of the field; the measurement behind it is a
    property of the cohort. this half is the former.
    """
    def fn(ctx: Ctx) -> DiagnosticRecord:
        v = _project(ctx).get(field)
        ok = v not in unresolved
        return DiagnosticRecord(
            step_id=field, status="pass" if ok else "fail",
            method="read the cohort declaration",
            observed={field: v},
            criterion=f"{field} resolved to a non-empty value",
            result=f"{field} = {v!r}" if ok else f"{field} unresolved",
            decision="proceed" if ok else
                     "decide it and declare it - blocking while unresolved")
    return fn


EDA_STEPS: list[Step] = [
    Step(id="cohort_counts", produces=frozenset({"counts"}),
         fn=lambda c: eda_steps.count_statistics(_adata(c), **c["params"])),

    # the contract declares positive_markers and negative_markers as two
    # required checks with two fields. eda_steps.marker_expression computes both
    # halves and returns ONE record, so each step passes only its own half - the
    # verdict is a conjunction over whichever groups are non-empty, so an empty
    # opposite half is a no-op rather than a silent pass.
    Step(id="positive_markers", requires=frozenset({"counts"}),
         produces=frozenset({"positive_markers"}),
         authority="cohort_calibrated",
         params_space=("positive", "min_pct_positive"),
         fn=lambda c: eda_steps.marker_expression(
             _adata(c), positive=c["params"]["positive"], negative={},
             min_pct_positive=c["params"].get("min_pct_positive", 1.0))),

    Step(id="negative_markers", requires=frozenset({"counts"}),
         produces=frozenset({"negative_markers"}),
         authority="cohort_calibrated",
         params_space=("negative", "max_pct_negative"),
         fn=lambda c: eda_steps.marker_expression(
             _adata(c), positive={}, negative=c["params"]["negative"],
             max_pct_negative=c["params"].get("max_pct_negative", 10.0))),

    Step(id="spatial_autocorrelation", requires=frozenset({"counts"}),
         produces=frozenset({"spatial_autocorrelation"}),
         authority="cohort_calibrated",
         applicable_when=_has_spatial,
         why_not_applicable="no spatial coordinates in obsm - not a spatial assay",
         params_space=("k", "n_perm", "threshold"),
         fn=lambda c: eda_steps.spatial_autocorrelation(_adata(c), **c["params"])),

    Step(id="batch_structure", requires=frozenset({"counts"}),
         produces=frozenset({"batch_structure"}),
         authority="cohort_calibrated",   # contract: the absolute floor is calibrated
         applicable_when=_multi_sample,
         why_not_applicable="single sample - no batch axis to test",
         fn=lambda c: eda_steps.batch_structure(_adata(c), **c["params"])),

    # declared in the contract, not yet implemented. registered so their absence
    # is a RECORD rather than a silence.
    # reads a DECLARATION, not the data: is the encoder input format resolved?
    Step(id="encoder_input_decision", requires=frozenset({"counts"}),
         produces=frozenset({"encoder_input_decision"}),
         fn=_declared_field("encoder_input_decision",
                            unresolved=("", "unresolved", None))),

    # contract: applies_when multi_modal
    Step(id="cross_modal_registration", requires=frozenset({"counts"}),
         produces=frozenset({"cross_modal_registration"}),
         applicable_when=_multi_modal,
         why_not_applicable="single modality declared - no correspondence to verify",
         fn=_unimplemented("cross_modal_registration")),

    # contract: applies_when foundation_model_encoders
    Step(id="model_tissue_fit", requires=frozenset({"counts"}),
         produces=frozenset({"model_tissue_fit"}),
         applicable_when=_uses_foundation_models,
         why_not_applicable="no foundation-model encoders declared",
         fn=_unimplemented("model_tissue_fit")),

    Step(id="segmentation_qc", requires=frozenset({"counts"}),
         produces=frozenset({"segmentation_qc"}),
         applicable_when=_has_segmentation,
         why_not_applicable="no segmentation masks - spot-resolution platform",
         fn=_unimplemented("segmentation_qc")),

    Step(id="multi_section_alignment", requires=frozenset({"counts"}),
         produces=frozenset({"multi_section_alignment"}),
         applicable_when=_multi_section,
         why_not_applicable="single section - nothing to align against",
         fn=_unimplemented("multi_section_alignment")),
]