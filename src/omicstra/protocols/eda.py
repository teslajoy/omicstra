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

from omicstra import measures
from omicstra.protocols import Step
from omicstra.records import DiagnosticRecord

Ctx = dict


def _samples(ctx: Ctx):
    """this cohort's sections, through the canonical adapter.

    the chain profiles a cohort the way the analysis did - per sample. nothing
    assembles a single object, and `adata_path` is gone: no inventory step ever
    produced one, so the chain could not run through the graph at all.
    """
    from omicstra.adapters.canonical import list_samples
    from omicstra.settings import settings
    return [x for x in list_samples(settings.project_root(ctx.get("project_id")))
            if x.counts is not None]


def _load(sample):
    import anndata as ad
    return ad.read_h5ad(sample.counts)


class _Undeclared(KeyError):
    """a cohort has not declared something the check requires. not a crash."""


def _panel(ctx: Ctx, which: str) -> dict:
    """the marker panel this cohort declares. absent is a refusal, not an error.

    the panel is a property of the tissue and disease context, so it is the
    cohort's to declare. it has never been: the exploratory notebook held it
    inline and the calibration record points at project.json, where it is not.
    saying so beats a KeyError that looks like a bug in the step.
    """
    v = ctx.get("params", {}).get(which)
    if not v:
        raise _Undeclared(
            f"no {which} marker panel declared for this cohort. the panel names genes "
            f"expected in this tissue, so it cannot be inherited from another cohort or "
            f"guessed from the matrix - declare it and re-run.")
    return v


def _attach_coords(a, sample) -> bool:
    """put the section's own coordinates on the object the measure receives.

    the ingest writes coordinates to a SIDECAR parquet on purpose, so the
    morphology arm never loads a counts matrix to read two columns. the spatial
    measure needs both, and reading the sidecar inside the step would put cohort
    file knowledge back in a protocol - so the canonical adapter supplies it and
    this only joins the two on the object it is about to hand over.
    """
    import numpy as np
    from omicstra.adapters.canonical import load_spots

    if "spatial" in a.obsm or sample.spots is None:
        return "spatial" in a.obsm
    try:
        df = load_spots(sample).set_index("spot_id")
    except Exception:
        return False
    idx = [i for i in a.obs_names if i in df.index]
    if len(idx) != a.n_obs:
        return False
    a.obsm["spatial"] = np.asarray(df.loc[list(a.obs_names), ["x", "y"]].values, dtype=float)
    return True


def _aggregate(step_id: str, rows: list, rule: str, p: float | None,
               why_none: str) -> DiagnosticRecord:
    """per-section records -> one status, by the rule the CONTRACT declares.

    the rows stay in the record. a status with no rows underneath cannot be
    argued with, and "which sections failed" is the first question anyone asks.
    """
    ran = [r for r in rows if r.get("status") in ("pass", "fail")]
    if not ran:
        return DiagnosticRecord(
            step_id=step_id, status="not_applicable", result=why_none,
            observed={"per_section": rows[:12], "n_sections": len(rows)},
            decision="no section could be measured - not a verdict about the cohort")

    passed = [r for r in ran if r["status"] == "pass"]
    frac = len(passed) / len(ran)
    if rule == "all":
        ok, crit = len(passed) == len(ran), "every section passes"
    else:
        ok, crit = frac >= p, f"at least {p:.0%} of sections pass"
    return DiagnosticRecord(
        step_id=step_id, status="pass" if ok else "fail",
        method=f"per section, aggregated by the declared rule: {rule}",
        scope=f"{len(ran)} of {len(rows)} sections measurable",
        observed={"n_sections": len(rows), "n_measured": len(ran),
                  "n_passed": len(passed), "fraction": round(frac, 4),
                  "failed_sections": [r.get("scope") or r.get("step_id")
                                      for r in ran if r["status"] == "fail"][:20],
                  "per_section": rows[:12]},
        criterion=crit,
        result=f"{len(passed)}/{len(ran)} sections pass ({frac:.1%})",
        decision="proceed" if ok else
                 "below the declared coverage - a cohort-level failure, not a bad section")


def _per_section(step_id: str, measure, applicable=None, why_not: str = ""):
    """run one measure over every section and aggregate by the contract's rule.

    the measure is untouched: `measures/` is already per sample, and this only
    decides WHICH sections it sees and how their answers combine.
    """
    def fn(ctx: Ctx) -> DiagnosticRecord:
        from omicstra.contracts.eda import load_contract
        spec = next((c for c in load_contract()["checks"] if c["id"] == step_id), {})
        agg = spec.get("aggregation") or {}
        rule, p = agg.get("rule", "all"), agg.get("p")
        if rule == "fraction" and p is None:
            return DiagnosticRecord(
                step_id=step_id, status="not_run",
                result="the contract declares a fraction rule with p unset",
                decision="declare p with a reason - it is never defaulted")

        rows = []
        for smp in _samples(ctx):
            a = _load(smp)
            _attach_coords(a, smp)
            if applicable is not None and not applicable(a, ctx):
                rows.append({"status": "not_applicable", "scope": smp.sample_id,
                             "result": why_not})
                continue
            try:
                rec = measure(a, ctx)
            except _Undeclared as e:
                return DiagnosticRecord(
                    step_id=step_id, status="not_run", result=str(e),
                    decision="declare it - a check with no declaration is not a failure "
                             "of the data")
            d = rec.model_dump() if hasattr(rec, "model_dump") else dict(rec)
            d["scope"] = smp.sample_id
            rows.append(d)
        return _aggregate(step_id, rows, rule, p,
                          why_not or "no section was applicable")
    return fn


def _unimplemented(step_id: str):
    def fn(ctx: Ctx) -> DiagnosticRecord:
        return DiagnosticRecord(
            step_id=step_id, status="not_run",
            result="declared in the contract, no implementation registered",
            decision="implement it, or declare it not applicable for this cohort",
        )
    return fn


# --- applicability predicates: measured, never declared ---------------------
# SECTION-unit predicates take a loaded section. a section that does not
# qualify is recorded not_applicable and excluded from the aggregation, rather
# than failing it - a spot platform has no segmentation to be bad at.
def _has_spatial(a, ctx: Ctx) -> bool:
    return "spatial" in a.obsm


def _has_segmentation(a, ctx: Ctx) -> bool:
    return any(c in a.obs for c in ("cell_id", "segmentation", "nucleus_id"))


# COHORT-unit predicates take the sample list. "more than one sample" is a
# property of the cohort, and asking it inside one object was only ever possible
# because a stacked object was assumed to exist.
def _multi_sample(ctx: Ctx) -> bool:
    return len(_samples(ctx)) > 1


def _multi_section(ctx: Ctx) -> bool:
    return len(_samples(ctx)) > 1


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


def _guarded(step_id: str, call):
    """an undeclared prerequisite is not_run with its reason, never a traceback."""
    try:
        return call()
    except _Undeclared as e:
        return DiagnosticRecord(
            step_id=step_id, status="not_run", result=str(e),
            decision="declare it - a check with no declaration is not a failure of the data")


def _pseudobulk(ctx: Ctx):
    """one row per section - and a refusal when the gene axis is not declared.

    batch_structure looks like it needs a stacked cohort but only ever uses the
    per-sample MEAN, so a few hundred mean vectors supply it exactly. what it
    cannot supply itself is the GENE AXIS: sections here span 24,344 to 27,567
    genes, so stacking requires a decision about how to reconcile them.

    that decision is not this function's. the join declares its own as-built
    semantics for units, and says explicitly that the gene-universe question
    lives upstream and is not settled there. no cohort declares it either. so
    rather than quietly intersecting - which would pick a gene set nobody chose
    and bury it in a mean - this refuses and says what is missing.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    samples = _samples(ctx)
    if not samples:
        return None

    axes, rows, ids = [], [], []
    for smp in samples:
        a = _load(smp)
        axes.append(tuple(a.var_names))
        rows.append(np.asarray(a.X.mean(axis=0)).ravel())
        ids.append(smp.sample_id)

    if len({len(x) for x in axes}) > 1 or len(set(axes)) > 1:
        raise _Undeclared(
            f"the gene axis is not declared. {len(samples)} sections span "
            f"{min(len(x) for x in axes)} to {max(len(x) for x in axes)} genes, so they "
            f"cannot be stacked without a rule for reconciling them - intersection, union "
            f"with zeros, or a declared panel. the join declares unit semantics and states "
            f"that the gene universe is settled upstream, not there; no cohort declaration "
            f"names it. declare it rather than letting a mean be taken over a gene set "
            f"nobody chose.")

    key = ctx.get("params", {}).get("sample_key", "sample")
    return ad.AnnData(np.vstack(rows), obs=pd.DataFrame({key: ids}, index=ids),
                      var=pd.DataFrame(index=list(axes[0])))


def _cohort_counts(ctx: Ctx) -> DiagnosticRecord:
    """the root: does this cohort exist, and how deep is what is in it.

    counts ADD across sections; rates are weighted by n_obs, because a mean of
    per-section means overweights a small section. measures/ is untouched - it is
    called once per section and the combining happens here.

    the verdict is the EXISTENCE question, per the contract: patients present and
    positive. the depth statistics ride along as observation. a cohort of shallow
    sections is a cohort; a cohort of zero units is not.
    """
    samples = _samples(ctx)
    if not samples:
        return DiagnosticRecord(
            step_id="cohort_counts", status="fail",
            method="per section through the canonical adapter",
            criterion="cohort counts present and non-zero",
            result="no section carries counts",
            decision="a cohort with no units is not a cohort - nothing downstream can run")

    rows, n_obs, n_vars, nnz = [], 0, 0, 0
    for smp in samples:
        a = _load(smp)
        r = measures.count_statistics(a, **{k: v for k, v in ctx["params"].items()
                                            if k in ("sample_key", "source")})
        o = (r.observed or {}) if hasattr(r, "observed") else {}
        rows.append({"sample": smp.sample_id, "n_obs": o.get("n_obs"),
                     "n_vars": o.get("n_vars"), "status": r.status})
        n_obs += int(o.get("n_obs") or 0)
        n_vars = max(n_vars, int(o.get("n_vars") or 0))
        nnz += int(o.get("nnz") or 0)

    # the subject axis the cohort declares, bound by the inventory
    subject = (_project(ctx).get("subject_id_column") or "").strip()
    with_units = [r for r in rows if (r["n_obs"] or 0) > 0]
    ok = bool(with_units) and n_obs > 0

    return DiagnosticRecord(
        step_id="cohort_counts", status="pass" if ok else "fail",
        method="count_statistics per section; counts summed, rates weighted by n_obs",
        scope=f"{len(samples)} sections",
        observed={
            "n_sections": len(samples),
            "n_sections_with_units": len(with_units),
            "n_obs_total": n_obs,
            "n_vars": n_vars,
            "sparsity": round(1.0 - nnz / (n_obs * n_vars), 4) if n_obs and n_vars else None,
            "subject_id_column": subject or None,
            "per_section": rows[:12],
            "two_questions": "the verdict is existence, per the contract. the depth "
                             "statistics are observation and carry no threshold here.",
        },
        criterion="cohort counts present and non-zero",
        result=f"{len(with_units)}/{len(samples)} sections carry units, {n_obs} total",
        decision="proceed" if ok else
                 "a cohort with no units is not a cohort - nothing downstream can run",
        caveats=[] if subject else
                ["no subject_id_column declared - the patient axis is unbound"])


EDA_STEPS: list[Step] = [
    # THE ROOT. nine steps require the token it produces, and the contract's
    # authority_basis says what it is for: "no threshold. a cohort with no units
    # is not a cohort". that is an EXISTENCE question, not a quality one - you
    # cannot ask whether markers are present if there are no units to look in,
    # and you certainly can ask it of shallow data.
    #
    # one id has carried two questions since before this port:
    #
    #   the contract     field `patients`, rule present_and_positive. universal
    #                    authority, no threshold. written from the analysis, and
    #                    the recorded summary carries exactly its fields
    #   the code         measures.count_statistics - per-spot depth, detection,
    #                    sparsity. added later with measures/, wired under this
    #                    id because both words contain "count". the summary
    #                    records NONE of its fields
    #
    # both are answered here and neither is bent. the depth statistics are summed
    # across sections rather than measured on an object that does not exist;
    # the existence question is answered from what the sections already report.
    # whether the depth half deserves its own id - with cohort_calibrated
    # authority and a declared threshold, neither of which it has ever had - is a
    # contract decision for after the acceptance run, on evidence.
    Step(id="cohort_counts", produces=frozenset({"counts"}),
         fn=lambda c: _cohort_counts(c)),

    # the contract declares positive_markers and negative_markers as two
    # required checks with two fields. measures.marker_expression computes both
    # halves and returns ONE record, so each step passes only its own half - the
    # verdict is a conjunction over whichever groups are non-empty, so an empty
    # opposite half is a no-op rather than a silent pass.
    Step(id="positive_markers", requires=frozenset({"counts"}),
         produces=frozenset({"positive_markers"}),
         authority="cohort_calibrated",
         params_space=("positive", "min_pct_positive"),
         fn=_per_section("positive_markers", lambda a, c: measures.marker_expression(
             a, positive=_panel(c, "positive"), negative={},
             min_pct_positive=c["params"].get("min_pct_positive", 1.0)))),

    Step(id="negative_markers", requires=frozenset({"counts"}),
         produces=frozenset({"negative_markers"}),
         authority="cohort_calibrated",
         params_space=("negative", "max_pct_negative"),
         fn=_per_section("negative_markers", lambda a, c: measures.marker_expression(
             a, positive={}, negative=_panel(c, "negative"),
             max_pct_negative=c["params"].get("max_pct_negative", 10.0)))),

    Step(id="spatial_autocorrelation", requires=frozenset({"counts"}),
         produces=frozenset({"spatial_autocorrelation"}),
         authority="cohort_calibrated",
         # `markers` was missing from the declared space while the measure
         # requires it, so the step could never satisfy its own call. the list
         # names genes expected to be spatially structured in THIS tissue, so
         # like the marker panels it is the cohort's to declare.
         params_space=("markers", "k", "n_perm", "threshold"),
         fn=_per_section("spatial_autocorrelation",
                         lambda a, c: measures.spatial_autocorrelation(
                             a, markers=_panel(c, "markers"),
                             **{k: v for k, v in c["params"].items() if k != "markers"}),
                         applicable=_has_spatial,
                         why_not="no spatial coordinates in obsm - not a spatial assay")),

    Step(id="batch_structure", requires=frozenset({"counts"}),
         produces=frozenset({"batch_structure"}),
         authority="cohort_calibrated",   # contract: the absolute floor is calibrated
         applicable_when=_multi_sample,
         why_not_applicable="single sample - no batch axis to test",
         fn=lambda c: _guarded("batch_structure", lambda:
             measures.batch_structure(_pseudobulk(c), **c["params"]))),

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
         applicable_when=lambda c: any(_has_segmentation(_load(x), c)
                                       for x in _samples(c)),
         why_not_applicable="no segmentation masks - spot-resolution platform",
         fn=_unimplemented("segmentation_qc")),

    Step(id="multi_section_alignment", requires=frozenset({"counts"}),
         produces=frozenset({"multi_section_alignment"}),
         applicable_when=_multi_section,
         why_not_applicable="single section - nothing to align against",
         fn=_unimplemented("multi_section_alignment")),
]