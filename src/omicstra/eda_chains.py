"""EDA steps as LCEL chains, with applicability resolved from the data.

three things this adds over a bare `name -> fn` registry:

  links       a step is a CHAIN. each link is its own traced span, so the params
              that decide a result are visible instead of buried in a body. this
              is where `n_perm` lives, and why 1000-vs-500 was invisible before.

  requires    an ASSERTION, not a search space. the analytical order is standard
              bioinformatics - moran's I after the weights, FDR after the null -
              so nothing may reorder it. `requires` exists to REFUSE an invalid
              order, never to discover a valid one.

  applicable_when
              a predicate over data properties. resolution, not planning: a
              cohort with one section cannot be judged on multi-section
              alignment. that returns `not_applicable`, which is a real status
              distinct from null and from not_run - the difference between
              "we looked and it does not apply" and "nobody looked".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from langchain_core.runnables import Runnable, RunnableLambda

from omicstra import eda_steps
from omicstra.records import DiagnosticRecord

Authority = Literal["universal", "cohort_calibrated", "advisory"]


# --- data properties: what applicability is resolved against -----------------
@dataclass(frozen=True)
class DataFacts:
    """measured off the object, never declared by a caller."""
    n_obs: int
    n_vars: int
    n_samples: int
    has_spatial: bool
    has_annotations: bool
    has_segmentation: bool
    n_sections: int
    counts_are_integer: bool

    @classmethod
    def measure(cls, adata, sample_key: str | None = None) -> "DataFacts":
        import numpy as np
        x = adata.X[:200]
        x = x.toarray() if hasattr(x, "toarray") else x
        obs = adata.obs
        return cls(
            n_obs=adata.n_obs, n_vars=adata.n_vars,
            n_samples=int(obs[sample_key].nunique()) if sample_key in obs else 1,
            has_spatial="spatial" in adata.obsm,
            has_annotations=any(c in obs for c in ("compartment", "annotation", "region")),
            has_segmentation=any(c in obs for c in ("cell_id", "segmentation", "nucleus_id")),
            n_sections=int(obs["section"].nunique()) if "section" in obs else 1,
            counts_are_integer=bool(np.allclose(x, np.round(x))),
        )


@dataclass
class Step:
    chain: Runnable
    produces: str
    requires: list[str] = field(default_factory=list)
    applicable_when: Callable[[DataFacts], bool] = lambda d: True
    why_not_applicable: str = "not applicable to this cohort"
    authority: Authority = "universal"


STEPS: dict[str, Step] = {}


def register_step(name: str, chain: Runnable, *, produces: str, requires=(),
                  applicable_when=lambda d: True, why_not_applicable="not applicable",
                  authority: Authority = "universal") -> None:
    STEPS[name] = Step(chain, produces, list(requires), applicable_when,
                       why_not_applicable, authority)


# --- link helper: names the span so the trace shows the parameter ------------
def link(name: str, fn: Callable) -> Runnable:
    return RunnableLambda(fn).with_config(run_name=name)


# --- cohort_counts -----------------------------------------------------------
counts_chain = link(
    "count_statistics",
    lambda s: eda_steps.count_statistics(
        s["adata"], sample_key=s["params"].get("sample_key"),
        source=s["params"].get("source")),
)

# --- marker expression -------------------------------------------------------
markers_chain = link(
    "marker_expression",
    lambda s: eda_steps.marker_expression(
        s["adata"], positive=s["params"]["positive"], negative=s["params"]["negative"],
        min_pct_positive=s["params"].get("min_pct_positive", 1.0),
        max_pct_negative=s["params"].get("max_pct_negative", 10.0)),
)

# --- spatial autocorrelation: the chain whose links carry the real params ----
# knn_weights(k) -> morans_i -> permutation null(n_perm) -> threshold
# n_perm sets the p-value floor. as a bound link it is in the trace and in the
# record; as a module constant it was 1000 in some scripts and 500 in others.
spatial_chain = link(
    "spatial_autocorrelation",
    lambda s: eda_steps.spatial_autocorrelation(
        s["adata"], markers=s["params"]["markers"],
        sample_key=s["params"].get("sample_key"),
        k=s["params"].get("k", 6),
        n_perm=s["params"].get("n_perm", 999),
        threshold=s["params"].get("threshold", 0.3)),
)

# --- batch structure ---------------------------------------------------------
batch_chain = link(
    "batch_structure",
    lambda s: eda_steps.batch_structure(
        s["adata"], sample_key=s["params"]["sample_key"],
        sample_meta=s["params"]["sample_meta"],
        technical=s["params"]["technical"], biological=s["params"]["biological"],
        n_pcs=s["params"].get("n_pcs", 10)),
)

register_step("cohort_counts", counts_chain, produces="counts")
register_step("positive_markers", markers_chain, produces="marker_expression",
              requires=["counts"])
register_step("spatial_autocorrelation", spatial_chain, produces="spatial_autocorrelation",
              requires=["counts"], authority="cohort_calibrated",
              applicable_when=lambda d: d.has_spatial,
              why_not_applicable="no spatial coordinates in obsm")
register_step("batch_structure", batch_chain, produces="batch_structure",
              requires=["counts"],
              applicable_when=lambda d: d.n_samples > 1,
              why_not_applicable="single sample - no batch axis to test")

# the two checks that currently sit NULL under a proceed verdict. declaring
# applicability makes their absence a measured statement rather than a gap.
register_step("segmentation_qc",
              link("segmentation_qc", lambda s: eda_steps_stub("segmentation_qc")),
              produces="segmentation_qc", requires=["counts"],
              applicable_when=lambda d: d.has_segmentation,
              why_not_applicable="no segmentation masks - spot-resolution platform")
register_step("multi_section_alignment",
              link("multi_section_alignment", lambda s: eda_steps_stub("multi_section_alignment")),
              produces="multi_section_alignment", requires=["counts"],
              applicable_when=lambda d: d.n_sections > 1,
              why_not_applicable="single section - nothing to align against")


def eda_steps_stub(name: str) -> DiagnosticRecord:
    return DiagnosticRecord(step_id=name, status="not_run",
                            result="no implementation registered",
                            decision="implement or mark not applicable")


# --- the driver: applicability first, then order, then run -------------------
def run_steps(adata, params: dict, facts: DataFacts, only: list[str] | None = None
              ) -> list[DiagnosticRecord]:
    names = only or list(STEPS)
    produced: set[str] = set()
    out: list[DiagnosticRecord] = []

    for name in names:
        s = STEPS.get(name)
        if s is None:
            out.append(DiagnosticRecord(step_id=name, status="not_run",
                       result="no step registered under this name",
                       decision="register it or drop it from the contract"))
            continue

        # 1 · applicability - resolution from data, deterministic, no model
        if not s.applicable_when(facts):
            out.append(DiagnosticRecord(step_id=name, status="not_applicable",
                       method="applicability predicate over measured data facts",
                       result=s.why_not_applicable,
                       decision="not judged - this check does not apply to this cohort"))
            produced.add(s.produces)      # satisfied, so dependents may still run
            continue

        # 2 · order - an assertion. the chain is the method; nothing reorders it.
        if missing := [r for r in s.requires if r not in produced]:
            out.append(DiagnosticRecord(step_id=name, status="error",
                       result=f"requires {missing}, not yet produced",
                       decision="invalid order - the analytical chain is fixed"))
            continue

        if name not in params:
            out.append(DiagnosticRecord(step_id=name, status="not_run",
                       result="cohort supplied no params for this step",
                       decision="a step with no params is skipped, never guessed at"))
            continue

        # 3 · run
        try:
            rec = s.chain.invoke({"adata": adata, "params": params[name]},
                                 config={"run_name": f"eda:{name}",
                                         "tags": [s.authority],
                                         "metadata": {"produces": s.produces}})
            out.append(rec)
            produced.add(s.produces)
        except Exception as e:
            out.append(DiagnosticRecord(step_id=name, status="error",
                       result=f"{type(e).__name__}: {e}",
                       decision="step failed; the gate cannot judge this check"))
    return out