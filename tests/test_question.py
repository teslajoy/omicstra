"""question-to-execution: NL question -> validate -> graph -> LCEL -> results.

exercises the full path with a controlled interpreter response (no live LLM
call). the LCEL calculations run on the real cohort data - mocking the analysis
would be insufficient.

the interpretation is a fixed dict, explicitly marked as such. a live
natural-language demonstration is a separate verification.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from omicstra.question import (
    AnalysisRequest,
    ValidationError,
    execute,
    interpret_and_execute,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/tnbc-92_v3"
NICHES = ROOT / "data/embeddings/niches_v3"
RUN = "R4_v3"


def _data_present() -> bool:
    return ((RUNS / RUN / "embeddings_test.parquet").is_file()
            and NICHES.is_dir())


# --- validation tests (no data needed) ------------------------------------

def test_validate_requires_project_id():
    with pytest.raises(ValidationError, match="no project_id"):
        validate({"task_id": "subject_identity_suppression", "run": "R4_v3"})


def test_validate_requires_task_id():
    with pytest.raises(ValidationError, match="no task_id"):
        validate({"run": "R4_v3"}, project_id="tnbc-92")


def test_validate_requires_run():
    with pytest.raises(ValidationError, match="no run"):
        validate({"task_id": "subject_identity_suppression"},
                 project_id="tnbc-92")


def test_validate_rejects_unknown_task_family():
    with pytest.raises(ValidationError, match="not a declared task family"):
        validate({"task_id": "imaginary_task", "run": "R4_v3"},
                 project_id="tnbc-92")


def test_validate_rejects_undeclared_run():
    with pytest.raises(ValidationError, match="not declared"):
        validate({"task_id": "subject_identity_suppression",
                  "run": "nonexistent_run"},
                 project_id="tnbc-92")


def test_validate_h2c_resolves_confounder_suppression_params():
    req = validate({
        "question": "how strong is the association?",
        "objective": "test H2-C",
        "task_id": "subject_identity_suppression",
        "run": "R4_v3",
    }, project_id="tnbc-92")
    assert req.hypothesis == "H2-C"
    assert req.task_id == "subject_identity_suppression"
    assert req.run == "R4_v3"
    assert "confounder_suppression" in req.resolved_params
    assert req.param_sources["confounder_suppression"] == "cohort.json#evaluate_params"
    cs = req.resolved_params["confounder_suppression"]
    assert cs["n_permutations"] == 100000
    assert cs["n_rho_null"] == 1000
    assert cs["seed"] == 42
    assert cs["alpha"] == 0.05
    assert cs["biology_labels"] == ["MC_global", "MC_tumor", "TIME"]
    assert req.missing == []


def test_validate_h2c_also_resolves_structure_and_its_null():
    """subject_identity_suppression requires both confounder_suppression
    AND structure_and_its_null (plus others). verify the full set."""
    req = validate({
        "question": "test",
        "objective": "test",
        "task_id": "subject_identity_suppression",
        "run": "R4_v3",
    }, project_id="tnbc-92")
    for step in ("confounder_suppression", "structure_and_its_null",
                 "label_granularity", "paired_identifiability",
                 "held_out_comparison", "hypothesis_verdict"):
        assert step in req.resolved_params, f"missing {step}"


def test_validate_h3_resolves_params():
    req = validate({
        "question": "do pathways transfer?",
        "objective": "test H3",
        "task_id": "pathway_transfer",
        "run": "R4_v3",
        "include_pathway_scores": True,
    }, project_id="tnbc-92")
    assert req.hypothesis == "H3"
    assert req.include_pathway_scores is True
    assert "association_and_specificity" in req.resolved_params


def test_validate_sets_interpretation_source():
    req = validate({
        "question": "test",
        "objective": "test",
        "task_id": "subject_identity_suppression",
        "run": "R4_v3",
        "interpretation_source": "claude_code",
        "interpretation_model": "claude-sonnet-4-5-20250514",
    }, project_id="tnbc-92")
    assert req.interpretation_source == "claude_code"
    assert req.interpretation_model == "claude-sonnet-4-5-20250514"


def test_validate_multiple_problems_reported_together():
    with pytest.raises(ValidationError) as exc_info:
        validate({})
    assert len(exc_info.value.problems) >= 3


# --- execution tests (require data) --------------------------------------

@pytest.mark.skipif(not _data_present(),
                    reason="run artifacts are git-ignored")
def test_h2c_question_to_execution():
    """the full H2-C path: validate -> graph -> LCEL -> results.

    uses a controlled interpretation (fixed dict), not a live LLM call.
    the LCEL chain runs on the real cohort arrays.
    """
    req = validate({
        "question": ("For this cohort and run R4_v3, how strong is the "
                     "association between the aligned embedding space and "
                     "the assigned biological labels, relative to patient "
                     "identity?"),
        "objective": ("measure whether alignment preserves biological "
                      "structure while suppressing patient identity"),
        "task_id": "subject_identity_suppression",
        "run": "R4_v3",
        "paired_view": "z_st",
        "interpretation_source": "test_fixture",
        "interpretation_model": "controlled_response",
    }, project_id="tnbc-92")

    result = execute(req)

    # execution completed
    assert result.report["status"] == "measured"
    assert result.report["run"] == "R4_v3"
    assert result.report["n_units"] > 0
    assert result.report["n_samples"] > 0

    # confounder_suppression step actually ran
    steps = result.report.get("steps", {})
    assert steps.get("confounder_suppression") == "pass", \
        f"confounder_suppression step: {steps.get('confounder_suppression')}"

    # H2-C verdict computed (separate from H2)
    verdicts = result.report.get("verdicts", {})
    assert "H2-C" in verdicts, f"verdicts keys: {sorted(verdicts)}"
    assert verdicts["H2-C"]["verdict"] in (
        "biological", "diagnostic", "unguarded", "not_run")
    # H2 verdict is separate and still mapped to structure_and_its_null
    assert "H2" in verdicts
    assert verdicts["H2"]["verdict"] in (
        "biological", "diagnostic", "unguarded", "not_run")

    # numerical findings: confounder_suppression has per_view data
    findings = result.findings()
    assert "confounder_suppression" in findings, \
        f"findings keys: {sorted(findings)}"
    cs = findings["confounder_suppression"]
    assert cs["status"] == "pass"
    assert "per_view" in cs, "findings must extract nested per_view data"

    # per_view structure: biology z/p/q, patient z/p, ratio/CI/G3
    pv = cs["per_view"]
    assert len(pv) > 0
    for view_name, vd in pv.items():
        assert "biology" in vd, f"view {view_name} missing biology"
        assert "patient" in vd, f"view {view_name} missing patient"
        assert "ratio" in vd, f"view {view_name} missing ratio"

        # patient: z and p_value
        assert "z" in vd["patient"]
        assert "p_value" in vd["patient"]
        assert isinstance(vd["patient"]["z"], (int, float))

        # biology: per label (MC_global, MC_tumor, TIME)
        for label in ("MC_global", "MC_tumor", "TIME"):
            if label in vd["biology"]:
                bl = vd["biology"][label]
                assert "z" in bl
                assert "p_value" in bl
                assert isinstance(bl["z"], (int, float))

        # ratio: per label with G3 guard and CI
        for label, rd in vd["ratio"].items():
            assert "ratio" in rd, f"ratio[{label}] missing ratio"
            assert "ratio_ci_low" in rd, f"ratio[{label}] missing ratio_ci_low"
            assert "ratio_ci_high" in rd, f"ratio[{label}] missing ratio_ci_high"
            assert "G3" in rd, f"ratio[{label}] missing G3 guard"
            assert "passed" in rd["G3"]

    # guards are present
    assert "guards" in cs
    guard_names = [g["guard"] for g in cs["guards"]]
    assert "G1_label_granularity" in guard_names
    assert "G3_bio_vs_confounder" in guard_names

    # provenance links question to execution
    prov = result.provenance()
    assert prov["question"].startswith("For this cohort")
    assert prov["task_id"] == "subject_identity_suppression"
    assert prov["run"] == "R4_v3"
    assert prov["interpretation_source"] == "test_fixture"
    assert prov["interpretation_model"] == "controlled_response"
    assert prov["run_id"].startswith("question-")

    # notebook: generated or explicitly failed
    if result.notebook_path:
        assert Path(result.notebook_path).suffix == ".ipynb"
    else:
        assert result.notebook_error is not None, \
            "notebook neither generated nor reported as failed"


@pytest.mark.skipif(not _data_present(),
                    reason="run artifacts are git-ignored")
def test_h2c_interpret_and_execute():
    """the CLI path: interpreter callable -> validate -> execute.

    the interpreter is a function returning a fixed dict. explicitly NOT a
    live LLM call - the test states this in its assertion.
    """
    def fixed_interpreter(question: str, context: dict) -> dict:
        return {
            "objective": "measure biology vs patient identity",
            "task_id": "subject_identity_suppression",
            "run": "R4_v3",
            "paired_view": "z_st",
            "interpretation_source": "test_fixture",
            "interpretation_model": "fixed_dict",
        }

    result = interpret_and_execute(
        "How strong is the biology signal relative to patient identity "
        "in run R4_v3?",
        interpreter=fixed_interpreter,
        project_id="tnbc-92",
    )

    assert result.report["status"] == "measured"
    assert result.request.interpretation_source == "test_fixture"
    assert result.request.interpretation_model == "fixed_dict"

    summary = result.summary()
    assert summary["hypothesis"] == "H2-C"
    assert "confounder_suppression" in summary["findings"]
    assert summary["findings"]["confounder_suppression"]["per_view"]
    assert summary["provenance"]["interpretation_source"] == "test_fixture"


@pytest.mark.skipif(not _data_present(),
                    reason="run artifacts are git-ignored")
def test_execution_result_matches_direct_evaluate():
    """the question path must agree with direct evaluate_measure on
    identical inputs and parameters."""
    from omicstra.graphs.evaluate import build_evaluate_graph
    from omicstra.graph import _load_cohort

    cohort = _load_cohort("tnbc-92")

    # direct path
    direct = build_evaluate_graph().invoke({
        "project_id": "tnbc-92",
        "arm": "evaluate",
        "cohort": cohort,
        "compute": True,
        "run": "R4_v3",
        "paired_view": "z_st",
    })

    # question path
    req = validate({
        "question": "biology vs patient identity for R4_v3",
        "objective": "H2-C",
        "task_id": "subject_identity_suppression",
        "run": "R4_v3",
        "paired_view": "z_st",
    }, project_id="tnbc-92")
    qresult = execute(req)

    # same verdicts
    d_verdicts = direct["report"].get("verdicts", {})
    q_verdicts = qresult.report.get("verdicts", {})
    for h in d_verdicts:
        if h in q_verdicts:
            assert d_verdicts[h]["verdict"] == q_verdicts[h]["verdict"], \
                f"verdict mismatch for {h}"

    # same step statuses
    d_steps = direct["report"].get("steps", {})
    q_steps = qresult.report.get("steps", {})
    for sid in d_steps:
        if sid in q_steps:
            assert d_steps[sid] == q_steps[sid], \
                f"step status mismatch for {sid}"

    # confounder_suppression findings match
    d_recs = {r["step_id"]: r for r in direct.get("records", [])
              if r.get("step_id")}
    q_recs = {r["step_id"]: r for r in qresult.records
              if r.get("step_id")}
    if "confounder_suppression" in d_recs and "confounder_suppression" in q_recs:
        d_obs = d_recs["confounder_suppression"].get("observed", {})
        q_obs = q_recs["confounder_suppression"].get("observed", {})
        d_pv = d_obs.get("per_view", {})
        q_pv = q_obs.get("per_view", {})
        for view in d_pv:
            if view not in q_pv:
                continue
            d_pt_z = d_pv[view].get("patient", {}).get("z")
            q_pt_z = q_pv[view].get("patient", {}).get("z")
            if d_pt_z is not None and q_pt_z is not None:
                assert d_pt_z == pytest.approx(q_pt_z), \
                    f"patient z mismatch in {view}"
            for label in d_pv[view].get("biology", {}):
                if label in q_pv[view].get("biology", {}):
                    d_bz = d_pv[view]["biology"][label].get("z")
                    q_bz = q_pv[view]["biology"][label].get("z")
                    if d_bz is not None and q_bz is not None:
                        assert d_bz == pytest.approx(q_bz), \
                            f"biology z mismatch for {label} in {view}"


# --- MCP tool tests -------------------------------------------------------

@pytest.mark.skipif(not _data_present(),
                    reason="run artifacts are git-ignored")
def test_mcp_ask_tool():
    """the MCP `ask` tool reaches the same execution path."""
    from omicstra.mcp.server import ask

    result = ask(
        question=("For R4_v3, how well does the aligned space preserve "
                  "biological labels vs patient identity?"),
        task_id="subject_identity_suppression",
        run="R4_v3",
        paired_view="z_st",
        objective="H2-C biology vs confounder",
        interpretation_model="test_fixture",
        project_id="tnbc-92",
    )

    assert result["status"] == "measured"
    assert result["hypothesis"] == "H2-C"
    assert "confounder_suppression" in result["findings"]
    cs = result["findings"]["confounder_suppression"]
    assert "per_view" in cs
    assert result["provenance"]["task_id"] == "subject_identity_suppression"
    assert result["provenance"]["interpretation_model"] == "test_fixture"


def test_mcp_ask_validation_failure():
    from omicstra.mcp.server import ask

    result = ask(
        question="test question",
        task_id="imaginary_task",
        run="R4_v3",
        project_id="tnbc-92",
    )
    assert result["status"] == "validation_failed"
    assert "problems" in result
    assert any("not a declared task family" in p for p in result["problems"])


# --- edge cases -----------------------------------------------------------

def test_no_prior_evidence_validates():
    """a supported analysis validates without prior target-specific routing
    evidence, when inputs and method are declared. this test validates only
    - execution requires data and is tested separately."""
    req = validate({
        "question": "test pathway transfer without prior evidence",
        "objective": "verify no-evidence path",
        "task_id": "pathway_transfer",
        "run": "R4_v3",
        "include_pathway_scores": True,
    }, project_id="tnbc-92")
    assert req.task_id == "pathway_transfer"
    assert req.missing == []