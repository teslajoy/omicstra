"""the guard library - the generalizable construct-validity IP.

each guard is a dataset-agnostic check; the tnbc-92 value is the example that
proved it fires, and no guard names a cohort. eval refuses to report a number
whose guard fails.

a PORT. the bodies are `scripts/eval.py` and `scripts/eval_alignment_biology.py`,
the code that produced the published grid, and those stay the oracles the way
`align.py` did. nothing here changes a formula.

why these six and not a general "check the statistics" pass: each one came from
a result that looked good and was not. that is the only reason a guard earns a
place, and it is why every docstring below names the failure it caught rather
than the property it checks.
"""
from __future__ import annotations

from pydantic import BaseModel


class GuardFailed(RuntimeError):
    """a reported number did not survive its guard.

    raised rather than returned where the guard's failure means the NUMBER is
    invalid rather than merely caveated - a supervision leak, or a label that
    encodes the confounder. a caller that wants the verdict without the raise
    calls the guard directly; the chain raises.
    """


class GuardResult(BaseModel):
    name: str
    passed: bool
    value: float | None = None
    note: str = ""
    observed: dict = {}

    def raise_if_failed(self) -> GuardResult:
        if not self.passed:
            raise GuardFailed(f"{self.name}: {self.note}")
        return self


# --- G1 ---------------------------------------------------------------------
def nmi_label_confounder(label, confounder, tau: float = 0.5) -> GuardResult:
    """NMI(candidate_label, subject) > tau -> the label carries the confounder.

    a clustering score against a label that encodes patient identity measures
    patient identity. on the seed cohort's niche join, archetype scores 0.645
    against patient_id - an archetype is assigned per patient, so every niche of
    a patient shares one - and "the aligned space recovers archetypes" is then
    largely "the aligned space recovers which patient a niche came from".
    mc_megacluster, a per-spot label, scores 0.437 and compartment 0.195.

    demote rather than discard: the label is still diagnostic, it just cannot
    carry a biological claim on its own.
    """
    import numpy as np
    from sklearn.metrics import normalized_mutual_info_score

    a = np.asarray(label).ravel()
    b = np.asarray(confounder).ravel()
    if len(a) != len(b):
        raise ValueError(f"label and confounder differ in length: {len(a)} vs {len(b)}")
    nmi = float(normalized_mutual_info_score(a, b))
    return GuardResult(
        name="G1_label_granularity", passed=nmi <= tau, value=nmi,
        observed={"tau": tau, "n": len(a),
                  "n_label_levels": len(np.unique(a)),
                  "n_confounder_levels": len(np.unique(b))},
        note=(f"NMI(label, confounder) = {nmi:.3f} <= {tau}" if nmi <= tau else
              f"NMI(label, confounder) = {nmi:.3f} > {tau} - the label encodes the "
              "confounder, so a clustering score against it measures the confounder. "
              "demote to diagnostic; it cannot carry a biological claim alone"))


# --- G2 ---------------------------------------------------------------------
def held_out_honesty(within_score: float, cross_score: float,
                     drop_tau: float = 0.5) -> GuardResult:
    """a score that survives within-subject and dies across it was the subject.

    on the seed cohort CCA scored 0.667 within-subarray and 0.000
    cross-subarray. the within number was not a weaker version of the cross
    number - it was a different quantity, and reporting it would have been
    reporting subject identity as biology.

    the cross-subject value is the only one that may be reported. this guard
    exists to make the WITHIN value's collapse visible rather than letting the
    two sit side by side as if comparable.
    """
    w, c = float(within_score), float(cross_score)
    drop = (w - c) / abs(w) if w else 0.0
    return GuardResult(
        name="G2_held_out_honesty", passed=drop <= drop_tau, value=drop,
        observed={"within": w, "cross": c, "drop_tau": drop_tau},
        note=(f"within {w:.3f} -> cross {c:.3f}, a {drop:.0%} drop" if drop <= drop_tau
              else f"within {w:.3f} -> cross {c:.3f}: a {drop:.0%} drop across the "
                   "confounder. the within-subject number is a different quantity, "
                   "not a stronger one - report the cross-subject value only"))


# --- G3 ---------------------------------------------------------------------
def bio_vs_confounder_ratio(bio_z: float, confounder_z: float,
                            floor: float | None = None) -> GuardResult:
    """below 1 the method amplifies the confounder more than the biology.

    the ratio is the claim, not the biology z alone: a method can score well on
    biology while scoring better on patient identity, and that is anti-helpful
    rather than merely weak. on the seed cohort one classical arm reached 0.071,
    which is why this is a contraindication and not a caution.

    `floor` is the raw-modality value when there is one. a method below the floor
    is worse than not aligning at all for that question.
    """
    b, c = float(bio_z), float(confounder_z)
    ratio = b / c if c else float("inf")
    passed = ratio >= 1.0 and (floor is None or ratio >= floor)
    why = []
    if ratio < 1.0:
        why.append(f"ratio {ratio:.3f} < 1 - the confounder is amplified more than "
                   "the biology, which is anti-helpful rather than weak")
    if floor is not None and ratio < floor:
        why.append(f"ratio {ratio:.3f} below the raw-modality floor {floor:.3f} - "
                   "worse than not aligning for this question")
    return GuardResult(
        name="G3_bio_vs_confounder", passed=passed, value=ratio,
        observed={"bio_z": b, "confounder_z": c, "floor": floor},
        note="; ".join(why) or f"ratio {ratio:.3f} clears 1"
             + (f" and the floor {floor:.3f}" if floor is not None else ""))


# --- G4 ---------------------------------------------------------------------
def specificity_matrix(axis_vectors, names=None, tau: float = 0.5) -> GuardResult:
    """a direction-magnitude claim ships with its off-diagonal cosine.

    "this axis encodes immune signal" is only a claim if the axis is distinct
    from the others. on the seed cohort the winning arm's pathway axes sat at
    0.97 mutual cosine - one direction wearing five labels - and the magnitudes
    alone read as five findings.

    aliasing is not automatically a defect: at coarse aggregation it was an
    artifact, and at DAG resolution the same arm separated real biology. so this
    reports the number and fails only past tau, rather than deciding what it
    means.
    """
    import numpy as np

    V = np.asarray(axis_vectors, dtype=np.float64)
    if V.ndim != 2 or V.shape[0] < 2:
        raise ValueError("specificity needs at least two axis vectors, as (k, dim)")
    n = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    C = np.abs(n @ n.T)
    np.fill_diagonal(C, 0.0)
    worst = float(C.max())
    i, j = np.unravel_index(int(C.argmax()), C.shape)
    lbl = (list(names) if names is not None else list(range(len(V))))
    return GuardResult(
        name="G4_specificity", passed=worst <= tau, value=worst,
        observed={"max_off_diagonal": worst, "mean_off_diagonal": float(C.sum() / (C.size - len(C))),
                  "worst_pair": [str(lbl[i]), str(lbl[j])], "tau": tau, "k": len(V)},
        note=(f"max off-diagonal cosine {worst:.3f} <= {tau}" if worst <= tau else
              f"max off-diagonal cosine {worst:.3f} between {lbl[i]!r} and {lbl[j]!r} - "
              "these axes are not distinct, so per-axis magnitudes are one direction "
              "wearing several labels. report the cosine beside every magnitude"))


# --- G5 ---------------------------------------------------------------------
def noise_floor(observed: float, null: object, tail: str = "greater") -> GuardResult:
    """a statistic is a finding only against its own permutation null.

    ported from `rho_with_null`: the EMPIRICAL p, (1 + #{null >= obs}) / (n + 1),
    not a normal approximation of the null - the published critique of this class
    of test is specifically about normal approximations, and the empirical form
    costs nothing once the permutations exist.

    z is reported beside p because at large n a p-value is significant for an
    effect far too small to matter, and the reverse at small n.
    """
    import numpy as np

    obs = float(observed)
    nul = np.asarray(null, dtype=np.float64).ravel()
    if nul.size < 20:
        raise ValueError(f"a permutation null needs samples; got {nul.size}")
    extreme = (nul >= obs) if tail == "greater" else (nul <= obs)
    p = float((extreme.sum() + 1) / (nul.size + 1))
    sd = float(nul.std())
    z = (obs - float(nul.mean())) / sd if sd > 0 else float("nan")
    return GuardResult(
        name="G5_noise_floor", passed=p < 0.05, value=p,
        observed={"observed": obs, "null_mean": float(nul.mean()), "null_sd": sd,
                  "z": z, "n_permutations": nul.size, "tail": tail,
                  "p_form": "empirical: (1 + #{null beyond obs}) / (n + 1)"},
        note=(f"p = {p:.2g}, z = {z:.2f} over {nul.size} permutations" if p < 0.05 else
              f"p = {p:.2g} over {nul.size} permutations - indistinguishable from its "
              "own null, so there is no effect to promote"))


# --- G6 ---------------------------------------------------------------------
def circular_supervision(feature_columns, supervision: str) -> GuardResult:
    """the supervision signal must not appear in the input vector.

    the implementation lives in `protocols/join.py`, because the join is where
    features and supervision sit in one table and therefore where the assertion
    has to bite. this wraps it so the guard library is complete and a caller
    reading the six does not have to know that one of them lives elsewhere.
    """
    from omicstra.protocols.join import (
        CircularSupervision,
        assert_no_circular_supervision,
    )

    try:
        assert_no_circular_supervision(feature_columns, supervision)
    except CircularSupervision as e:
        return GuardResult(name="G6_circular_supervision", passed=False, note=str(e),
                           observed={"features": sorted(feature_columns),
                                     "supervision": supervision})
    return GuardResult(
        name="G6_circular_supervision", passed=True,
        observed={"features": sorted(feature_columns), "supervision": supervision},
        note=f"{supervision!r} is not among the input features")


GUARDS = {
    "G1": nmi_label_confounder,
    "G2": held_out_honesty,
    "G3": bio_vs_confounder_ratio,
    "G4": specificity_matrix,
    "G5": noise_floor,
    "G6": circular_supervision,
}


# --- the chain: guards attached to each hypothesis --------------------------
#
# a guard runs only on inputs that exist. the H1/H2/H3 rollups carry some of the
# inputs the six guards need and not others, and a guard "applied" to a number
# it was never given is a fabricated verdict. so each hypothesis reports three
# kinds of result: guards that RAN, and guards that are not_run with the input
# that is missing named - the same not_run / not_applicable discipline the EDA
# gate uses.

def _not_run(name: str, missing: str) -> dict:
    return {"guard": name, "status": "not_run",
            "note": f"needs {missing}, which this rollup does not carry"}


def gate_h2(summary: dict, label_nmi: float, label: str,
            confounder: str = "patient_id", tau: float = 0.5) -> dict:
    """H2 scores clustering against a label, so the label is gated before the score.

    the rollup's ARI and silhouette - its raw_reference included - are only a
    biological claim if the label they are scored against does not encode the
    confounder. `label_nmi` is measured on the cohort's units, not taken from the
    summary, because the summary does not record it.
    """
    g1 = GuardResult(
        name="G1_label_granularity", passed=label_nmi <= tau, value=label_nmi,
        observed={"label": label, "confounder": confounder, "tau": tau},
        note=(f"{label} NMI vs {confounder} = {label_nmi:.3f}"
              + ("" if label_nmi <= tau else
                 " - the label encodes the confounder, so every clustering score in "
                 "this rollup is diagnostic and cannot carry a biological claim alone")))
    return {
        "hypothesis": "H2",
        "scored_against": label,
        "claims_biology": g1.passed,
        "n_runs": len(summary.get("per_run", {})),
        "guards": [g1.model_dump(),
                   _not_run("G2_held_out_honesty", "a within-subject score to compare"),
                   _not_run("G5_noise_floor", "a permutation null over ARI")],
        "reference": summary.get("raw_reference"),
        "reference_status": "biological" if g1.passed else "diagnostic - same label as the scores",
    }


def gate_h3(summary: dict, test_patients: int, min_patients: int = 5) -> dict:
    """H3 is cross-patient by construction, and that is checked rather than assumed.

    the patient probe in the rollup is reported as a CONFOUNDER STRENGTH, not a
    pass/fail: it says how much of the aligned space is patient identity, which is
    the denominator a biology claim has to beat (G3). it is not itself a verdict
    on the pathway signal.
    """
    per = summary.get("per_run", {})
    n_pat = {r.get("n_patients") for r in per.values()}
    split_ok = n_pat == {test_patients} and test_patients >= min_patients
    probe = {rid: (r.get("patient_probe_z_mean") or {}).get("accuracy")
             for rid, r in per.items()}
    return {
        "hypothesis": "H3",
        "cross_patient": split_ok,
        "test_patients": sorted(p for p in n_pat if p is not None),
        "n_runs": len(per),
        "patient_identity_in_aligned_space": probe,
        "guards": [
            {"guard": "G2_held_out_honesty", "status": "pass" if split_ok else "fail",
             "note": (f"every run evaluated on the same {test_patients} held-out patients"
                      if split_ok else f"patient counts across runs: {sorted(n_pat, key=str)}")},
            _not_run("G4_specificity", "per-pathway axis vectors"),
            _not_run("G5_noise_floor", "the permutation null over CCA"),
        ],
    }


def gate_h2c(ci: dict, floor: float | None = None, floor_source: str | None = None) -> dict:
    """H2-C on the artifact the pack cites: the bio/patient ratio per run.

    reads `metrics_h2_ci.json -> h2c` - per run, the TIME-signature z and the
    patient-identity z from biology.parquet, their ratio and a delta-method
    interval. G3 is applied to exactly those two z's.

    G1 is not re-run and not skipped: TIME is assigned per sample, which is the
    label-granularity hazard, and the biology test is built around it - only
    cross-patient pairs are scored, and the null permutes the label across
    patients rather than across niches. that is recorded as satisfied by design,
    with the reason, so it reads differently from a guard that was never checked.

    `floor` is the raw-modality ratio. on the seed cohort it is cited from the
    results summary and exists in no machine-readable artifact, so the caller
    passes it with its source or not at all.
    """
    per = ci.get("h2c", {})
    runs = {}
    for rid, r in per.items():
        g3 = bio_vs_confounder_ratio(r["z_TIME"], r["z_patient"], floor)
        runs[rid] = {"ratio": r["ratio"], "ci": [r["ratio_ci_low"], r["ratio_ci_high"]],
                     "z_bio": r["z_TIME"], "z_confounder": r["z_patient"],
                     "G3": g3.model_dump()}
    ranking = sorted(runs, key=lambda k: runs[k]["ratio"], reverse=True)
    return {
        "hypothesis": "H2-C",
        "scored_from": "metrics_h2_ci.json -> h2c",
        "n_runs": len(runs),
        "ranking": ranking,
        "runs": runs,
        "floor": {"value": floor, "source": floor_source} if floor is not None else None,
        "guards": [
            {"guard": "G1_label_granularity", "status": "satisfied_by_design",
             "note": "TIME is per-sample; the test scores cross-patient pairs only and "
                     "permutes the label at patient level"},
            {"guard": "G3_bio_vs_confounder", "status": "per_run",
             "note": f"{sum(v['G3']['passed'] for v in runs.values())} of {len(runs)} runs "
                     "clear a ratio of 1"},
            _not_run("G5_noise_floor", "the permutation nulls behind each z "
                     "(biology_nulls.parquet, per run)"),
        ],
    }


def gate_h3_pathway(rows: list[dict], st_features: dict[str, list[str]],
                    view: str = "z_he", alpha: float = 0.05,
                    pathway_feature: str = "gpath2vec_niche") -> dict:
    """H3 on the artifact the pack cites: per-pathway CCA on held-out patients.

    `rows` are per_pathway_cca.parquet records. per run, in the headline view:
    how many TESTABLE pathways survive BH-FDR on the held-out-patient null
    (option A), which pathways were excluded as untestable, and three guards.

      G2  every run scored on the same held-out split - n_test identical
      G5  significance is the script's empirical permutation p, BH-corrected;
          read as recorded. the null samples are pinned by sha, not re-derived
      G6  a view is circular when the run's ST input carries the pathway
          embedding. derived here from each run's st_features, NOT from the
          artifact's view_clean column, and any disagreement is reported: the
          script hardcodes the circular runs, and a run added to the grid after
          it was written is silently marked clean
    """
    import math

    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)

    n_test = {r["n_test"] for r in rows if r["view"] == view}
    runs, disagreements = {}, []
    for rid, rs in sorted(by_run.items()):
        carries = pathway_feature in (st_features.get(rid) or [])
        for r in rs:
            circular = carries and r["view"] != "z_he"
            if bool(r["view_clean"]) == circular:
                disagreements.append({"run": rid, "view": r["view"],
                                      "artifact_view_clean": bool(r["view_clean"]),
                                      "derived_circular": circular})
        head = [r for r in rs if r["view"] == view]
        testable = [r for r in head if r["testable"]]
        sig = [r["pathway_name"] for r in testable
               if r["fdr_A"] is not None and not math.isnan(r["fdr_A"]) and r["fdr_A"] < alpha]
        runs[rid] = {"n_significant": len(sig), "n_testable": len(testable),
                     "significant": sig,
                     "z": {r["pathway_name"]: r["z_A"] for r in testable},
                     "excluded_untestable": [r["pathway_name"] for r in head if not r["testable"]],
                     "view_circular": carries and view != "z_he"}
    # dedupe: one disagreement per (run, view), not per pathway row
    seen, uniq = set(), []
    for d in disagreements:
        if (d["run"], d["view"]) not in seen:
            seen.add((d["run"], d["view"]))
            uniq.append(d)
    split_ok = len(n_test) == 1
    return {
        "hypothesis": "H3",
        "scored_from": "per_pathway_cca.parquet (option A: held-out-patient null)",
        "view": view, "alpha": alpha,
        "n_runs": len(runs),
        "runs": runs,
        "guards": [
            {"guard": "G2_held_out_honesty", "status": "pass" if split_ok else "fail",
             "note": (f"every run's {view} view scored on the same {next(iter(n_test))} held-out niches"
                      if split_ok else f"held-out sizes differ across runs: {sorted(n_test)}")},
            {"guard": "G5_noise_floor", "status": "as_recorded",
             "note": "empirical permutation p per pathway, BH-FDR across the testable family, "
                     "read from the artifact"},
            {"guard": "G6_circular_supervision",
             "status": "pass" if not uniq else "flag_disagreement",
             "note": ("the artifact's view_clean agrees with st_features on every run" if not uniq
                      else f"{len(uniq)} run/view pair(s) where view_clean disagrees with the "
                           f"run's st_features; the headline {view} view is "
                           + ("affected" if any(d['view'] == view for d in uniq) else "not affected")),
             "disagreements": uniq},
            _not_run("G4_specificity", "per-pathway axis vectors (pathway_axes.parquet)"),
        ],
    }


def coverage(summary: dict, declared_grid: list[str]) -> dict:
    """which runs of the declared grid a rollup actually scored.

    the three rollups each drop a different run, and nothing downstream noticed
    until they were pinned side by side. a missing run is a gap in a comparison,
    so it is reported with the rollup rather than discovered from it.
    """
    have = set(summary.get("per_run", {}))
    return {"n_declared": len(declared_grid), "n_scored": len(have & set(declared_grid)),
            "missing": sorted(set(declared_grid) - have),
            "unexpected": sorted(have - set(declared_grid))}
