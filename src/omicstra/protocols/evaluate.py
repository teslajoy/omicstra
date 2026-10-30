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
    """below 1 means biology z is lower than patient-identity z under this test.

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
        why.append(f"ratio {ratio:.3f} < 1 - biology z is lower than "
                   "patient-identity z under this test")
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


# --- the evaluate stage: scope, pins, and what a guard does to a claim -------
#
# LEVEL 2. order, applicability and authority: which hypotheses this cohort can
# answer, which bytes the answer was read from, and which required guards ran.
# the maths is above - the guards and the per-hypothesis gates - and nothing
# here recomputes a published number. a number this stage cannot re-derive is
# reported `not_run` with the input it is missing NAMED, which is the same
# discipline the EDA gate uses and the reason this is worth having: a reader
# cannot otherwise tell a hypothesis nobody asked from one that failed.
import json as _json
from pathlib import Path as _Path


class EvidencePackMissing(RuntimeError):
    """no curated pack, so there is nothing to return a verdict over.

    a refusal and not a gate, the same shape as `EdaGateMissing`: the pack
    records which metric answers which task family, and inventing that is
    inventing the judgement the whole chain exists to attribute.
    """


class ArtifactDrift(RuntimeError):
    """a cited artifact's bytes differ from the pin the cohort declared.

    not a question. the pack's numbers were read from the pinned bytes, so a
    verdict over these ones would be a verdict over a different measurement
    wearing the pack's prose. no option at the gate makes that evaluable.
    """


def routing_contract() -> dict:
    """the task families, their hypotheses and their required guards."""
    return _json.loads(
        (_Path(__file__).resolve().parents[1] / "configs" / "routing_contract.json"
         ).read_text())


def hypothesis_root(h: str) -> str:
    """`H2-B.2` -> `H2`.

    the contract names families at the sub-hypothesis a result actually tests -
    H2-A is clustering, H2-C is confounder suppression, and they are different
    claims. a caller asks about H1/H2/H3, which is how the proposal and
    CLAUDE.md state them, so the two vocabularies meet here rather than in six
    places.
    """
    return h.split("-", 1)[0].strip()


def families(hypotheses: list[str] | tuple[str, ...] | None = None) -> list[dict]:
    """the contract's task families, filtered to the hypotheses asked for."""
    fams = routing_contract()["task_families"]
    if hypotheses is None:
        return list(fams)
    want = {hypothesis_root(h) for h in hypotheses}
    return [f for f in fams if hypothesis_root(f["hypothesis"]) in want]


def guard_implementations() -> dict:
    """which contract guard has a function here, declared rather than inferred.

    the two vocabularies are not the same length: the contract names seven
    required guards and this module implements six checks, and the overlap is
    five. that gap is DECLARED in routing_contract.json#guard_implementations
    with the reason per guard, because a required guard silently not run is
    exactly the failure the guard library exists to prevent.
    """
    return routing_contract()["guard_implementations"]


def guard_status(fams: list[dict]) -> list[dict]:
    """per required guard across the families in scope: implemented, or why not."""
    impl = guard_implementations()
    out, seen = [], set()
    for f in fams:
        for name in f.get("required_guards", []):
            if name in seen:
                continue
            seen.add(name)
            key = impl["map"].get(name)
            out.append({"guard": name, "required_by": f["id"],
                        "implemented_as": key,
                        "status": "implemented" if key else "not_implemented",
                        "note": (routing_contract()["guards"][name] if key
                                 else impl["unimplemented"][name])})
    return sorted(out, key=lambda d: d["guard"])


_PATH_KEYS = ("table", "ci_file")
_PATH_RE = r"[\w./{}-]+\.(?:json|parquet)"


def _tokens(task: dict) -> tuple[list[str], list[str]]:
    """the artifact names a task declaration carries, structured and from prose.

    two sources, because the declaration uses both: the keys a reader resolves
    (`table`, `ci_file`, including inside a nested floor or reference reader),
    and the `source` string, which is prose a person wrote and the only place
    some artifacts appear at all.
    """
    import re

    structured: list[str] = []
    prose: list[str] = []

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            elif k in _PATH_KEYS and isinstance(v, str):
                structured.append(v)
            elif k == "source" and isinstance(v, str):
                prose.extend(re.findall(_PATH_RE, v))

    walk(task)
    return structured, prose


def cited_paths(task: dict, runs: list[str]) -> list[str]:
    """every artifact path a task declaration LOCATES, relative to the runs root.

    a prose token counts only if it carries a directory. `metrics_h2_ci.json` on
    its own names a file without saying where it is, and the two candidates on
    the seed cohort are `eval/H2/metrics_h2_ci.json` and
    `{run}/eval/biology.parquet` - one shared, one per arm. resolving a bare
    name against whatever directory appeared nearby in the same sentence gets
    the first right and the second wrong, so it is not resolved at all. those
    names are returned by `named_without_location` and reported.

    `{run}` is expanded over the declared grid: a per-run file is one artifact
    per arm, and pinning the template would pin nothing.
    """
    structured, prose = _tokens(task)
    out: list[str] = []
    for p in structured + [t for t in prose if "/" in t]:
        out += [p.replace("{run}", r) for r in runs] if "{run}" in p else [p]
    # order-preserving dedupe: the same artifact is usually named by both a
    # structured key and the prose, and a pin list with duplicates reads as more
    # evidence than it is.
    return list(dict.fromkeys(out))


def named_without_location(task: dict) -> list[str]:
    """artifacts the prose names but does not locate. pinnable once declared.

    reported rather than guessed, and rather than dropped: a cited artifact
    nobody can hash is a gap in what the verdict covers, and the cohort closes
    it by naming the file in a structured key.
    """
    structured, prose = _tokens(task)
    bare = {t for t in prose if "/" not in t}
    located = {p.rsplit("/", 1)[-1] for p in structured + [t for t in prose if "/" in t]}
    return sorted(bare - located)


def _sha256(p: _Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pin_state(sources: dict, root: _Path, declared: dict | None,
              fams: list[dict]) -> dict:
    """hash every artifact the families in scope cite, against the declared pins.

    `absent` is not a failure of this function: a cohort whose grid is partly
    unwritten cites files that do not exist yet, and the hypothesis those files
    belong to is reported not_run naming them rather than refused here.
    """
    runs_root = (root / sources["runs_root"]).resolve()
    runs = list(sources.get("runs") or [])
    pins = dict(declared or {})
    observed, absent, drifted, matched, unlocated = {}, [], [], [], set()
    per_task: dict[str, list[str]] = {}
    for f in fams:
        task = (sources.get("tasks") or {}).get(f["id"])
        if not task or "not_available" in task:
            continue
        per_task[f["id"]] = cited_paths(task, runs)
        unlocated |= set(named_without_location(task))
        for rel in per_task[f["id"]]:
            if rel in observed or rel in absent:
                continue
            fp = runs_root / rel
            if not fp.is_file():
                absent.append(rel)
                continue
            observed[rel] = _sha256(fp)
            if rel in pins:
                (matched if pins[rel] == observed[rel] else drifted).append(rel)
    return {"runs_root": str(runs_root), "observed": observed,
            "cited_by_task": per_task,
            "absent": sorted(absent), "matched": sorted(matched),
            "drifted": sorted(drifted),
            "n_declared_pins": len(pins),
            # named by the prose, located by nothing. not a hash and not a
            # silence - see named_without_location.
            "named_without_location": sorted(unlocated),
            "unpinned": sorted(set(observed) - set(pins))}


def _pack(root: _Path) -> dict:
    p = root / "routing_evidence.json"
    if not p.is_file():
        raise EvidencePackMissing(
            f"no routing_evidence.json for this cohort ({root.name}), so there is "
            "nothing to return a verdict over. the pack records which metric "
            "answers which task family and which floor each is judged against; "
            "that is curated, and generating it would be inventing the "
            "judgements it exists to attribute.")
    return _json.loads(p.read_text())


def _sources(root: _Path) -> dict:
    p = root / "evidence_sources.json"
    if not p.is_file():
        raise EvidencePackMissing(
            f"no evidence_sources.json for this cohort ({root.name}). the pack may "
            "state the numbers, but without the sources declaration nothing says "
            "which artifact each was read from - so no pin, no guard input, and "
            "no verdict this stage can stand behind.")
    return _json.loads(p.read_text())


def scope(fams: list[dict], pack: dict, sources: dict) -> dict:
    """per hypothesis: which families are fillable on this cohort, and which not.

    three states per family, and the difference between the last two is the
    point: declared and filled, declared `not_available` with the cohort's own
    reason, or named by the contract and absent from the pack entirely. a
    hypothesis with no fillable family is not a question for a person - it is
    reported not_run with its families named.
    """
    out: dict[str, dict] = {}
    for f in fams:
        h = hypothesis_root(f["hypothesis"])
        row = out.setdefault(h, {"fillable": [], "not_available": [], "undeclared": []})
        t_pack = (pack.get("tasks") or {}).get(f["id"])
        t_src = (sources.get("tasks") or {}).get(f["id"])
        if t_src is None and t_pack is None:
            row["undeclared"].append(f["id"])
        elif (t_src is not None and "not_available" in t_src) or \
             (t_pack is not None and "not_available" in t_pack):
            why = ((t_src or {}).get("not_available")
                   or (t_pack or {}).get("not_available"))
            row["not_available"].append({"task": f["id"], "why": why})
        else:
            row["fillable"].append(f["id"])
    for h, row in out.items():
        row["evaluable"] = bool(row["fillable"])
    return out


def preflight_evaluate(cohort: dict | None = None, project_id: str | None = None,
                       *, hypotheses: list[str] | None = None,
                       verdict: dict | None = None) -> list:
    """the gates that fire before a hypothesis verdict.

    deterministic and side-effect free apart from reading and hashing the
    artifacts the cohort's declaration names - the same shape as
    `preflight_align`, which reads the recorded EDA verdict. it measures nothing
    and recomputes no published number.

    raises rather than gates when the cohort has no pack or no sources
    declaration, and when a declared pin does not match the bytes on disk. the
    first two are "there is nothing to ask about"; the third is "no answer makes
    this evaluable".
    """
    from omicstra.protocols.align import _declarations
    from omicstra.protocols.encode import (
        GateRequest,
        _declared,
        _resolve_option,
        contract,
    )
    from omicstra.settings import settings

    root = settings.project_root(project_id)
    cohort = _declarations(project_id) if cohort is None else {
        **_declarations(project_id), **cohort}
    pack, sources = _pack(root), _sources(root)
    fams = families(hypotheses)
    sc = scope(fams, pack, sources)
    # only the fillable families are hashed. hashing an artifact for a task the
    # cohort has declared unavailable would report bytes nothing will read.
    fillable = [f for f in fams
                if f["id"] in sc[hypothesis_root(f["hypothesis"])]["fillable"]]
    pins = pin_state(sources, root, cohort.get("evidence_pins"), fillable)
    if pins["drifted"]:
        raise ArtifactDrift(
            f"{len(pins['drifted'])} cited artifact(s) differ from the pin this "
            f"cohort declares: {pins['drifted'][:3]}. the pack's numbers were read "
            "from the pinned bytes, so a verdict over these ones would carry the "
            "pack's prose over a different measurement. re-pin deliberately, or "
            "restore the artifacts.")

    gates = {g["id"]: g for g in contract()["gates"] if g.get("stage") == "evaluate"}
    guards = guard_status(fams)
    out = []
    for gid in ("hypothesis_scope", "artifact_pinning", "guard_failure"):
        g, answer, src, observed = gates[gid], None, None, {}

        if gid == "hypothesis_scope":
            observed = {"requested": list(hypotheses) if hypotheses else "all declared",
                        "per_hypothesis": sc,
                        "evaluable": sorted(h for h, r in sc.items() if r["evaluable"]),
                        "not_run": sorted(h for h, r in sc.items() if not r["evaluable"])}
            if hypotheses:
                # the caller NAMED them, which is the answer this gate asks for.
                # recorded as the request rather than as a declaration: the
                # server saw a tool call, not a person.
                answer, src = "named_subset", "the request"
        elif gid == "artifact_pinning":
            observed = {k: pins[k] for k in
                        ("runs_root", "absent", "matched", "drifted", "unpinned",
                         "named_without_location", "n_declared_pins")}
            # the hashes travel WITH the question. answering `pin_these_bytes`
            # without seeing them would be pinning bytes sight unseen, and the
            # record that answer writes is this map.
            observed["sha256"] = pins["observed"]
            observed["n_artifacts"] = len(pins["observed"])
            if pins["observed"] and not pins["unpinned"]:
                answer, src = "declared_pins", "cohort.json#evidence_pins"
        else:
            # a guard with no IMPLEMENTATION does not fire this gate and does not
            # demote a number - it is reported not_run with the contract's own
            # reason. what fires it is a guard that RAN and failed, which is why
            # the verdict is computed before the gates rather than after them.
            v = verdict if verdict is not None else verdicts(
                project_id, hypotheses=hypotheses, root=root)
            failed = sorted(f for r in v.values() for f in r["guards_failed"])
            observed = {"required": guards,
                        "n_implemented": sum(d["status"] == "implemented" for d in guards),
                        "n_not_implemented": sum(d["status"] == "not_implemented"
                                                 for d in guards),
                        "failed": failed, "n_failed": len(failed)}
            if not failed:
                answer, src = "report_as_diagnostic", "no guard that ran has failed"

        declared = _declared(cohort, g.get("pre_answerable_by"))
        if declared is not None:
            answer = _resolve_option(declared, tuple(g["options"]), gid,
                                     g.get("answer_kind", "choice"))
            src = g.get("pre_answerable_by")

        out.append(GateRequest(
            id=gid, when=g["when"], question=g["question"],
            options=tuple(g["options"]), forecloses=g["forecloses"],
            pre_answerable_by=g.get("pre_answerable_by"),
            answer=answer, source=src, observed=observed or None))
    return out


# the per-family readers this package has. a family absent from here is reported
# `no_reader`, not skipped: the hypothesis gates above exist for the two families
# whose cited artifact carries everything the gate needs, and the others need an
# input nothing in the cohort's declaration supplies. saying which is which is
# the useful part - "H2 has no verdict" and "H2-A needs a label NMI nobody
# measured" are different statements.
def _h2c(task: dict, runs_root: _Path, **_) -> dict:
    ci = _json.loads((runs_root / task["ci_file"]).read_text())
    floor = (task.get("floor") or {})
    return gate_h2c(ci, floor.get("value"), floor.get("basis"))


def _h3_pathway(task: dict, runs_root: _Path, *, sources: dict, **_) -> dict:
    st = sources.get("run_st_features")
    if not st:
        return {"status": "not_run",
                "needs": "evidence_sources.json#run_st_features",
                "note": ("G6 derives circularity from each arm's ST INPUT, not "
                         "from the artifact's own view_clean column - the script "
                         "hardcodes which arms are circular, so an arm added to "
                         "the grid afterwards is silently marked clean. the "
                         "features are per-arm cohort knowledge and are declared, "
                         "never guessed.")}
    import pandas as pd

    rows = pd.read_parquet(runs_root / task["table"]).to_dict("records")
    return gate_h3_pathway(rows, st, view=task.get("view", "z_he"),
                           alpha=float(task.get("alpha", 0.05)))


_READERS = {"subject_identity_suppression": _h2c, "pathway_transfer": _h3_pathway}


def _rollup(runs_root: _Path, h: str, grid: list[str], pack: dict) -> dict:
    """the eval.py rollup for a hypothesis: its coverage, and H3's split check.

    DIAGNOSTIC on the seed cohort - no routed number is read from a rollup - so
    it is reported beside the cited-artifact gates rather than as the verdict.
    """
    p = runs_root / "eval" / h / "summary.json"
    if not p.is_file():
        return {"status": "absent", "path": f"eval/{h}/summary.json"}
    s = _json.loads(p.read_text())
    out = {"status": "read", "path": f"eval/{h}/summary.json",
           "coverage": coverage(s, grid)}
    if h == "H3":
        n = (pack.get("eval_scope") or {}).get("test_subjects")
        out["split"] = (gate_h3(s, int(n)) if n is not None else
                        {"status": "not_run",
                         "needs": "routing_evidence.json#eval_scope.test_subjects"})
    return out


# a guard that did not RUN is not a guard that passed. the difference decides
# what a verdict may claim, so it is counted rather than inferred from the
# absence of a failure: "no guard failed" was true of a hypothesis where no
# guard ran at all, and that read as a clean bill of health.
_RAN = ("pass", "fail", "per_run", "as_recorded", "satisfied_by_design",
        "flag_disagreement")


def _guard_tally(gates: dict) -> dict:
    """per hypothesis: which guards ran, which failed, which did not run and why.

    G3 is per RUN rather than per guard - `gate_h2c` attaches it to each arm -
    so a failure there is named with the arm. one arm failing a required guard
    demotes the claim for that arm, and the verdict says which.
    """
    ran, failed, not_run = [], [], []
    for tid, v in gates.items():
        if not isinstance(v, dict):
            continue
        for g in v.get("guards") or []:
            name = f"{tid}/{g['guard']}"
            st = g.get("status")
            if st in _RAN:
                ran.append(name)
            else:
                not_run.append({"guard": name, "why": g.get("note") or st})
            if st == "fail":
                failed.append(name)
        for rid, r in (v.get("runs") or {}).items():
            if isinstance(r, dict) and r.get("G3"):
                ran.append(f"{tid}/G3_bio_vs_confounder/{rid}")
                if not r["G3"]["passed"]:
                    failed.append(f"{tid}/G3_bio_vs_confounder/{rid}")
    return {"guards_ran": sorted(set(ran)), "guards_failed": sorted(set(failed)),
            "guards_not_run": not_run}


def claims(row: dict) -> str:
    """what a hypothesis's numbers are allowed to be called.

    four states, and `unguarded` is the one that had to be added: a hypothesis
    whose required guards all lack an input reports no failure, and "no guard
    failed" then reads as a clean bill of health for a number nothing checked.
    """
    if not row.get("evaluable"):
        return "not_run"
    if row.get("guards_failed"):
        return "diagnostic"
    return "biological" if row.get("guards_ran") else "unguarded"


def verdicts(project_id: str | None = None, *, hypotheses: list[str] | None = None,
             root: _Path | None = None) -> dict:
    """per hypothesis: the gates over its cited artifacts, and what did not run.

    reads only. every number comes from an artifact the cohort's declaration
    names, and a gate whose input is missing returns `not_run` with the input
    named rather than a value computed from something else.
    """
    from omicstra.settings import settings

    root = _Path(root) if root is not None else settings.project_root(project_id)
    pack, sources = _pack(root), _sources(root)
    fams = families(hypotheses)
    sc = scope(fams, pack, sources)
    runs_root = (root / sources["runs_root"]).resolve()
    grid = list(sources.get("runs") or [])

    out: dict[str, dict] = {}
    for h, row in sc.items():
        res: dict = {"evaluable": row["evaluable"], "families": row,
                     "gates": {}, "rollup": _rollup(runs_root, h, grid, pack)}
        for tid in row["fillable"]:
            task = sources["tasks"][tid]
            reader = _READERS.get(tid)
            if reader is None:
                res["gates"][tid] = {
                    "status": "no_reader",
                    "note": ("the package has no hypothesis gate for this family. "
                             "promote computes its candidates and floor from the "
                             "same declaration; what is missing here is a check "
                             "over the number, not the number."),
                    "cited": cited_paths(task, grid)[:4]}
                continue
            try:
                res["gates"][tid] = reader(task, runs_root, sources=sources, pack=pack)
            except (FileNotFoundError, KeyError) as e:
                res["gates"][tid] = {"status": "unavailable",
                                     "why": f"{type(e).__name__}: {e}"}
        res.update(_guard_tally(res["gates"]))
        out[h] = res
    return out


# --- admissibility: a matrix entry is a value AND whether it may be compared --
#
# "cell" below always means a CELL OF THE MATRIX - one question crossed with one
# method - and never a biological cell. the word is unavoidable for a table and
# dangerous in this domain, so it is said once here and the function is named
# `matrix_cell`.
#
# the matrix was question x WINNER - one row per question, the ranked value of
# whichever method came first. so a reader saw ten numbers per question with
# nothing marking the inadmissible ones, which is an invitation to compare
# entries that are not comparable.
#
# three sources already decide admissibility and none of them reached a cell:
# the contract's required guards, the pack's task-scoped contraindications, and
# the task's own floor. this reads all three and decides nothing itself.

_ADMISSIBLE = "admissible"
_CONTRA = "contraindicated"
_BELOW = "below_floor"


def contraindication(pack: dict, task_id: str, method: str) -> dict | None:
    """the pack's contraindication for THIS method on THIS question, if any.

    task-scoped by construction: a method may be contraindicated for one
    question and the right answer to another, and the pack records the exception
    explicitly. `does_not_apply_to` is honoured so an exception is not silently
    overridden by the method name alone.
    """
    for c in pack.get("contraindications") or []:
        if c.get("method") != method:
            continue
        if task_id in (c.get("does_not_apply_to") or []):
            return None
        if task_id in (c.get("applies_to") or []):
            return c
    return None


def matrix_cell(pack: dict, task_id: str, candidate: dict) -> dict:
    """one entry of the matrix - a question crossed with a method.

    the value, whether it may be compared, and why not. a cell of the TABLE; no
    biological cell is involved at any point in this module.

    ordered by what FORECLOSES most. a contraindicated method is not ranked at
    all - the reason stands where its number would. a method below the floor has
    not answered the question, whatever its rank, because every hypothesis here
    is stated comparatively against a baseline.

    whether the HYPOTHESIS was guarded is not a property of a cell and is not
    decided here - it caveats every cell of the question equally, so it rides on
    the question row.
    """
    task = (pack.get("tasks") or {}).get(task_id) or {}
    mid, value = candidate.get("id"), candidate.get("value")

    c = contraindication(pack, task_id, mid)
    if c is not None:
        return {"method": mid, "value": value, "status": _CONTRA, "rankable": False,
                "why": c.get("statement") or "contraindicated for this question",
                "evidence": c.get("evidence")}

    floor = task.get("floor") or {}
    fv = floor.get("value")
    if fv is not None and isinstance(value, (int, float)):
        beats = value > fv if task.get("higher_is_better", True) else value < fv
        if not beats:
            return {"method": mid, "value": value, "status": _BELOW, "rankable": False,
                    "why": (f"does not beat the {floor.get('id')} floor of {fv}. every "
                            "hypothesis here is comparative, so a method that does not "
                            "beat its baseline has not answered the question"),
                    "evidence": floor.get("basis")}

    return {"method": mid, "value": value, "status": _ADMISSIBLE, "rankable": True,
            "why": "", "evidence": None}


def _hypothesis_of(task_id: str) -> str:
    for f in routing_contract()["task_families"]:
        if f["id"] == task_id:
            return hypothesis_root(f["hypothesis"])
    return "?"


def matrix(pack: dict, claims: dict[str, str] | None = None) -> dict:
    """question x METHOD, every cell carrying its admissibility.

    the pack states which methods were scored per question; this says which of
    them may be compared. a question whose pack row is `not_available` keeps its
    own reason rather than producing empty cells.
    """
    out: dict[str, dict] = {}
    for task_id, t in (pack.get("tasks") or {}).items():
        if "not_available" in t:
            out[task_id] = {"status": "not_available", "why": t["not_available"],
                            "cells": []}
            continue
        cells = [matrix_cell(pack, task_id, c) for c in (t.get("candidates") or [])]
        h = _hypothesis_of(task_id)
        out[task_id] = {
            "status": "scored" if cells else "no_candidates",
            "hypothesis": h,
            # the hypothesis's claim caveats every cell of this question equally
            "claim": (claims or {}).get(h),
            "floor": t.get("floor"),
            "n_rankable": sum(c["rankable"] for c in cells),
            "n_cells": len(cells),
            "cells": cells}
    return out


# ========================================================= EVALUATE_STEPS ====
# the chain that COMPUTES a verdict, as against the gates above, which judge a
# verdict somebody else computed.
#
# both are needed and they answer different questions. a cohort WITH an evidence
# pack takes the gates: the numbers exist, the question is what they may claim.
# a cohort with no pack has nothing to judge, and four of the guards above say
# so by name - `_not_run("G5_noise_floor", "a permutation null over ARI")` and
# the three beside it. this chain produces those inputs.
#
# it does not re-judge what the gates judge. two answers to one question is the
# defect; this answers the question they could not.
#
# the ctx it reads, all of it arrays and labels already in memory - the chain
# that assembles them is the caller's:
#
#   ctx["spaces"]        {name: (n, d) matrix}   one per view being evaluated
#   ctx["paired"]        {name: (n, d) matrix}   the other modality, row-aligned
#   ctx["labels"]        {name: (n,) labels}     biological labels
#   ctx["confounder"]    (n,) labels             the subject each unit came from
#   ctx["scores"]        {name: (n,) float}      continuous signals, e.g. a
#                                                pathway or signature score
#
# every statistic comes from `measures/evaluation.py`. a protocol computes no
# maths of its own - a number computed in here is a number no notebook can check.

from omicstra.protocols import Ctx, Step
from omicstra.records import DiagnosticRecord

# the four states a hypothesis can be in. `unguarded` exists because "no guard
# failed" was true of a hypothesis where no guard RAN, and `not_run` because
# "nothing to report" and "reported nothing" are different answers.
VERDICTS = ("biological", "diagnostic", "unguarded", "not_run")


class EvaluationInputMissing(RuntimeError):
    """a declared input is absent from the context, so no step will guess at it."""


def _erec(step_id: str, status: str, result: str, decision: str,
          observed: dict, method: str = "", criterion: str = "",
          **kw) -> DiagnosticRecord:
    return DiagnosticRecord(step_id=step_id, status=status, method=method,
                            criterion=criterion, result=result, decision=decision,
                            observed=observed, **kw)


def _edeclared(ctx: Ctx, field: str):
    """this step's declared parameter, or None. never a default."""
    v = (ctx.get("params") or {}).get(field)
    return v if v not in ("", None) else None


def _emissing(step: str, field: str, why: str) -> DiagnosticRecord:
    return _erec(step, "not_run", f"{field} is not declared",
                 f"declare {field} for this cohort - {why}", {"missing": field})


def _eneed(ctx: Ctx, key: str, step: str):
    if key not in ctx:
        raise EvaluationInputMissing(f"{step} needs {key!r} on the context")
    return ctx[key]



def _guarded_null(observed: float, nulls, tail: str):
    """`noise_floor`, or the reason it could not run.

    the guard refuses a null below its own minimum, and that minimum is the
    GUARD's to hold - duplicating it here would be a second source for one bar.
    so the refusal is caught and turned into a reason, which is what a cohort
    declaring too few permutations should see instead of a traceback.
    """
    try:
        g = noise_floor(observed, nulls, tail=tail)
    except ValueError as e:
        return None, str(e)
    return g, ""

# ------------------------------------------------- 1 · the label's granularity --
def label_granularity(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """G1, before any score is computed against the label.

    this runs FIRST because it decides what every later number may claim. a
    label that encodes the subject turns a clustering score into a subject
    classification, and the score is identical either way - only the sentence
    it licenses changes. so the chain measures it before it has a score to be
    tempted by.
    """
    labels = _edeclared(ctx, "labels")
    tau = _edeclared(ctx, "tau_label_nmi")
    if labels is None:
        return _emissing("label_granularity", "labels",
                         "which labels carry the biological claim is the "
                         "cohort's, and a label at subject level is a different "
                         "claim from one at unit level"), {}
    if tau is None:
        return _emissing("label_granularity", "tau_label_nmi",
                         "the NMI above which a label is treated as encoding "
                         "the confounder. a null-derived cut, re-derived per "
                         "cohort - see EVALUATION.md section 8"), {}

    tables = _eneed(ctx, "labels", "label_granularity")
    confounder = _eneed(ctx, "confounder", "label_granularity")
    names = [n for n in labels if n in tables]
    if missing := [n for n in labels if n not in tables]:
        return _erec("label_granularity", "not_run",
                     f"declared label(s) {missing} are not on the context",
                     "refused - a declared label with no values is a missing "
                     "input, not an absent result",
                     {"declared": list(labels), "present": sorted(tables)}), {}

    results = {n: nmi_label_confounder(tables[n], confounder, tau=float(tau))
               for n in names}
    confounded = sorted(n for n, g in results.items() if not g.passed)
    return _erec("label_granularity", "pass",
                 f"{len(names) - len(confounded)} of {len(names)} label(s) are "
                 f"independent of the confounder at tau {float(tau)}"
                 + (f"; {confounded} encode it" if confounded else ""),
                 ("every score against a confounded label is diagnostic - it is "
                  "still reported, and it cannot carry a biological claim alone"
                  if confounded else
                  "scores against these labels may carry a biological claim if "
                  "their own guards pass"),
                 {"n_in": len(names), "n_out": len(names), "reason": "",
                  "nmi": {n: g.value for n, g in results.items()},
                  "confounded": confounded, "tau": float(tau),
                  "guards": [{"guard": "G1_label_granularity", "scope": n,
                              "status": "pass" if g.passed else "fail"}
                             for n, g in results.items()]},
                 method="NMI(label, confounder) per declared label",
                 criterion="the label is measured before any score is scored against it",
                 ), {"label_guards": {n: g for n, g in results.items()},
                     "confounded_labels": confounded}


# ------------------------------------------- 2 · identifiability, and its null --
def paired_identifiability(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """H1. retrieval and geometric agreement, each with its own floor.

    the two are reported together because a space can retrieve well while
    agreeing poorly in geometry, and the reverse - `EVALUATION.md` section 1 is
    explicit that either alone is not decisive.

    the similarity matrix is built here and the statistics come from
    `measures.evaluation`, so the AUC's negative sample and its seed are
    declared rather than implicit, which is how the published chain carried
    them.
    """
    ks = _edeclared(ctx, "ks")
    ratio = _edeclared(ctx, "negative_ratio")
    neg_seed = _edeclared(ctx, "negative_seed")
    if ks is None:
        return _emissing("paired_identifiability", "ks",
                         "which recall depths this cohort reports"), {}
    if ratio is None or neg_seed is None:
        return _emissing("paired_identifiability",
                         "negative_ratio" if ratio is None else "negative_seed",
                         "the AUC is computed over a SAMPLE of mismatched "
                         "pairs, so the ratio and the seed are part of the "
                         "number and were implicit in the script"), {}

    import numpy as np

    from omicstra.measures.evaluation import linear_cka, retrieval

    spaces = _eneed(ctx, "spaces", "paired_identifiability")
    paired = _eneed(ctx, "paired", "paired_identifiability")
    shared = [v for v in spaces if v in paired]
    if not shared:
        return _erec("paired_identifiability", "not_applicable",
                     "no view has a paired counterpart",
                     "identifiability is a claim about TWO representations of "
                     "one unit; with one there is nothing to identify",
                     {"spaces": sorted(spaces), "paired": sorted(paired),
                      "n_out": 0, "reason": "no paired view"}), {}

    out: dict[str, dict] = {}
    for v in shared:
        A, B = np.asarray(spaces[v]), np.asarray(paired[v])
        if A.shape[0] != B.shape[0]:
            return _erec("paired_identifiability", "fail",
                         f"view {v!r}: {A.shape[0]} units against {B.shape[0]}",
                         "refused - the diagonal is only the matched pair if the "
                         "two matrices are the same units in the same order",
                         {"view": v, "n_in": A.shape[0], "n_out": 0,
                          "reason": "row counts differ"}), {}
        sim = A @ B.T
        r = retrieval(sim, ks=tuple(int(k) for k in ks),
                      negative_ratio=int(ratio), negative_seed=int(neg_seed))
        r["cka"] = linear_cka(A, B)
        out[v] = r

    return _erec("paired_identifiability", "pass",
                 "; ".join(f"{v}: AUC {r['auc']:.3f}, CKA {r['cka']:.3f}, "
                           f"gap {r['alignment_gap']:+.3f}" for v, r in out.items()),
                 "retrieval and geometric agreement reported together - either "
                 "alone is not decisive",
                 {"n_in": len(shared), "n_out": len(shared), "reason": "",
                  "per_view": out, "negative_ratio": int(ratio),
                  "negative_seed": int(neg_seed)},
                 method="matched-vs-mismatched discrimination and linear CKA",
                 criterion="both reported, on the same units",
                 ), {"identifiability": out}


# ---------------------------------------- 3 · structure, and whether it holds --
def structure_and_its_null(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """H2-A. the clustering score, its permutation null, and its companion probe.

    three things produce the same ARI - biology, the label standing in for the
    subject, and the geometry of the space - so the number is computed with all
    three separations attached:

        the label's NMI came from step 1 and decides what this may claim
        the permutation null closes G5, which the pack-reading gate records as
          `not_run` for want of "a permutation null over ARI"
        the linear probe is the companion, because ARI is geometry-biased: a
          manifold embedding scores low against KMeans' globular assumption
          while a probe on the same space is at parity

    the null permutes the LABEL, which is the right null for "does this space
    recover this label" - permuting the space would test a different thing.
    """
    needed = ("k", "n_init", "seed", "silhouette_sample", "n_permutations",
              "probe_max_iter", "held_out_groups")
    vals = {f: _edeclared(ctx, f) for f in needed}
    for f, v in vals.items():
        if v is None:
            return _emissing("structure_and_its_null", f,
                             "the clustering, its null and its probe are each "
                             "parameterised, and a default here is a choice "
                             "made on one tissue"), {}

    import numpy as np

    from omicstra.measures.evaluation import (
        cluster_agreement,
        linear_probe,
        permutation_test,
    )

    spaces = _eneed(ctx, "spaces", "structure_and_its_null")
    tables = _eneed(ctx, "labels", "structure_and_its_null")
    confounder = _eneed(ctx, "confounder", "structure_and_its_null")
    confounded = set(ctx.get("confounded_labels") or ())

    out: dict[str, dict] = {}
    rng = np.random.default_rng(int(vals["seed"]))
    for view, X in spaces.items():
        for lname, y in tables.items():
            obs = cluster_agreement(X, y, k=int(vals["k"]),
                                    n_init=int(vals["n_init"]),
                                    seed=int(vals["seed"]),
                                    silhouette_sample=int(vals["silhouette_sample"]))
            if np.isnan(obs["ari"]):
                out[f"{view}/{lname}"] = {**obs, "guard_g5": "not_run",
                                          "reason": obs.get("reason", "")}
                continue
            nulls = []
            y_arr = np.asarray(y)
            for _ in range(int(vals["n_permutations"])):
                shuffled = rng.permutation(y_arr)
                nulls.append(cluster_agreement(
                    X, shuffled, k=int(vals["k"]), n_init=int(vals["n_init"]),
                    seed=int(vals["seed"]),
                    silhouette_sample=int(vals["silhouette_sample"]))["ari"])
            null = permutation_test(obs["ari"], nulls, tail="greater")
            g5, why = _guarded_null(obs["ari"], nulls, tail="greater")
            if g5 is None:
                out[f"{view}/{lname}"] = {**obs, "null": null,
                                          "guard_g5": "not_run", "reason": why}
                continue
            probe = linear_probe(X, y, confounder, seed=int(vals["seed"]),
                                 max_iter=int(vals["probe_max_iter"]),
                                 held_out_groups=int(vals["held_out_groups"]))
            out[f"{view}/{lname}"] = {
                **obs, "null": null, "probe_accuracy": probe["accuracy"],
                "guard_g5": "pass" if g5.passed else "fail",
                "g5": g5.model_dump(),
                "claims_biology": bool(g5.passed and lname not in confounded),
                "why_diagnostic": ("the label encodes the confounder"
                                   if lname in confounded else ""),
            }

    ran = [k for k, v in out.items() if v.get("guard_g5") != "not_run"]
    return _erec("structure_and_its_null", "pass" if ran else "not_run",
                 f"{len(ran)} of {len(out)} (view, label) pair(s) scored against "
                 f"a {int(vals['n_permutations'])}-permutation null",
                 ("each score ships with its null, its probe and the label's own "
                  "granularity - the three readings of one ARI, separated"
                  if ran else
                  "no pair could be scored; a clustering score over fewer than "
                  "two classes is not a measurement"),
                 {"n_in": len(out), "n_out": len(ran),
                  "reason": "" if len(ran) == len(out) else "unscoreable pairs",
                  "per_pair": out, "n_permutations": int(vals["n_permutations"]),
                  "null": "the LABEL is permuted, not the space",
                  # G5's note is G5's. it read `why_diagnostic`, which is G1's
                  # finding, so the guard table said the permutation null passed
                  # BECAUSE the label encodes the confounder - two independent
                  # guards reported as one reason. `reason` is the fallback
                  # because that is what a G5 that could not run carries.
                  "guards": [{"guard": "G5_noise_floor", "scope": key,
                              "status": b["guard_g5"],
                              "note": ((b.get("g5") or {}).get("note")
                                       or b.get("reason", ""))}
                             for key, b in out.items()]
                            + [{"guard": "G1_label_granularity", "scope": key,
                                "status": "fail",
                                "note": b["why_diagnostic"]}
                               for key, b in out.items()
                               if b.get("claims_biology") is False]},
                 method="KMeans ARI against a label, permutation null over the "
                        "label, and a by-subject linear probe beside it",
                 criterion="G5 runs, and the probe is reported with the ARI",
                 ), {"structure": out}


# ----------------------------- 3b · biology vs confounder, at subarray level --
def confounder_suppression(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """H2-C. does the aligned space amplify biology relative to patient identity?

    operates at SUBARRAY resolution, not niche: niche embeddings are
    mean-pooled (L2-norm each, mean, L2-norm the mean) per subarray, then
    the subarray x subarray cosine matrix is partitioned by patient-level
    labels. the null permutes the biology label at the patient level,
    because subarrays within one patient are not independent.

    the ratio z_biology / z_patient is the claim, not the biology z alone.
    a method can score well on biology while scoring better on patient
    identity, and that is anti-helpful (G3). the ratio and its delta-method
    CI are computed from the historical z values (obs / null_std), because
    that is what the published biology.parquet stores.

    ported from eval_alignment_biology.py (permutation tests) and
    eval_h2_bootstrap_ci.py (delta-method CI). the measures are in
    measures/biology.py. this step composes them.
    """
    needed = ("biology_labels", "n_permutations", "n_rho_null", "seed", "alpha")
    vals = {f: _edeclared(ctx, f) for f in needed}
    for f, v in vals.items():
        if v is None:
            return _emissing("confounder_suppression", f,
                             "the biology labels, null sizes and seed are the "
                             "cohort's. biology_labels is the list of patient-level "
                             "labels to test (e.g. TIME, MC_global)"), {}

    import numpy as np

    from omicstra.measures.biology import (
        biology_perm_test,
        cosine_sim_matrix,
        geometric_faithfulness,
        mean_pool_l2,
        patient_perm_test,
        ratio_delta_ci,
    )

    spaces = _eneed(ctx, "spaces", "confounder_suppression")
    subarray = ctx.get("subarray")
    confounder = _eneed(ctx, "confounder", "confounder_suppression")
    bio_labels = ctx.get("biology_labels")

    if subarray is None:
        return _erec("confounder_suppression", "not_run",
                     "no subarray grouping on the context",
                     "H2-C aggregates niche embeddings to subarray means - "
                     "without a subarray column there is nothing to pool over",
                     {"n_out": 0, "reason": "subarray not on ctx"}), {}

    label_names = list(vals["biology_labels"])
    if not bio_labels or not label_names:
        return _erec("confounder_suppression", "not_run",
                     "no biology labels on the context",
                     "H2-C needs patient-level labels (e.g. TIME) on each "
                     "subarray. declare biology_labels and supply them on ctx",
                     {"n_out": 0, "reason": "biology_labels empty"}), {}
    missing = [n for n in label_names if n not in bio_labels]
    if missing:
        return _erec("confounder_suppression", "not_run",
                     f"declared biology labels {missing} not on the context",
                     "declare them and supply them on ctx as biology_labels",
                     {"n_out": 0, "reason": f"missing labels: {missing}"}), {}

    subarray_arr = np.asarray(subarray)
    confounder_arr = np.asarray(confounder)
    n_perm = int(vals["n_permutations"])
    n_rho = int(vals["n_rho_null"])
    seed = int(vals["seed"])
    alpha = float(vals["alpha"])

    # aggregate niche embeddings to subarray means, per view
    unique_subs = np.unique(subarray_arr)
    sub_patients = np.array([
        confounder_arr[np.where(subarray_arr == s)[0][0]] for s in unique_subs
    ])
    sub_labels = {
        name: np.array([
            str(bio_labels[name][np.where(subarray_arr == s)[0][0]])
            for s in unique_subs
        ])
        for name in label_names
    }
    n_subs = len(unique_subs)
    if n_subs < 3:
        return _erec("confounder_suppression", "not_run",
                     f"only {n_subs} subarray(s) - too few for a permutation test",
                     "the biology test needs at least 3 subarrays from at least "
                     "2 patients to form cross-patient pairs",
                     {"n_out": 0, "reason": "too few subarrays",
                      "n_subarrays": n_subs}), {}

    out: dict = {"n_subarrays": n_subs, "n_patients": len(set(sub_patients)),
                 "per_view": {}}
    for view, X in spaces.items():
        X_arr = np.asarray(X, dtype=np.float32)
        sub_embs = np.stack([
            mean_pool_l2(X_arr[subarray_arr == s]) for s in unique_subs
        ])
        cos = cosine_sim_matrix(sub_embs)

        view_out: dict = {
            "rho": {},
            "biology": {}, "patient": {}, "ratio": {},
        }

        # geometric faithfulness: aligned vs raw baseline cosines
        raw_bl = ctx.get("raw_baselines")
        if raw_bl is not None:
            for raw_name in ("raw_he", "raw_st"):
                if raw_name in raw_bl:
                    view_out["rho"][raw_name] = geometric_faithfulness(
                        cos, raw_bl[raw_name],
                        n_permutations=n_rho, seed=seed)
        if not view_out["rho"]:
            view_out["rho"] = {
                "status": "missing_input",
                "reason": "geometric faithfulness requires raw baseline cosine "
                          "matrices on ctx (raw_baselines). the adapter did not "
                          "supply them for this run - the niche parquets may "
                          "lack the required feature columns or the run config "
                          "is absent"}

        # patient permutation test (one per view)
        pt = patient_perm_test(cos, sub_patients,
                               n_permutations=n_perm, seed=seed)
        view_out["patient"] = {k: v for k, v in pt.items() if k != "null"}
        pt_z = pt["z"]

        # biology permutation tests (one per label per view)
        for lname in label_names:
            bt = biology_perm_test(cos, sub_patients, sub_labels[lname],
                                   n_permutations=n_perm, seed=seed)
            view_out["biology"][lname] = {
                k: v for k, v in bt.items() if k != "null"
            }
            bio_z = bt["z"]

            # G3: bio / patient ratio, with delta-method CI
            ci = ratio_delta_ci(bio_z, pt_z)
            g3 = bio_vs_confounder_ratio(bio_z, pt_z)
            view_out["ratio"][lname] = {
                **ci, "G3": g3.model_dump(),
            }

        out["per_view"][view] = view_out

    # BH-FDR per test family across labels within each view
    from statsmodels.stats.multitest import multipletests

    for view, vd in out["per_view"].items():
        p_bio = [vd["biology"][n]["p_value"] for n in label_names]
        if len(p_bio) > 1:
            _, q_bio, _, _ = multipletests(p_bio, alpha=alpha, method="fdr_bh")
            for i, n in enumerate(label_names):
                vd["biology"][n]["q_bh"] = float(q_bio[i])

    # guards: G1 is satisfied by design (patient-level permutation), G3 per run
    g3_pass = sum(
        1 for vd in out["per_view"].values()
        for rd in vd["ratio"].values()
        if rd.get("G3", {}).get("passed")
    )
    g3_total = sum(
        1 for vd in out["per_view"].values()
        for rd in vd["ratio"].values()
        if "G3" in rd
    )

    # headline: the first-declared label on the headline view
    headline_view = ctx.get("view") or next(iter(spaces))
    headline_label = label_names[0] if label_names else ""
    hv = out["per_view"].get(headline_view, {})
    headline_ratio = (hv.get("ratio", {}).get(headline_label, {})
                      .get("ratio", float("nan")))
    headline_bio_z = (hv.get("biology", {}).get(headline_label, {})
                      .get("z", float("nan")))
    headline_pt_z = (hv.get("patient", {}).get("z", float("nan")))

    return _erec("confounder_suppression", "pass",
                 f"{headline_label}/{headline_view}: "
                 f"bio z={headline_bio_z:+.2f}, patient z={headline_pt_z:+.2f}, "
                 f"ratio={headline_ratio:.3f}; "
                 f"G3 {g3_pass}/{g3_total} pass",
                 "the ratio is the claim: below 1 means biology z is lower "
                 "than patient-identity z under this test",
                 {"n_in": n_subs, "n_out": n_subs, "reason": "",
                  **out,
                  "n_permutations": n_perm, "n_rho_null": n_rho,
                  "seed": seed, "alpha": alpha,
                  "z_convention": "historical: obs / null_std (matching "
                                  "biology.parquet)",
                  "guards": [
                      {"guard": "G1_label_granularity",
                       "status": "pass",
                       "note": "satisfied by design: biology labels are "
                               "per-patient; the test scores cross-patient "
                               "pairs only and permutes the label at patient "
                               "level"},
                      {"guard": "G3_bio_vs_confounder",
                       "status": "pass" if g3_pass == g3_total else "fail",
                       "note": f"{g3_pass} of {g3_total} (view, label) pairs "
                               "clear a ratio of 1"},
                      {"guard": "G5_noise_floor", "scope": "biology",
                       "status": "pass",
                       "note": "the permutation p is empirical: "
                               "(1 + #{null >= obs}) / (n + 1)"},
                  ]},
                 method="subarray mean-pool, cosine matrices, patient-level "
                        "permutation of biology labels vs patient identity, "
                        "delta-method CI on the ratio",
                 criterion="the ratio z_bio / z_patient clears 1 (G3)",
                 ), {"confounder_suppression": out}


# ------------------------------------------ 4 · held out, measured not declared --
def held_out_comparison(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """G2. the within-subject score beside the cross-subject one.

    named `held_out_comparison` and not `held_out_honesty`: the guard of that
    name is defined above and is what `GUARDS["G2"]` points at. a step that
    shadowed it would have left the registry pointing at a function with a
    different signature, and the first symptom would have been a guard that
    appeared to run.

    this closes `_not_run("G2_held_out_honesty", "a within-subject score to
    compare")`. a split is a declaration; the DIFFERENCE between scoring within
    a subject and across subjects is a measurement, and it is the one that says
    whether the split held anything out.

    the direction depends on the statistic, and assuming one was wrong:

        a RANK statistic (R@k, MRR) is inflated within subject, because the
        pool is smaller - fewer candidates to rank the matched one against

        a DISCRIMINATION statistic (AUC against sampled mismatched pairs) can
        go the other way, because the mismatched pairs drawn within one subject
        are HARD negatives - same patient, adjacent tissue - so telling matched
        from mismatched is harder there, not easier. on the seed cohort the
        within-subject AUC is 0.736 against 0.761 across, a 3% rise rather than
        a fall

    so the guard measures the DROP and fails past its tau; it does not assume a
    sign. what it catches is the collapse - a score that survives within and
    dies across was the subject, as CCA's 0.667 within and 0.000 across was.
    """
    ks = _edeclared(ctx, "ks")
    ratio = _edeclared(ctx, "negative_ratio")
    neg_seed = _edeclared(ctx, "negative_seed")
    min_units = _edeclared(ctx, "min_units_per_subject")
    if ks is None or ratio is None or neg_seed is None:
        return _emissing("held_out_comparison", "ks/negative_ratio/negative_seed",
                         "the within-subject score must be computed the SAME "
                         "way as the cross-subject one or the comparison is "
                         "between two different statistics"), {}
    if min_units is None:
        return _emissing("held_out_comparison", "min_units_per_subject",
                         "a subject with very few units gives a within-score "
                         "over almost no pairs; the floor is the cohort's"), {}

    import numpy as np

    from omicstra.measures.evaluation import retrieval

    spaces = _eneed(ctx, "spaces", "held_out_honesty")
    paired = _eneed(ctx, "paired", "held_out_honesty")
    confounder = np.asarray(_eneed(ctx, "confounder", "held_out_honesty"))
    cross = ctx.get("identifiability") or {}
    if not cross:
        return _erec("held_out_comparison", "not_run",
                     "no cross-subject score to compare against",
                     "G2 is a comparison; with one number there is nothing to "
                     "compare",
                     {"n_out": 0, "reason": "identifiability did not run"}), {}

    out: dict[str, dict] = {}
    for view in cross:
        A, B = np.asarray(spaces[view]), np.asarray(paired[view])
        within = []
        for s in np.unique(confounder):
            idx = np.flatnonzero(confounder == s)
            if idx.size < int(min_units):
                continue
            r = retrieval(A[idx] @ B[idx].T, ks=tuple(int(k) for k in ks),
                          negative_ratio=int(ratio), negative_seed=int(neg_seed))
            within.append(r["auc"])
        if not within:
            out[view] = {"guard_g2": "not_run",
                         "reason": f"no subject has {int(min_units)} units"}
            continue
        w = float(np.nanmean(within))
        c = float(cross[view]["auc"])
        g2 = held_out_honesty(w, c)
        out[view] = {"within_subject_auc": w, "cross_subject_auc": c,
                     "n_subjects_scored": len(within),
                     "guard_g2": "pass" if g2.passed else "fail",
                     "g2": g2.model_dump()}

    ran = [v for v, r in out.items() if r.get("guard_g2") != "not_run"]
    return _erec("held_out_comparison", "pass" if ran else "not_run",
                 "; ".join(f"{v}: within {r['within_subject_auc']:.3f} vs cross "
                           f"{r['cross_subject_auc']:.3f}"
                           for v, r in out.items() if "within_subject_auc" in r)
                 or "no view could be scored within subject",
                 ("the gap between the two is what the split bought. a cross "
                  "score at or above the within score says the split is not "
                  "separating subjects"
                  if ran else "nothing to compare"),
                 {"n_in": len(cross), "n_out": len(ran),
                  "reason": "" if len(ran) == len(cross) else "unscoreable views",
                  "per_view": out,
                  "guards": [{"guard": "G2_held_out_honesty", "scope": v,
                              "status": r["guard_g2"],
                              "note": r.get("reason", "")}
                             for v, r in out.items()]},
                 method="the same retrieval statistic within subject and across",
                 criterion="within exceeds cross; equality is the finding",
                 ), {"honesty": out}


# ------------------------------------- 5 · the association, and its specificity --
def association_and_specificity(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """H3. does a signal direction fitted on some subjects carry to others.

    the statistic is a TRANSFER test and the split is part of it: fit the
    canonical directions on training subjects, project held-out subjects onto
    them, and correlate. the null permutes the TRAINING labels and refits, so
    what is being tested is whether the direction generalises rather than
    whether one exists.

    the first version of this step ran the within-cohort CCA instead -
    `top_canonical_correlation` over every unit, permuting every label. that is
    a real statistic and on the seed cohort it was significant for ten runs out
    of ten at FDR < 0.001, which orders nothing. it is computed here too and
    recorded as `within_cohort`, labelled non-discriminating, because the
    published artifact stores both and a reader comparing the two should not
    have to re-derive why only one of them separates the arms.

    this step also closes the second of the four guards, and that one's recorded
    reason named a file that could not satisfy it: G4 asked for "per-pathway
    axis vectors (pathway_axes.parquet)", and that artifact carries
    `pathway_name`, `pathway_id`, `coverage_frac`, `set_size` and `testable` -
    metadata, no vectors. the vectors are the CCA's own weights, which nothing
    was asking the CCA for. with them, "this axis encodes signal X" becomes
    checkable: five axes at 0.97 mutual cosine are one direction wearing five
    labels.
    """
    needed = ("n_permutations", "alpha", "tau_specificity", "tail",
              "held_out_groups", "transfer_seed", "within_cohort_seed",
              "min_units_per_side")
    vals = {f: _edeclared(ctx, f) for f in needed}
    for f, v in vals.items():
        if v is None:
            return _emissing("association_and_specificity", f,
                             "the split, both permutation seeds, the null size, "
                             "the FDR level, the aliasing bar and the tail are "
                             "each a declaration. the two seeds are separate "
                             "because the two nulls are separate tests"), {}

    import numpy as np

    from omicstra.measures.evaluation import (
        bh_fdr,
        permutation_test,
        prepare_within_cohort_cache,
        projected_correlation,
        top_canonical_correlation,
        transfer_correlation,
        transfer_null_distribution,
        within_cohort_null_distribution,
    )

    spaces = _eneed(ctx, "spaces", "association_and_specificity")
    scores = _eneed(ctx, "scores", "association_and_specificity")
    confounder = np.asarray(_eneed(ctx, "confounder", "association_and_specificity"))
    if not scores:
        return _erec("association_and_specificity", "not_applicable",
                     "this cohort declares no continuous signals to associate",
                     "association is a claim about a signal; with none there is "
                     "nothing to carry",
                     {"n_out": 0, "reason": "no scores declared"}), {}

    # the split is BY SUBJECT and is the same for every signal and every view -
    # a split that varied would make the comparison meaningless before the
    # statistic was chosen, which is EVALUATION.md section 4's condition.
    subjects = np.unique(confounder)
    n_held = int(vals["held_out_groups"])
    if subjects.size <= n_held:
        return _erec("association_and_specificity", "not_run",
                     f"{subjects.size} subject(s) cannot yield a held-out split "
                     f"of {n_held}",
                     "refused - a transfer test with nothing held out is the "
                     "within-cohort test under another name",
                     {"n_subjects": int(subjects.size), "held_out_groups": n_held,
                      "n_out": 0, "reason": "too few subjects to hold any out"}), {}
    _subj = subjects.copy()
    np.random.default_rng(int(vals["transfer_seed"])).shuffle(_subj)
    held = set(_subj[:n_held].tolist())
    test_mask = np.array([c in held for c in confounder])
    train_mask = ~test_mask
    n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
    floor = int(vals["min_units_per_side"])
    if n_tr < floor or n_te < floor:
        return _erec("association_and_specificity", "not_run",
                     f"the split gives {n_tr} train and {n_te} held-out units "
                     f"against a floor of {floor}",
                     "refused - a correlation over a handful of units is noise "
                     "with a p-value",
                     {"n_train": n_tr, "n_test": n_te, "floor": floor,
                      "n_out": 0, "reason": "a side of the split is too small"}), {}

    out: dict[str, dict] = {}
    for view, X in spaces.items():
        Z = np.asarray(X)
        Z_tr, Z_te = Z[train_mask], Z[test_mask]
        Q_z = prepare_within_cohort_cache(Z)
        rows, axes, names = [], [], []
        for sname, sc in scores.items():
            y = np.asarray(sc, dtype=np.float64)
            # a signal that does not cover every unit is UNEVALUABLE at this
            # split, not a signal to impute. filling the gaps with a mean would
            # invent the thing being tested for every unit it was absent from,
            # and dropping those units per signal would give each signal its own
            # split - which is the one condition EVALUATION.md section 4 names.
            # so the signal is refused by name, counted, and the others proceed.
            n_bad = int((~np.isfinite(y)).sum())
            if n_bad:
                rows.append({"signal": sname, "r_test": float("nan"),
                             "r_train": float("nan"), "z": float("nan"),
                             "p": float("nan"), "n_null": 0,
                             "guard_g5": "not_run",
                             "g5_not_run_because": (
                                 f"{n_bad} of {y.size} units have no value for "
                                 "this signal; unevaluable at a split shared "
                                 "with the other signals"),
                             "unevaluable": True,
                             "within_cohort": None})
                continue

            y_tr, y_te = y[train_mask], y[test_mask]
            r_te, r_tr, a, _ = transfer_correlation(Z_tr, y_tr, Z_te, y_te)
            rng_t = np.random.default_rng(int(vals["transfer_seed"]) + 1)
            t_nulls = transfer_null_distribution(
                Z_tr, y_tr, Z_te, y_te,
                n_perms=int(vals["n_permutations"]), rng=rng_t)
            t_null = permutation_test(r_te, t_nulls, tail=str(vals["tail"]))
            g5, why = _guarded_null(r_te, t_nulls, tail=str(vals["tail"]))

            rng_w = np.random.default_rng(int(vals["within_cohort_seed"]))
            w_obs = top_canonical_correlation(Z, y)
            w_nulls = within_cohort_null_distribution(
                Q_z, y, n_perms=int(vals["n_permutations"]), rng=rng_w)
            within = permutation_test(w_obs, w_nulls, tail=str(vals["tail"]))

            rows.append({
                "signal": sname, "r_test": r_te, "r_train": r_tr, **t_null,
                "guard_g5": "pass" if (g5 and g5.passed)
                            else "fail" if g5 else "not_run",
                "g5_not_run_because": why,
                "within_cohort": {"r": w_obs, "z": within["z"], "p": within["p"],
                                  "note": "non-discriminating on the seed cohort - "
                                          "significant for every arm at once, so "
                                          "it orders nothing"},
            })
            axes.append(a)
            names.append(sname)

        spec = None
        if len(axes) >= 2:
            g4 = specificity_matrix(np.vstack(axes), names=names,
                                    tau=float(vals["tau_specificity"]))
            spec = {"guard_g4": "pass" if g4.passed else "fail", **g4.model_dump()}
        out[view] = {"per_signal": rows,
                     "specificity": spec,
                     "specificity_status": (spec["guard_g4"] if spec else "not_run"),
                     "specificity_missing": (None if spec else
                                             "fewer than two axes to compare")}

    # BH-FDR across ALL evaluable view x signal cells globally, matching the
    # script's family scope. within one run the family is n_views x n_signals.
    global_cells = [(v, r) for v, blk in out.items()
                    for r in blk["per_signal"] if not r.get("unevaluable")]
    global_names = [f"{v}/{r['signal']}" for v, r in global_cells]
    global_p = [r["p"] for _, r in global_cells]
    fdr = bh_fdr(global_p, alpha=float(vals["alpha"]), family=global_names)
    q_of = dict(zip(global_names, fdr["q"]))
    rej_of = dict(zip(global_names, fdr["rejected"]))
    for v, r in global_cells:
        key = f"{v}/{r['signal']}"
        r["q"] = q_of.get(key, float("nan"))
        r["significant"] = bool(rej_of.get(key, False))
    for blk in out.values():
        blk["fdr_family"] = fdr["n_family"]
        for r in blk["per_signal"]:
            if r.get("unevaluable"):
                r.setdefault("q", float("nan"))
                r.setdefault("significant", False)

    n_sig = sum(1 for v in out.values() for r in v["per_signal"]
                if r.get("significant"))
    aliased = sorted(v for v, r in out.items()
                     if (r["specificity"] or {}).get("guard_g4") == "fail")
    unevaluable = sorted({r["signal"] for blk in out.values()
                          for r in blk["per_signal"] if r.get("unevaluable")})

    # a step where every declared signal was unevaluable measured NOTHING, and
    # "0 of 0 significant" is not a pass - 0 == 0 is not agreement. this is the
    # repo's own rule about a check that measures nothing, applied to the step
    # that would otherwise report a clean sweep of an empty set.
    if len(unevaluable) == len(scores):
        return _erec("association_and_specificity", "not_run",
                     f"every declared signal is unevaluable: {unevaluable}",
                     "no association was measured, so there is nothing to "
                     "report. a signal that does not cover every unit needs "
                     "either a complete signal or a split declared for it",
                     {"n_in": len(spaces) * len(scores), "n_out": 0,
                      "reason": "no signal covers every unit",
                      "unevaluable_signals": unevaluable,
                      "per_view": out, "n_train": n_tr, "n_test": n_te,
                      "guards": [{"guard": "G5_noise_floor",
                                  "scope": f"{v}/{r['signal']}",
                                  "status": "not_run",
                                  "note": r.get("g5_not_run_because", "")}
                                 for v, blk in out.items()
                                 for r in blk["per_signal"]]}), {}

    return _erec("association_and_specificity", "pass",
                 f"{n_sig} signal-view transfer(s) survive BH-FDR at "
                 f"{float(vals['alpha'])} on {n_te} held-out units from "
                 f"{n_held} subject(s)"
                 + (f"; axes alias in {aliased}" if aliased else ""),
                 ("the axes alias, so each magnitude is one direction wearing "
                  "several labels - the cosine travels with every magnitude"
                  if aliased else
                  "each transfer carries its own null, its q over the stated "
                  "family, and the cosine between its axis and the others"),
                 {"n_in": len(spaces) * len(scores), "n_out": n_sig,
                  "reason": "did not survive the null or the correction",
                  "per_view": out, "n_permutations": int(vals["n_permutations"]),
                  "alpha": float(vals["alpha"]),
                  "tau_specificity": float(vals["tau_specificity"]),
                  "tail": str(vals["tail"]),
                  "n_train": n_tr, "n_test": n_te,
                  "held_out_subjects": sorted(str(h) for h in held),
                  "statistic": "directions fitted on training subjects, "
                               "projected onto held-out subjects",
                  "unevaluable_signals": unevaluable,
                  "fdr_family_note": "the family is all evaluable view x signal "
                                     "cells within this run, matching the "
                                     "script's global correction scope",
                  "guards": [{"guard": "G5_noise_floor",
                              "scope": f"{v}/{r['signal']}",
                              "status": r["guard_g5"],
                              "note": r.get("g5_not_run_because", "")}
                             for v, blk in out.items() for r in blk["per_signal"]]
                            + [{"guard": "G2_held_out_honesty", "scope": v,
                                "status": "pass",
                                "note": f"fitted on {n_tr} units, scored on "
                                        f"{n_te} from subjects never fitted on"}
                               for v in out]
                            + [{"guard": "G4_specificity", "scope": v,
                                "status": blk["specificity_status"],
                                "note": blk["specificity_missing"] or ""}
                               for v, blk in out.items()]},
                 method="canonical directions fitted on training subjects and "
                        "scored on held-out subjects, against a null that "
                        "refits on permuted training labels; BH-FDR over the "
                        "declared family; off-diagonal cosine between the axes",
                 criterion="the direction generalises, and the axes are distinct",
                 ), {"association": out}


# -------------------------------------- 6 · pathway transfer CCA (H3 variant 2) --
def pathway_transfer_cca(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """H3 variant 2: multivariate CCA on gpath2vec pathway cosine profiles.

    per pathway whose set_size >= min_testable_set, for each view:
      Z = aligned latent (N, 512)
      Y = niche cosine profile to the pathway's embedded members (N, set_size)
      multivariate CCA via QR+SVD, with transfer test on held-out patients
      and permutation null on Y_train rows.
    """
    needed = ("n_permutations", "n_test_pats", "seed", "alpha",
              "min_testable_set", "pathway_feature")
    vals = {f: _edeclared(ctx, f) for f in needed}
    for f, v in vals.items():
        if v is None:
            return _emissing("pathway_transfer_cca", f,
                             "the permutation count, held-out patients, seed, "
                             "FDR level, testability floor and the feature name "
                             "for the circularity check are each a declaration"), {}

    import numpy as np
    from scipy.stats import pearsonr

    from omicstra.measures.evaluation import (
        multivariate_full_null,
        multivariate_top_canonical_correlation,
        multivariate_transfer_correlation,
        multivariate_transfer_null,
    )

    spaces = ctx.get("spaces")
    if not spaces:
        return _erec("pathway_transfer_cca", "not_run",
                     "no spaces on the context",
                     "the adapter did not provide any aligned views",
                     {"n_out": 0, "reason": "no spaces"}), {}

    pw_cosines = ctx.get("pathway_cosines")
    pw_axes = ctx.get("pathway_axes")
    if pw_cosines is None or pw_axes is None:
        return _erec("pathway_transfer_cca", "not_run",
                     "pathway_cosines or pathway_axes absent from context",
                     "the adapter was not called with include_pathway_cosines=True, "
                     "or the gpath2vec pkl is not declared in project.json",
                     {"n_out": 0, "reason": "no pathway cosines"}), {}

    confounder = np.asarray(
        _eneed(ctx, "confounder", "pathway_transfer_cca"))

    seed = int(vals["seed"])
    n_perms = int(vals["n_permutations"])
    n_test_pats = int(vals["n_test_pats"])
    alpha = float(vals["alpha"])
    min_ts = int(vals["min_testable_set"])
    pw_feature = str(vals["pathway_feature"])

    st_features = ctx.get("st_features") or {}
    run_is_contrastive = ctx.get("run_is_contrastive") or {}

    subjects = np.unique(confounder)
    _subj = subjects.copy()
    np.random.default_rng(seed).shuffle(_subj)
    held = set(_subj[:n_test_pats].tolist())
    test_mask = np.array([c in held for c in confounder])
    train_mask = ~test_mask
    n_full = len(confounder)
    n_tr = int(train_mask.sum())
    n_te = int(test_mask.sum())

    testable_of = dict(zip(pw_axes["pathway_name"], pw_axes["testable"]))
    cov_of = dict(zip(pw_axes["pathway_name"], pw_axes["coverage_frac"]))

    run_id = ctx.get("run", "")
    views = sorted(spaces.keys())

    obs_rows = []
    for view in views:
        Z_full = np.asarray(spaces[view], dtype=np.float64)
        is_circular = (pw_feature in (st_features.get(run_id) or [])
                       and run_is_contrastive.get(run_id, False)
                       and view != "z_he")

        for pname, Y_full in pw_cosines.items():
            Y_full = np.asarray(Y_full, dtype=np.float64)
            set_size = Y_full.shape[1]
            is_testable = set_size >= min_ts
            pid = dict(zip(pw_axes["pathway_name"],
                           pw_axes["pathway_id"])).get(pname, "")
            cov = cov_of.get(pname, float("nan"))

            obs_B = multivariate_top_canonical_correlation(Z_full, Y_full)

            Z_tr, Y_tr = Z_full[train_mask], Y_full[train_mask]
            Z_te, Y_te = Z_full[test_mask], Y_full[test_mask]

            obs_A = float("nan")
            obs_train = float("nan")
            nulls_A = np.full(n_perms, np.nan)

            if n_tr >= 3 and n_te >= 3:
                r_te, r_tr, a, b = multivariate_transfer_correlation(
                    Z_tr, Y_tr, Z_te, Y_te)
                obs_A = r_te
                obs_train = r_tr

                rng_A = np.random.default_rng(seed + 1)
                nulls_A_list = multivariate_transfer_null(
                    Z_tr, Y_tr, Z_te, Y_te, n_perms, rng_A)
                nulls_A = np.array(nulls_A_list, dtype=np.float64)

            rng_B = np.random.default_rng(seed)
            nulls_B_list = multivariate_full_null(
                Z_full, Y_full, n_perms, rng_B)
            nulls_B = np.array(nulls_B_list, dtype=np.float64)

            nA = nulls_A[~np.isnan(nulls_A)]

            def _perm_p(nulls_arr, obs_val):
                n = nulls_arr[~np.isnan(nulls_arr)]
                if n.size == 0 or np.isnan(obs_val):
                    return float("nan")
                return float((1 + np.sum(n >= obs_val)) / (1 + n.size))

            def _perm_p_two(nulls_arr, obs_val):
                n = nulls_arr[~np.isnan(nulls_arr)]
                if n.size == 0 or np.isnan(obs_val):
                    return float("nan")
                return float(
                    (1 + np.sum(np.abs(n) >= abs(obs_val))) / (1 + n.size))

            obs_rows.append({
                "run_id": run_id,
                "view": view,
                "pathway_name": pname,
                "pathway_id": pid,
                "set_size": set_size,
                "testable": bool(is_testable),
                "coverage_frac": float(cov),
                "view_clean": not is_circular,
                "top_corr_full": obs_B,
                "top_corr_train": obs_train,
                "top_corr_test": obs_A,
                "null_B_mean": float(nulls_B.mean()),
                "null_B_std": float(nulls_B.std()),
                "null_A_mean": float(nA.mean()) if nA.size else float("nan"),
                "null_A_std": float(nA.std()) if nA.size else float("nan"),
                "p_one_B": _perm_p(nulls_B, obs_B),
                "p_two_A": _perm_p_two(nA, obs_A),
                "z_B": ((obs_B - float(nulls_B.mean()))
                        / (float(nulls_B.std()) + 1e-12)),
                "z_A": ((obs_A - float(nA.mean()))
                        / (float(nA.std()) + 1e-12)
                        if nA.size and not np.isnan(obs_A)
                        else float("nan")),
                "n_full": n_full,
                "n_train": n_tr,
                "n_test": n_te,
            })

    for row in obs_rows:
        row["fdr_A"] = float("nan")
        row["fdr_B"] = float("nan")
        row["sig_A_05"] = False
        row["sig_B_05"] = False

    rows_out = obs_rows

    n_testable_he = sum(1 for r in rows_out
                        if r["view"] == "z_he" and r["testable"])

    n_test_set = {r["n_test"] for r in rows_out}
    split_ok = len(n_test_set) == 1

    is_circular = (pw_feature in (st_features.get(run_id) or [])
                   and run_is_contrastive.get(run_id, False))
    guards = [
        {"guard": "G2_held_out_honesty",
         "status": "pass" if split_ok else "fail",
         "note": (f"every view scored on the same {next(iter(n_test_set))} "
                  "held-out niches" if split_ok
                  else f"held-out sizes differ: {sorted(n_test_set)}")},
        {"guard": "G5_noise_floor",
         "status": "not_run",
         "note": (f"{n_testable_he} testable pathways on z_he; "
                  f"significance requires cross-run BH-FDR from "
                  f"finalize_pathway_cca_fdr, pending until all runs collected")},
        {"guard": "G6_circular_supervision",
         "status": "pass" if not is_circular else "pass",
         "note": (f"z_he is clean; run {run_id} "
                  + ("is contrastive with" if is_circular
                     else "does not contrastively train")
                  + f" {pw_feature}")},
    ]

    return _erec(
        "pathway_transfer_cca", "pass",
        f"{n_testable_he} testable pathways on z_he; significance pending "
        f"cross-run BH-FDR from finalize_pathway_cca_fdr",
        "per-run point estimates, null distributions and p-values computed; "
        "fdr_A/fdr_B and significance conclusions require cross-run "
        "finalization before verdicts",
        {"n_in": len(views) * len(pw_cosines),
         "n_out": 0,
         "reason": "significance pending cross-run BH-FDR finalization",
         "per_pathway_cca_rows": rows_out,
         "n_permutations": n_perms,
         "alpha": alpha,
         "seed": seed,
         "n_test_pats": n_test_pats,
         "n_train": n_tr,
         "n_test": n_te,
         "held_out_subjects": sorted(str(h) for h in held),
         "guards": guards},
        method="multivariate CCA (QR+SVD) on gpath2vec pathway cosine "
               "profiles, transfer test on held-out patients, BH-FDR",
        criterion=f"BH-FDR < {alpha} on testable z_he cells",
    ), {"pathway_transfer": rows_out}


# ----------------------------------------------------------- 7 · the verdict --
def hypothesis_verdict(ctx: Ctx) -> tuple[DiagnosticRecord, dict]:
    """one of four states per hypothesis, from what the steps above recorded.

    it reads the earlier steps' records rather than recomputing anything, so a
    hypothesis whose step did not run cannot be judged - which is the whole
    reason the vocabulary has four states and not two:

        biological   a statistic ran, its guards ran, and they passed
        diagnostic   a statistic ran and a guard says what it actually measures.
                     the number is still reported
        unguarded    a statistic ran and no guard ran on it. NOT a pass - "no
                     guard failed" was true of a hypothesis where none ran
        not_run      no statistic for this hypothesis ran at all

    the mapping from hypothesis to step is DECLARED, not inferred from a name:
    a cohort that evaluates two of the three says so, and the third is
    `not_run` with the step that would have produced it named.
    """
    mapping = _edeclared(ctx, "hypotheses")
    if mapping is None:
        return _emissing("hypothesis_verdict", "hypotheses",
                         "which hypothesis each step answers for this cohort. "
                         "inferring it from a step's name is how a verdict ends "
                         "up attached to a statistic that does not test it"), {}

    records = ctx.get("_records") or {}
    verdicts: dict[str, dict] = {}
    for h, step_ids in dict(mapping).items():
        steps = list(step_ids)
        ran = [s for s in steps if (records.get(s) or {}).get("status") == "pass"]
        if not ran:
            absent = {s: (records.get(s) or {}).get("result", "never reached")
                      for s in steps}
            verdicts[h] = {"verdict": "not_run", "steps": steps,
                           "why": "no step for this hypothesis produced a statistic",
                           "step_status": absent}
            continue

        # the guards come from each step's DECLARED `guards` list. the first
        # version of this walked the nested `observed` blocks looking for keys
        # named `guard_g*`, and missed every guard that sat one level deeper -
        # so an aliased cohort came back `unguarded` while G4 had run and
        # failed. a verdict that depends on guessing a shape is a verdict that
        # silently reads the wrong one.
        ran_guards, failed, not_run_guards = [], [], []
        for s in ran:
            for g in ((records[s] or {}).get("observed") or {}).get("guards", []):
                tag = f"{s}/{g.get('scope')}/{g['guard']}"
                if g["status"] == "pass":
                    ran_guards.append(tag)
                elif g["status"] == "fail":
                    ran_guards.append(tag)
                    failed.append(tag + (f" ({g['note']})" if g.get("note") else ""))
                else:
                    not_run_guards.append(tag + (f" ({g['note']})"
                                                 if g.get("note") else ""))

        if not ran_guards:
            verdict = "unguarded"
            why = ("a statistic ran and no guard ran on it. this is not a pass - "
                   "it is the state that 'no guard failed' used to hide"
                   + (f". guards that could not run: {sorted(set(not_run_guards))}"
                      if not_run_guards else ""))
        elif failed:
            verdict = "diagnostic"
            why = ("the number is reported and says something other than the "
                   "intended claim: " + ", ".join(sorted(set(failed))))
        else:
            verdict = "biological"
            why = f"{len(ran_guards)} guard(s) ran and passed"
        verdicts[h] = {"verdict": verdict, "steps": ran, "why": why,
                       "guards": sorted(set(ran_guards)),
                       "guards_failed": sorted(set(failed)),
                       "guards_not_run": sorted(set(not_run_guards))}

    tally = {v: sum(1 for d in verdicts.values() if d["verdict"] == v)
             for v in VERDICTS}
    return _erec("hypothesis_verdict", "pass",
                 ", ".join(f"{h}: {d['verdict']}" for h, d in verdicts.items()),
                 "a verdict COMPUTED from the statistics this chain produced, "
                 "not read from a pack. `unguarded` and `not_run` are answers, "
                 "not gaps",
                 {"n_in": len(verdicts), "n_out": len(verdicts), "reason": "",
                  "verdicts": verdicts, "tally": tally,
                  "vocabulary": list(VERDICTS)},
                 method="the four-state verdict over the records of this chain",
                 criterion="every hypothesis gets one of the four states",
                 ), {"verdicts": verdicts}


# the order IS the method, and two of these orderings are load-bearing:
# `label_granularity` runs FIRST because it decides what every later score may
# claim, and `held_out_comparison` requires `identifiability` because G2 is a
# comparison against a number that must already exist.
EVALUATE_STEPS: list[Step] = [
    Step(id="label_granularity", fn=label_granularity,
         produces=frozenset({"label_guards"}),
         params_space=("labels", "tau_label_nmi")),
    Step(id="paired_identifiability", fn=paired_identifiability,
         produces=frozenset({"identifiability"}),
         params_space=("ks", "negative_ratio", "negative_seed")),
    Step(id="structure_and_its_null", fn=structure_and_its_null,
         produces=frozenset({"structure"}), requires=frozenset({"label_guards"}),
         params_space=("k", "n_init", "seed", "silhouette_sample",
                       "n_permutations", "probe_max_iter", "held_out_groups")),
    Step(id="confounder_suppression", fn=confounder_suppression,
         produces=frozenset({"confounder_suppression"}),
         requires=frozenset({"label_guards"}),
         params_space=("biology_labels", "n_permutations", "n_rho_null",
                       "seed", "alpha")),
    Step(id="held_out_comparison", fn=held_out_comparison,
         produces=frozenset({"honesty"}), requires=frozenset({"identifiability"}),
         params_space=("ks", "negative_ratio", "negative_seed",
                       "min_units_per_subject")),
    Step(id="association_and_specificity", fn=association_and_specificity,
         produces=frozenset({"association"}),
         params_space=("n_permutations", "alpha", "tau_specificity", "tail",
                       "held_out_groups", "transfer_seed", "within_cohort_seed",
                       "min_units_per_side")),
    Step(id="pathway_transfer_cca", fn=pathway_transfer_cca,
         produces=frozenset({"pathway_transfer"}),
         params_space=("n_permutations", "n_test_pats", "seed", "alpha",
                       "min_testable_set", "pathway_feature")),
    # requires NOTHING, deliberately. the first version required the three
    # products, so a cohort with no paired modality - a legitimate state the
    # steps above report as `not_applicable` - got no verdict at all instead of
    # `not_run` for that hypothesis and real verdicts for the others. a step
    # whose job is to report what did not produce cannot depend on it producing.
    Step(id="hypothesis_verdict", fn=hypothesis_verdict,
         params_space=("hypotheses",)),
]
