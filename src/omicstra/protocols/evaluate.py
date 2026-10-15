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


# --- admissibility: a cell is a value AND whether it may be compared ---------
#
# the matrix was question x WINNER - one row per question, the ranked value of
# whichever method came first. so a reader saw ten numbers per question with
# nothing marking the inadmissible ones, which is an invitation to compare cells
# that are not comparable.
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


def cell(pack: dict, task_id: str, candidate: dict) -> dict:
    """one matrix cell: the value, whether it may be compared, and why not.

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
        cells = [cell(pack, task_id, c) for c in (t.get("candidates") or [])]
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
