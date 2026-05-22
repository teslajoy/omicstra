# 05_summary_umaps.ipynb — accuracy review

review date: 2026-05-15. scope: concrete code-vs-markdown-vs-data audit of
`notebooks/final/05_summary_umaps.ipynb` and its use in the H1/H2/H3 answers.
every quantitative claim was checked against the artifacts the notebook actually
loads (`runs/tnbc-92/eval/H3/pathway_cca/dag_full/`, `runs/tnbc-92/R1/embeddings_test.parquet`,
`data/embeddings/biological_signals/niche_aucell_5targets.parquet`).

line numbers refer to the `jupyter nbconvert --to script` rendering of the notebook.

---

## summary

the code is mostly sound and reproducible. the research narrative in §10 / §11 / §13
is materially wrong: blockers 1-2 make the H3 headline section caption ~10% while the
code plots ~79%, and invert the commit-3.6 conclusion that classical baselines are
competitive at DAG resolution. these are the load-bearing claims for the H3 answer.

severity legend: 🔴 blocker · 🟠 major · 🟡 minor

---

## 🔴 blocker 1 — §11 markdown contradicts the numbers §11's own code plots

- **code** (`In[14]`, `sig_summary`): `n_tests` computed *per run* = **1185** (= 395 nodes × 3 views),
  `pct_sig = n_sig / 1185`.
- **bar chart actually rendered:** R4 **78.8%**, B2 74.2%, B1 70.2%, B3 64.4%,
  R6 36.9%, R1 27.1%, R2 23.0%, R3 18.4%.
- **markdown (line 700) claims:** "9480 total cells per run = 395 DAG nodes × 3 views ×
  ~8 cell-types … R4 dominates at 9.9%; B2 9.3%, B1 8.8% … R3 2.3%".

three independent errors in one sentence:

1. there is **no cell-type dimension** in `per_node_cca.parquet` (verified columns:
   run_id/view/node_id/obs_*/null_*/p_*/z_*/fdr_*/sig_*). the "~8" the author called
   "cell-types" is the **8 runs**.
2. **9480 is the total across all 8 runs**, not "per run". per run = 1185.
3. the "9.9% / 9.3% …" figures are `n_sig / 9480` (per-run sig ÷ all-runs total) —
   a denominator that matches neither the code nor any valid grid. the reader sees a
   bar at ~79% while the caption says ~10%.

§13 line 781 repeats the same wrong numbers ("R4 wins 9.9%; … R6/R1/R2/R3 fall to 2-5%").

root cause traced: `memory/project_h3_dag_pathway_findings.md` carries both the wrong
"934/9480 = 9.9%" headline table and the correct "78.8%" in its description; the
notebook copied the wrong one.

**fix:** rewrite line 700 / line 781 to "1185 DAG-tests per run (395 nodes × 3 views);
R4 78.8%, B2 74.2%, B1 70.2%, B3 64.4%, R6 36.9%, R1 27.1%, R2 23.0%, R3 18.4%".

verification:

```
per_node_cca rows: 9480 total; per run: B1..R6 = 1185 each; 395 nodes × 3 views = 1185
no column matching /cell/ exists
per-run sig_A_two_05: R4 934/1185=78.8%, B2 879=74.2%, B1 832=70.2%, B3 763=64.4%,
                      R6 437=36.9%, R1 321=27.1%, R2 273=23.0%, R3 218=18.4%
```

## 🔴 blocker 2 — research framing reversed by blocker 1

§11 says "R4 **dominates**"; §13 closing says "classical baselines lose H1 or
**saturate non-discriminatingly**". true option-A DAG numbers are R4 78.8 vs
**B2 74.2 vs B1 70.2** — classical baselines are competitive, which is exactly the
commit-3.6 finding ("classical catch up significantly at fine pathway resolution;
the 'classical lose H3' framing holds for coarse 5-parent, **NOT** DAG"). the notebook
narrative reverts to the strong story that finding explicitly retracted. the §11/§12
panels (option A, cross-patient) show classical near-parity, contradicting §13's own prose.

**fix:** soften §11 "dominates" → "wins, with classical baselines (B2 74%, B1 70%)
competitive at DAG resolution"; correct §13 closing to match.

## 🟠 major 3 — §10 named hits don't match the panels the code renders

§10 markdown + intro line 38 name "**collagen biosynthesis (z=5.28)**" as a headline hit.
actual R4 top-8 by `z_A` (what `In[12]` selects and `In[13]` plots):

| rank | node | parent | z_A |
|---|---|---|---|
| 1 | Co-inhibition by BTLA | Immune | 6.92 |
| 2 | Interleukin receptor SHC signaling | Immune | 6.80 |
| 3 | Integrin cell surface interactions | ECM | 6.21 |
| 4 | ECM proteoglycans | ECM | 5.73 |
| 5 | Regulation of IFNA/IFNB signaling | Immune | 5.71 |
| 6 | Extracellular matrix organization | ECM | 5.49 |
| 7 | Degradation of the extracellular matrix | ECM | 5.47 |
| 8 | Generation of second messenger molecules | Immune | 5.45 |

BTLA / SHC / integrin are correct; "collagen biosynthesis" is **not in the top-8** —
the panel shows ECM-proteoglycans / IFN / ECM-organization / etc. instead.

**fix:** replace "collagen biosynthesis (5.28)" with "ECM proteoglycans (5.73),
IFN-α/β regulation (5.71)" in §10 markdown + intro line 38.

## 🟠 major 4 — §13 line 777 internally contradicts §7 and is numerically fabricated

- §7 line 422: "12 distinct classes … test cohort" — **correct** (verified 12).
- §13 line 777: "**90/260 subarrays covered, 15 classes populated**" — both wrong.
  verified: **13/38** subarrays carry any compartment; **12** classes in test; the
  test cohort is **38** subarrays total, not 260. "90/260" and "15" are stale/fabricated.
- coverage "~35%, 15,953/45,661" in §7 is **correct** (verified 34.9%).

**fix:** §13 line 777 → "13/38 subarrays covered, 12 classes populated".

## 🟡 minor 5 — "5% FDR floor" hline is statistically wrong and now off-scale

`In[14]`: `fig.add_hline(y=5, annotation_text='5% FDR floor')`. under BH-FDR<0.05 the
global-null expected significant *fraction* is **not** 5% — BH controls FDR among
rejections, not the rejection rate; 5% is the *uncorrected*-p expectation. with the
real y-axis at 18-79% (post-blocker-1) a line at y=5 is also visually meaningless.

**fix:** remove the hline, or relabel it as a permutation/uncorrected reference only
if the axis warrants.

## 🟡 minor 6 — hardcoded stats not reproduced in-notebook

§5/§6 patient-z (B1 122.13, R4 22.97) and §9 (15/15, z 3.8-5.7) are asserted only in
markdown. traced to `projects/tnbc-92/current_report.md` and
`docs/tnbc92_routing_matrix.md` — internally **consistent**, but no cell recomputes
them, against CLAUDE.md's "notebooks rerunnable top-to-bottom" expectation.

**fix:** cite the source artifact inline, or add a small read-back cell.

## 🟡 minor 7 — §12 per-parent attribution is single-parent

`In[15]` merges `dag_cca` on `parent_name` (one primary parent), but `dag_metadata`
has `all_parents` (DAG nodes can have multiple parents). defensible (avoids
double-counting) but the §12 markdown should state "each node assigned to its primary
parent" so the heatmap is not read as a partition of all DAG-tests.

---

## verified correct — no action

- test geometry: 45,661 niches / 14 patients / 38 subarrays ✓ (intro accurate).
- archetype = f(patient): **14/14** test patients single-archetype ✓; §5 computes
  NMI **live** — reproducible.
- compartment coverage 34.9%, 15,953/45,661, 12 classes ✓ (§7 body correct; only
  §13's restatement is wrong).
- §10 artifacts real; BTLA 6.92 / SHC 6.80 / integrin 6.21 exact ✓.
- UMAP cache keyed by hyperparam hash, L2-norm before cosine UMAP, global stable
  color map — all sound.
- gpath2vec → "Reactome DAG" label corrected earlier this session
  (commit `d74dde3` introduced the wrong "gpath2vec hierarchy" wording in §10 /
  intro line 38 / §13 line 780; analysis itself was always Reactome DAG AUCell + CCA) ✓.

---

## net assessment for the H3 answer

§11 / §13 are the load-bearing claims for H3. as written they state ~10% while the
code plots ~79%, and invert the "classical competitive at DAG resolution" conclusion.
H1 / H2 sections (§3-§8) are factually accurate; the construct-validity argument in
§4-§5 is well-supported and reproducible. fix blockers 1-2 and majors 3-4 before this
notebook is cited as the H3 visualization deliverable.