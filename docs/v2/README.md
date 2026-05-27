# docs/v2 - v3 gpath2vec retrain reports

these are the **manuscript-current** reports. they replace the v1 reports under [`docs/v1/`](../v1/), which are preserved unchanged as the historical baseline.

**v1 vs v2:** the alignment grid was retrained from scratch on the v3 gpath2vec build (`fisher_madmean_low_dim512_e5_s1234`, sha-locked, reproducible) replacing the v1 build the original runs used. only the ST-side pathway feature changed; novae, virchow2, mc_weights unchanged. all H1/H2/H3 numbers regenerated on the v3 grid.

| doc | status | notes |
|---|---|---|
| [`tnbc92_results_summary.md`](tnbc92_results_summary.md) | **v3-current** | H1/H2/H3 result tables on the v3 grid |
| [`tnbc92_routing_matrix.md`](tnbc92_routing_matrix.md) | **v3-current** | task-conditional routing rule with v3 evidence |
| `tnbc92_three_hypotheses.md` | not redone | framework brief - structure unchanged by data version; v1 lives at [`../v1/tnbc92_three_hypotheses.md`](../v1/tnbc92_three_hypotheses.md). if the manuscript wants a v3 redo of the long brief, that is a separate writing pass. |

**writing voice.** tables and numbers in these docs are auto-populated from `runs/tnbc-92_v3/eval/*`. prose-level interpretation is preserved from the v1 framework or marked `<!-- TODO: voice pass -->` where v3 differs from v1; the manuscript author writes the final prose. see memory `feedback_writing_voice`.

**v3 retrain summary.** all 8 steps of `projects/tnbc-92/v3_phase2_plan.md` landed 2026-05-21 to 2026-05-26. headline reproduces v1: R4 wins H1 (AUC 0.859) + H3 cross-patient (gpath2vec arm, 4/4 testable pathways), R6/R1 win niche-level biology coherence (mc_megacluster ARI). see memory `project_v3_retrain_result`.
