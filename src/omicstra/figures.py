"""evidence figures, drawn from the same file a route resolves against.

the picture and the number cannot disagree, because they read one source. a
cohort that declares evidence gets its figures for free; nothing here names
tnbc-92 or any metric it happens to use.

one generic renderer covers every task family: a forest of candidates with their
intervals, the floor and reference drawn as lines, contraindicated methods greyed
and labelled. a multi-axis task gets one panel per axis, which is what makes an
escalation legible - you can see the two axes disagree.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from omicstra.routing import _blocks, load_routing_contract, load_routing_evidence
from omicstra.settings import settings

warnings.filterwarnings("ignore")

# fixed compartment -> colour, declared once. a palette assigned per figure by
# alphabetical index is not stable: a section missing one compartment shifts every
# colour after it, and the same tissue class ends up two different colours across
# two slides.
COMPARTMENT_COLORS = {
    "Tumor": "#C1443C", "in situ": "#E8877F",
    "Lymphoid nodule": "#5B3E96", "Lymphocyte": "#8C6BB1",
    "High TIL stroma": "#6B8FC9", "Low TIL stroma": "#A8C0E0",
    "Stroma cell": "#2E7D5B", "Acellular stroma": "#8FBF9F",
    "Fat tissue": "#E3C567", "Lactiferous duct": "#B08D3E",
    "Vessels": "#7A2E2E", "Necrosis": "#8A8A8A",
}
# colourblind-safe variant, built on Okabe-Ito. the palette above separates the
# compartments by hue family, which reads well in print but collapses under
# deuteranopia and at projector distance - Lymphoid nodule #5B3E96 and Lymphocyte
# #8C6BB1 are one colour on a wall, and they are the pair the TLS work depends on.
# here the two lymphoid calls are separated on LIGHTNESS within one hue, and the
# six compartments the talk uses are six distinct Okabe-Ito anchors.
# the eight Okabe-Ito anchors go to the STRUCTURES a pathologist names in cancer -
# perineural invasion, TLS, TILs, angiogenesis, DCIS, necrosis - not to the classes
# with the most niches. Nerve is 4 niches in the whole cohort and still earns black,
# because perineural invasion is a staging feature and rarity is not the axis.
# the stroma variants, which are 48% of all calls, take the leftover tints.
COMPARTMENT_COLORS_CB = {
    "Tumor": "#D55E00", "in situ": "#E69F00",
    "Lymphoid nodule": "#0072B2", "Lymphocyte": "#56B4E9",
    "High TIL stroma": "#009E73", "Vessels": "#CC79A7",
    "Nerve": "#000000", "Necrosis": "#999999", "Fat tissue": "#F0E442",
    "Low TIL stroma": "#7FD4B8", "Stroma cell": "#A87C9F",
    "Acellular stroma": "#E8B4D0", "Lactiferous duct": "#B8A200",
    "Tumor region": "#8C3B00", "Heterologous elements": "#6D6D6D",
}
LEAD, MUTED, BAD, RULE = "#4A5D7E", "#AAAAAA", "#B5544F", "#2E7D5B"


def evidence_dir(project_id: str | None = None) -> Path:
    p = settings.resolve(settings.runs_dir) / (project_id or "unrouted") / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def evidence_plot(task_id: str, project_id: str | None = None,
                  out: Path | None = None) -> tuple[Path, str]:
    """render this task's evidence. returns (png path, one-line caption)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    ev = load_routing_evidence(project_id)
    contract = load_routing_contract()
    task = (ev.get("tasks") or {}).get(task_id)
    if not task:
        raise KeyError(f"this cohort records no evidence for {task_id!r}")

    fam = next((f for f in contract["task_families"] if f["id"] == task_id), {})
    names = {k: v for k, v in (ev.get("method_names") or {}).items() if k != "note"}
    banned = {c["method"] for c in ev.get("contraindications", [])
              if task_id in c.get("applies_to", [])}
    blocks = _blocks(task)

    sns.set_theme(style="ticks", palette="Set2", context="notebook")
    h = max(2.6, 0.34 * max(len(b["candidates"]) for b in blocks) + 1.5)
    fig, axes = plt.subplots(1, len(blocks), figsize=(7.2 * len(blocks), h), squeeze=False)

    for ax, blk in zip(axes[0], blocks):
        hib = blk.get("higher_is_better", True)
        cands = sorted(blk["candidates"], key=lambda c: c["value"], reverse=not hib)
        # the sort already puts the best candidate last on either polarity, so the
        # leader is cands[-1] in both cases. keying it off `hib` a second time
        # inverts it, and marks the worst method as the winner on a
        # lower-is-better axis.
        best = next((c for c in reversed(cands) if c["id"] not in banned), None)
        for i, c in enumerate(cands):
            ban = c["id"] in banned
            lead = (not ban) and c is best
            col = MUTED if ban else (LEAD if lead else "#6E7F9E")
            if c.get("ci"):
                ax.plot(c["ci"], [i, i], lw=6, alpha=.3, color=col, solid_capstyle="butt")
            ax.plot(c["value"], i, "o", ms=8, color=col)
        for bar, style, lbl in ((blk.get("floor"), (BAD, "--"), "floor"),
                                (blk.get("reference"), (RULE, ":"), "reference")):
            if bar and bar.get("value") is not None:
                ax.axvline(bar["value"], color=style[0], ls=style[1], lw=1.5)
                # y is inverted below, so -0.55 is above the top row - keeps the
                # label clear of the tick labels and the x-axis
                ax.text(bar["value"], -0.55, f" {lbl}: {bar['id']} {bar['value']}",
                        color=style[0], fontsize=8, va="bottom")
        ax.set_yticks(range(len(cands)))
        ax.set_yticklabels([names.get(c["id"], c["id"]) + ("  (contraindicated)"
                                                          if c["id"] in banned else "")
                            for c in cands], fontsize=8.5)
        ax.set_xlabel(blk.get("metric", "")[:88] + ("…" if len(blk.get("metric", "")) > 88 else ""),
                      fontsize=8.5)
        ax.invert_yaxis()
        sns.despine(ax=ax)

    title = f"{task_id}  ·  {fam.get('hypothesis', '')}"
    if len(blocks) > 1:
        title += "   — two axes; they must agree or the task escalates"
    fig.suptitle(title, fontsize=11.5)
    if ev.get("scope_statement"):
        fig.text(.5, .005, ev["scope_statement"], ha="center", fontsize=7.5,
                 style="italic", color="#555")
    fig.tight_layout(rect=[0, .04, 1, .94])

    out = out or evidence_dir(ev.get("project_id") or project_id) / f"{task_id}.png"
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)

    cap = (f"{fam.get('asks', task_id)}. "
           + ("intervals from " if any(c.get("ci") for b in blocks for c in b["candidates"])
              else "point estimates from ")
           + str(task.get("source", "the cohort's evidence"))[:150])
    return out, cap