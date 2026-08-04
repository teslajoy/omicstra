"""UMAP of the 238 TLS niches in gpath2vec space, coloured four ways.

the clustermap in `h3_tls_subtypes.py` shows the three sub-types separating on
pathways that were selected, by ANOVA F, across those same three clusters. a
UMAP coloured by cluster has the same problem one step earlier: the clusters
were cut in this exact space, so three separated blobs are guaranteed by
construction and carry no evidence.

so three of the four panels colour by something the clustering never saw:

    patient_id        is the geometry patient-driven? the H2-UMAP question,
                      asked where it is actually answerable
    mc_megacluster    does gpath2vec space reproduce Wang's NMF classes, or
                      cross-cut them? this is the claim under test
    tls score         is the axis just "more TLS-like", or orthogonal to
                      intensity?
    cluster           the Ward cut itself - descriptive, labelled as such

why this is not done on the shared-space run UMAPs, which would be the real
out-of-space test: only 24 of the 238 TLS niches fall in the 35,594 held-out
test niches, across 3 patients. eight points per sub-type cannot support it.

usage:
    ./venv/bin/python scripts/h3_tls_subtype_umap.py
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage

warnings.filterwarnings("ignore")

# identities from scripts/extract_mc14_differential_markers.R
MC_NAMES = {
    1: "CAF / desmoplastic stroma", 2: "inflamed / IFN epithelium",
    3: "transcriptionally active", 4: "basal-like epithelium",
    5: "plasma cell zone", 6: "apocrine / secretory",
    7: "basal myoepithelial", 8: "proliferating stem-like",
    9: "plasma cells + IFN", 10: "fibroblastic stroma, immune-mixed",
    11: "mitochondrial-high (QC flag)", 12: "B-cell follicle / germinal centre",
    13: "secretory + FDC", 14: "ribosomal-high",
}
SUBTYPE_NAMES = {1: "C1 transcription / chromatin",
                 2: "C2 neutrophil / innate",
                 3: "C3 translation / effector"}


def load_tls_niches(niches_dir: Path) -> pd.DataFrame:
    """the 238 Lymphoid-nodule niches. same filter as h3_tls_subtypes.py."""
    parts = [
        pd.read_parquet(p, columns=["subarray", "patient_id", "compartment",
                                    "mc_megacluster", "tls"]).reset_index()
        for p in sorted(niches_dir.glob("*.parquet"))
    ]
    df = pd.concat(parts, ignore_index=True)
    tls = df[df["compartment"] == "Lymphoid nodule"].copy()
    tls["niche_id"] = "cluster_" + tls["subarray"] + "::" + tls["spot_id"]
    tls["mc_megacluster"] = tls["mc_megacluster"].astype(int)
    return tls.reset_index(drop=True)


def cluster_niches(emb: np.ndarray, k: int = 3) -> np.ndarray:
    """the published cut, reproduced. ward on L2-normed embeddings, deterministic.

    ||x - y||^2 = 2(1 - cos) when x and y are unit norm, so euclidean ward here
    is a cosine criterion. no seed involved - this is not a re-derivation, it
    recovers the same labels the clustermap used.
    """
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return fcluster(linkage(emb_n, method="ward", metric="euclidean"),
                    t=k, criterion="maxclust")


def agreement_stats(tls: pd.DataFrame) -> dict:
    """turn the three non-circular panels into numbers.

    the eye can be talked into anything on 238 points, so each panel gets the
    statistic that would falsify what it appears to show.
    """
    from scipy.stats import kruskal
    from sklearn.metrics import adjusted_rand_score as ari
    from sklearn.metrics import normalized_mutual_info_score as nmi

    groups = [g.tls.to_numpy() for _, g in tls.groupby("subtype")]
    _, p = kruskal(*groups)
    return {
        "ari_subtype_mc": ari(tls.subtype, tls.mc_megacluster),
        "nmi_subtype_mc": nmi(tls.subtype, tls.mc_megacluster),
        "ari_subtype_patient": ari(tls.subtype, tls.patient_id),
        "nmi_subtype_patient": nmi(tls.subtype, tls.patient_id),
        "nmi_mc_patient": nmi(tls.mc_megacluster, tls.patient_id),
        "tls_medians": "  ".join(f"C{i + 1} {np.median(g):.2f}" for i, g in enumerate(groups)),
        "tls_p": p,
    }


def _panel(ax, xy, values, title, *, palette=None, continuous=False, legend=True):
    if continuous:
        s = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap="viridis", s=26,
                       linewidths=0.3, edgecolors="white")
        plt.colorbar(s, ax=ax, fraction=0.046, pad=0.03)
    else:
        order = sorted(pd.unique(values), key=str)
        colors = palette or sns.color_palette("Set2", len(order))
        for c, v in zip(colors, order):
            m = np.asarray(values) == v
            ax.scatter(xy[m, 0], xy[m, 1], s=26, color=c, linewidths=0.3,
                       edgecolors="white", label=f"{v}  (n={int(m.sum())})")
        if legend:
            ax.legend(fontsize=7, frameon=False, loc="best", handletextpad=0.2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


@click.command()
@click.option("--niches-dir", default="data/embeddings/niches_v3",
              help="per-subarray niche parquets.")
@click.option("--embeddings", default="data/embeddings/gpath2vec/"
              "fisher_madmean_low_dim512_e5_s1234/fisher_madmean_low_embeddings.pkl",
              help="gpath2vec node embeddings, the sha-locked v3 build.")
@click.option("--out", default="runs/tnbc-92_v3/eval/figures/h3_tls_subtype_umap.png")
@click.option("--k", default=3, help="sub-types. matches the published cut.")
@click.option("--seed", default=42)
def main(niches_dir: str, embeddings: str, out: str, k: int, seed: int) -> None:
    """UMAP the TLS niches, coloured by patient, mc, TLS score and sub-type."""
    import umap

    tls = load_tls_niches(Path(niches_dir))
    click.echo(f"TLS niches: {len(tls)} across {tls.patient_id.nunique()} patients")

    with open(embeddings, "rb") as fh:
        node_emb = pickle.load(fh)
    keep = [i for i, n in enumerate(tls.niche_id) if n in node_emb]
    if len(keep) < len(tls):
        click.echo(f"  {len(tls) - len(keep)} niche(s) absent from the embedding, dropped")
    tls = tls.iloc[keep].reset_index(drop=True)
    emb = np.stack([np.asarray(node_emb[n], dtype=np.float32) for n in tls.niche_id])
    click.echo(f"embeddings: {emb.shape}")

    tls["subtype"] = cluster_niches(emb, k)
    sizes = tls.subtype.value_counts().sort_index()
    click.echo("sub-type sizes: " + ", ".join(f"C{i}={n}" for i, n in sizes.items())
               + "   (published: C1=138, C2=23, C3=77)")

    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                   random_state=seed).fit_transform(
                       emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9))

    stats = agreement_stats(tls)

    sns.set_theme(style="ticks", palette="Set2", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))

    _panel(axes[0, 0], xy, tls.patient_id.astype(str),
           f"patient  ({tls.patient_id.nunique()} patients)\n"
           f"NMI(sub-type, patient) = {stats['nmi_subtype_patient']:.3f}"
           f"   vs NMI(mc, patient) = {stats['nmi_mc_patient']:.3f}",
           palette=sns.color_palette("husl", tls.patient_id.nunique()), legend=False)
    _panel(axes[0, 1], xy, tls.mc_megacluster.map(lambda m: f"mc{m:02d} {MC_NAMES.get(m, '')}"),
           "Wang mc_megacluster\n"
           f"ARI(sub-type, mc) = {stats['ari_subtype_mc']:+.3f}  - cross-cuts, does not reproduce",
           palette=sns.color_palette("tab10", tls.mc_megacluster.nunique()))
    _panel(axes[1, 0], xy, tls.tls.to_numpy(),
           "TLS gene-signature score\n"
           f"medians {stats['tls_medians']}  Kruskal-Wallis p = {stats['tls_p']:.1e}"
           "  - real, but the IQRs overlap", continuous=True)
    _panel(axes[1, 1], xy, tls.subtype.map(SUBTYPE_NAMES),
           "gpath2vec sub-type (Ward, k=3)\n"
           "DESCRIPTIVE - cut in this same space, so it separates by construction",
           palette=sns.color_palette("Set1", k))

    fig.suptitle(
        f"TLS niches in gpath2vec space - {len(tls)} 'Lymphoid nodule' niches, "
        f"UMAP(cosine, n_neighbors=15, seed={seed})\n"
        "three panels colour by variables the clustering never saw; the fourth is "
        "the clustering itself", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    click.echo(f"wrote {out_path}")

    click.echo("\ndoes the cut reproduce mc, or cross-cut it?")
    click.echo(f"  ARI(sub-type, mc)       {stats['ari_subtype_mc']:+.3f}"
               f"   NMI {stats['nmi_subtype_mc']:.3f}   1 = same partition")
    click.echo("is the cut patient-driven?")
    click.echo(f"  NMI(sub-type, patient)   {stats['nmi_subtype_patient']:.3f}")
    click.echo(f"  NMI(mc, patient)         {stats['nmi_mc_patient']:.3f}"
               "   <- Wang's own label is the patient-entangled one")
    click.echo("is the cut just TLS intensity?")
    click.echo(f"  medians {stats['tls_medians']}   Kruskal-Wallis p={stats['tls_p']:.1e}")

    # what the panels are worth reading against, as numbers rather than eyeballs
    click.echo("\nper sub-type composition:")
    for st, g in tls.groupby("subtype"):
        top_mc = g.mc_megacluster.value_counts(normalize=True).head(3)
        click.echo(f"  {SUBTYPE_NAMES[st]:32} n={len(g):3}  patients={g.patient_id.nunique():2}"
                   f"  median TLS={g.tls.median():.2f}")
        for m, f in top_mc.items():
            click.echo(f"      mc{m:02d} {MC_NAMES.get(m, '')[:34]:36} {f:.0%}")


if __name__ == "__main__":
    main()