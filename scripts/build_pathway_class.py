"""roll every Reactome pathway up to its top-level class.

figures 5 and 6 both depend on this. the point is to state a pathway's identity
at a level a bench audience can hear - IMMUNE SYSTEM, EXTRACELLULAR MATRIX
ORGANISATION - rather than reading out a depth-5 leaf name and hoping.

Reactome is a DAG, not a tree, so a leaf can descend from several roots. the
choice matters and is made by rule rather than by judgement:

  keep ALL top-level ancestors, and drop the Disease branch.

dropping Disease is what removes the Dengue / HIV / Influenza leaves that
contaminate every enrichment on this cohort. those leaves host real host-response
machinery, which is why they score, but they are filed under Infectious disease
and they are not what anyone means by the result. excluding the branch by rule is
honest; blacklisting individual pathway names is not.

`n_classes` is kept because a node with several roots is itself informative - a
pathway that is simultaneously Immune System and Signal Transduction is doing
different work from one that is only Immune System.

this also settles something the report gets wrong. its "five named parents" are
not one level: Immune System, Extracellular matrix organization, Cell Cycle and
Programmed Cell Death are Reactome roots, but TGF-beta signalling is not - it
sits under Signal Transduction. "wins every parent" is therefore four roots
against one mid-level node. `depth` in this file gives a defensible level at
which to state the claim instead.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import click
import pandas as pd

warnings.filterwarnings("ignore")
sys.setrecursionlimit(20000)

ROOT = Path(__file__).resolve().parents[1]
REACTOME = ROOT / "knowledge/reactome"
DISEASE_ROOT = "R-HSA-1643685"          # Disease
OUT = ROOT / "knowledge/reactome/pathway_class.tsv"


def _hierarchy():
    rel = pd.read_csv(REACTOME / "ReactomePathwaysRelation.txt", sep="\t",
                      header=None, names=["parent", "child"])
    rel = rel[rel.parent.str.startswith("R-HSA")]
    up: dict[str, list[str]] = {}
    for p, c in rel.itertuples(index=False):
        up.setdefault(c, []).append(p)
    roots = set(rel.parent) - set(rel.child)

    nm = pd.read_csv(REACTOME / "ReactomePathways.txt", sep="\t", header=None,
                     names=["stid", "name", "species"])
    nm = nm[nm.species == "Homo sapiens"].drop_duplicates("stid")
    return up, roots, dict(zip(nm.stid, nm.name))


def _gene_sets() -> dict[str, int]:
    """gene-set sizes, from whichever cached set is available."""
    import json
    p = ROOT / "runs/tnbc-92/eval/H3/pathway_cca/dag_full/dag_gene_sets.json"
    if p.exists():
        return {k: len(v) for k, v in json.loads(p.read_text()).items()}
    return {}


@click.command()
@click.option("--out", type=click.Path(path_type=Path), default=OUT)
@click.option("--keep-disease", is_flag=True, default=False,
              help="keep the Disease branch. off by default, see docstring")
def main(out, keep_disease):
    """emit pathway_class.tsv."""
    up, roots, names = _hierarchy()
    sizes = _gene_sets()
    drop = set() if keep_disease else {DISEASE_ROOT}

    anc_cache: dict[str, set[str]] = {}
    dep_cache: dict[str, int] = {}

    def ancestors(n, seen=None):
        if n in anc_cache:
            return anc_cache[n]
        seen = seen or set()
        out_ = {n} if n in roots else set()
        for p in up.get(n, []):
            if p not in seen:
                out_ |= ancestors(p, seen | {p})
        anc_cache[n] = out_
        return out_

    def depth(n, seen=None):
        """shortest hop count to any root. reported, never used to pick a class."""
        if n in dep_cache:
            return dep_cache[n]
        if n in roots:
            dep_cache[n] = 0
            return 0
        seen = seen or set()
        ds = [depth(p, seen | {p}) + 1 for p in up.get(n, []) if p not in seen]
        dep_cache[n] = min(ds) if ds else -1
        return dep_cache[n]

    rows = []
    for stid in sorted(names):
        anc = ancestors(stid)
        kept = sorted(anc - drop, key=lambda r: names.get(r, r))
        rows.append({
            "r_hsa": stid,
            "name": names[stid],
            "top_class": "; ".join(names.get(r, r) for r in kept) or "(none)",
            "depth": depth(stid),
            "n_classes": len(kept),
            "gene_set_size": sizes.get(stid, ""),
            "disease_only": len(anc) > 0 and len(kept) == 0,
        })
    d = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, sep="\t", index=False)

    click.echo(f"wrote {out.relative_to(ROOT)}   {len(d)} pathways")
    click.echo(f"  roots: {len(roots)}   dropped as Disease-only: {int(d.disease_only.sum())}")
    click.echo(f"  multi-class nodes: {int((d.n_classes > 1).sum())} "
               f"({100 * (d.n_classes > 1).mean():.0f}%)")
    click.echo("\n  top classes by membership:")
    ex = d[d.n_classes > 0].top_class.str.split("; ").explode().value_counts()
    for k, v in ex.head(8).items():
        click.echo(f"    {v:5d}  {k}")
    click.echo("\n  the report's five 'parents':")
    for q in ["Immune System", "Extracellular matrix organization", "Cell Cycle",
              "Programmed Cell Death", "Signaling by TGF-beta Receptor Complex"]:
        r = d[d.name == q]
        if len(r):
            w = r.iloc[0]
            click.echo(f"    {w['name'][:42]:<42} depth {w.depth}  ->  {w.top_class}")


if __name__ == "__main__":
    main()