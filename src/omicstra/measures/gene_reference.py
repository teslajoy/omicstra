"""which gene annotation a counts matrix is on, from the matrix alone.

why it is worth measuring rather than declaring
-----------------------------------------------
gene identifiers are not stable labels. HGNC renames on the order of a few
percent of symbols a year, and an Ensembl gene's version suffix increments
whenever its structure changes between releases - so two matrices carrying the
same ENSG stem at `.14` and `.17` describe different exon sets and therefore
different counts. this is why the field's convention is Ensembl-native storage
with symbols reserved for display, and why comparing or stacking matrices built
against different references is a confounder rather than a nuisance.

it bites twice in this package. symbol-matching encoders (novae's zero-shot path
resolves `var_names` against a fixed vocabulary) see a gene count that depends on
the mapping release, not on the tissue. and any cross-section statistic assumes
one reference; the second cohort pools three studies and does not have one.

what is inferable here and what is not
--------------------------------------
the exact release is NOT recoverable from a matrix - a release is a GTF, and
nothing in an .h5ad points at one. what is recoverable is a fingerprint that
distinguishes references and flags the cases that matter: the identifier space,
whether versions are carried, the version distribution, gene count, duplicate
stems, and MT gene presence. matching that fingerprint to a named release needs a
declared GTF, which is a cohort's to supply.
"""
from __future__ import annotations

import re
from collections import Counter

ENSG = re.compile(r"^ENSG\d{11}(\.\d+)?$")
ENST = re.compile(r"^ENST\d{11}(\.\d+)?$")
# MT- prefixed symbols and the 13 protein-coding mitochondrial genes. their
# fraction is the standard per-spot QC covariate, so their absence changes which
# EDA checks are answerable at all.
MT_SYMBOL = re.compile(r"^MT-")
# the 13 protein-coding genes of the human mitochondrial genome, by symbol so the
# list is checkable. count, not membership, is what matters: 13 means the MT
# genome survived the reference and pct_mt is computable; fewer means it was
# filtered upstream and pct_mt is unanswerable, not zero.
MT_ENSG = {
    "ENSG00000198888": "MT-ND1", "ENSG00000198763": "MT-ND2",
    "ENSG00000198804": "MT-CO1", "ENSG00000198712": "MT-CO2",
    "ENSG00000228253": "MT-ATP8", "ENSG00000198899": "MT-ATP6",
    "ENSG00000198938": "MT-CO3", "ENSG00000198840": "MT-ND3",
    "ENSG00000212907": "MT-ND4L", "ENSG00000198886": "MT-ND4",
    "ENSG00000198786": "MT-ND5", "ENSG00000198695": "MT-ND6",
    "ENSG00000198727": "MT-CYB",
}
assert len(MT_ENSG) == 13


def identifier_space(names) -> str:
    """`ensembl_versioned` | `ensembl` | `ensembl_transcript` | `symbol` | `mixed` | `unknown`.

    majority vote, because real matrices carry stragglers: HEST objects keep
    clone-based names like `FO538757.1` beside symbols, and a strict rule would
    call a symbol matrix unknown on its first row.
    """
    names = [str(n) for n in names]
    if not names:
        return "unknown"
    ensg = [n for n in names if ENSG.match(n)]
    enst = [n for n in names if ENST.match(n)]
    half = len(names) / 2
    if len(enst) > half:
        return "ensembl_transcript"
    if len(ensg) > half:
        return "ensembl_versioned" if sum("." in n for n in ensg) > len(ensg) / 2 else "ensembl"
    if ensg:
        return "mixed"
    # not Ensembl and not empty: symbols, or something nobody has seen. a symbol
    # is upper-case with digits, hyphens and dots allowed (HLA-DRB1, FO538757.1).
    ok = sum(bool(re.match(r"^[A-Z0-9][A-Z0-9._@-]*$", n)) for n in names)
    return "symbol" if ok > half else "unknown"


def fingerprint(names) -> dict:
    """what distinguishes one annotation from another, from the names alone."""
    names = [str(n) for n in names]
    space = identifier_space(names)
    stems = [n.split(".")[0] for n in names]
    versions = Counter(n.split(".")[1] for n in names if "." in n and n.startswith("ENS"))
    dup = [s for s, c in Counter(stems).items() if c > 1]

    mt = (sum(bool(MT_SYMBOL.match(n)) for n in names) if space == "symbol"
          else len(set(MT_ENSG) & set(stems)))
    return {
        "identifier_space": space,
        "n_genes": len(names),
        "n_unique_stems": len(set(stems)),
        "n_duplicate_stems": len(dup),
        "duplicate_stems": sorted(dup)[:10],
        "versioned": space == "ensembl_versioned",
        # the spread, not the mode. one release gives many versions across genes,
        # so a wide spread is normal and a narrow one on a large matrix is not.
        "version_values": len(versions),
        "version_top": versions.most_common(5),
        "n_mitochondrial": int(mt),
        "mt_complete": mt >= 13,
    }


def coverage(names, vocabulary, *, strip_version: bool = True) -> dict:
    """what fraction of a matrix a target vocabulary can actually reach.

    the number that decides whether a symbol-matching encoder sees a
    transcriptome or a fragment of one. novae asserts outright at zero; between
    zero and adequate it degrades silently, which is the case worth reporting.
    """
    names = [str(n) for n in names]
    keys = [n.split(".")[0] if strip_version else n for n in names]
    vocab = set(map(str, vocabulary))
    hit = sum(k in vocab for k in keys)
    return {"n_genes": len(names), "n_matched": hit,
            "fraction": round(hit / len(names), 4) if names else 0.0,
            "n_unmatched": len(names) - hit}


def agreement(fingerprints: dict) -> dict:
    """do these sections share one reference? the precondition for stacking them.

    a cohort assembled from several studies usually does not, and a single
    cohort-level claim is then wrong for most of it. reported per distinct
    identifier space so the disagreement is legible rather than averaged away.
    """
    by_space: dict[str, list[str]] = {}
    for sid, f in fingerprints.items():
        by_space.setdefault(f["identifier_space"], []).append(sid)
    counts = {sid: f["n_genes"] for sid, f in fingerprints.items()}
    n = sorted(counts.values())
    return {
        "n_sections": len(fingerprints),
        "identifier_spaces": {k: len(v) for k, v in sorted(by_space.items())},
        "one_space": len(by_space) == 1,
        "sections_by_space": {k: sorted(v)[:8] for k, v in sorted(by_space.items())},
        "n_genes_range": [n[0], n[-1]] if n else None,
        # identical gene counts across sections imply one reference AND one
        # filter; a spread means at least one of the two differs, which is enough
        # to make an unrestricted stack unsafe.
        "one_gene_count": len(set(counts.values())) == 1,
        "stackable_without_a_declared_universe": len(by_space) == 1 and len(set(counts.values())) == 1,
    }
