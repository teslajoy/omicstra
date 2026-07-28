"""
niche_aucell.py - per-niche pathway activity via AUCell on Reactome.

pipeline:
  1. build niche count matrix: center + 6 neighbors = 7-spot sum per niche
     - filter MT-* / MTRNR* genes before aggregation (standard ST QC)
     - cache to disk so re-runs are fast
  2. normalize per niche: normalize_total(target_sum=1e4) -> log1p
  3. run AUCell via decoupler against two pathway universes:
     (a) 835 low-level Reactome pathways (H3 exploration matrix)
     (b) 5 proposal target parent pathways (built by BFS rolling up
         leaf-level gene sets to parent unions):
         TGF-beta Signaling      R-HSA-170834
         Immune System           R-HSA-168256
         ECM Organization        R-HSA-1474244
         Cell Cycle              R-HSA-1640170
         Programmed Cell Death   R-HSA-5357801
  4. per-subarray z-score the 5 target columns (same logic as novae
     per-subarray normalization). full 835 matrix left raw.

outputs:
  data/embeddings/biological_signals/niche_counts_cache.npz         (sparse + meta)
  data/embeddings/biological_signals/reactome_parent_gene_sets.json (5 target unions)
  data/embeddings/biological_signals/niche_aucell_low_level.parquet (n_niches x 835 + meta)
  data/embeddings/biological_signals/niche_aucell_5targets.parquet  (n_niches x 5 raw + zscored + meta)
"""

import io
import json
import re
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
BIO = ROOT / "data" / "embeddings" / "biological_signals"
NICHE_DIR = ROOT / "data" / "embeddings" / "virchow2_niche"
BYARRAY = ROOT / "data" / "inputs" / "byArray"
REACTOME = ROOT / "knowledge" / "reactome"

NICHE_CACHE = BIO / "niche_counts_cache.npz"
NICHE_META_CACHE = BIO / "niche_counts_cache_meta.parquet"
PARENT_SETS_OUT = BIO / "reactome_parent_gene_sets.json"
LOW_LEVEL_OUT = BIO / "niche_aucell_low_level.parquet"
TARGET_OUT = BIO / "niche_aucell_5targets.parquet"

# 5 target parent pathways
TARGET_PATHWAYS = {
    "TGF-beta_Signaling":     "R-HSA-170834",
    "Immune_System":          "R-HSA-168256",
    "ECM_Organization":       "R-HSA-1474244",
    "Cell_Cycle":             "R-HSA-1640170",
    "Programmed_Cell_Death":  "R-HSA-5357801",
}

K_NEIGHBORS = 6
MIN_PATHWAY_SIZE = 3
MAX_PATHWAY_SIZE = 500
MT_PREFIXES = ("MT-", "MTRNR")

# pilot filter: if set, only subarrays whose stem starts with one of these prefixes
# are processed. empty tuple = all 280 subarrays.
PILOT_PREFIXES = ()  # empty = all 280 subarrays  # ("TNBC1_", "TNBC3_", "TNBC83_")


# ========================================================================
# phase 1: build 5 parent gene sets via BFS on Reactome hierarchy
# ========================================================================

def build_parent_gene_sets():
    """BFS from each of 5 target stIds, union descendant gene sets."""
    # parent -> children map
    children_of = defaultdict(set)
    with open(REACTOME / "ReactomePathwaysRelation.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2 and "-HSA" in parts[0]:
                children_of[parts[0]].add(parts[1])

    # load gene sets from gmt (we already have this in low_level_pathways.json
    # but need FULL gmt including high-level entries)
    import zipfile
    gmt = {}
    with zipfile.ZipFile(REACTOME / "ReactomePathways.gmt.zip") as z:
        with z.open(z.namelist()[0]) as f:
            for line in f:
                parts = line.decode("utf8").strip().split("\t")
                if len(parts) >= 3 and parts[1].startswith("R-HSA-"):
                    gmt[parts[1]] = set(parts[2:])

    def descendants(root):
        visited = {root}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for child in children_of.get(node, []):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        return visited

    parent_sets = {}
    for name, stId in TARGET_PATHWAYS.items():
        desc = descendants(stId)
        genes = set()
        for d in desc:
            if d in gmt:
                genes.update(gmt[d])
        parent_sets[name] = {
            "stId": stId,
            "n_descendants": len(desc),
            "n_descendants_with_genes": sum(1 for d in desc if d in gmt),
            "n_genes": len(genes),
            "genes": sorted(genes),
        }
        print(f"  {name} ({stId}): {len(desc)} descendants, {len(genes)} unique genes")

    PARENT_SETS_OUT.write_text(json.dumps(parent_sets, indent=1))
    print(f"  saved -> {PARENT_SETS_OUT.relative_to(ROOT)}")
    return parent_sets


# ========================================================================
# phase 2: niche count matrix build + cache
# ========================================================================

def _build_subarray_chunk(
    mpath, e2s_map, symbol_to_global_col, n_global,
):
    """Build one csr_matrix chunk [n_center_niches x n_global] for one subarray.

    Returns (X_sub, metadata_rows) or (None, None) if subarray should be skipped.
    Memory-bounded per call: roughly (n_niches * ~2k nonzero * 12 bytes) ~ 10-50 MB.
    """
    stem = mpath.stem.replace("_meta", "")
    parts = stem.split("_", 1)
    if len(parts) != 2 or not parts[0].startswith("TNBC"):
        return None, None
    patient = int(parts[0].replace("TNBC", ""))
    slide, pos = parts[1].split("_")
    rdata = BYARRAY / slide / pos / "selection.RData"
    if not rdata.exists():
        return None, None

    r = subprocess.run(
        ["Rscript", "-e",
         f'load("{rdata}"); write.table(cnts, stdout(), sep="\\t", quote=FALSE, col.names=NA)'],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [err] {stem}: Rscript failed")
        return None, None
    cnts = pd.read_csv(io.StringIO(r.stdout), sep="\t", index_col=0)

    # Ensembl.version -> HGNC -> drop unmapped, dupes, MT. Local columns mapped to global.
    stripped = [re.sub(r"\.\d+$", "", g) for g in cnts.columns]
    keep_local_idx = []
    keep_global_col = []
    seen = set()
    for i, eid in enumerate(stripped):
        sym = e2s_map.get(eid)
        if sym is None or sym in seen:
            continue
        if sym.startswith(MT_PREFIXES):
            continue
        gcol = symbol_to_global_col.get(sym)
        if gcol is None:
            continue  # symbol not in our global universe
        seen.add(sym)
        keep_local_idx.append(i)
        keep_global_col.append(gcol)
    if not keep_local_idx:
        return None, None

    keep_local_idx = np.asarray(keep_local_idx, dtype=np.int64)
    keep_global_col = np.asarray(keep_global_col, dtype=np.int64)

    counts_local = cnts.values[:, keep_local_idx].astype(np.float32)  # [spots x n_kept_local]

    # align with virchow2 niche meta
    meta = pd.read_csv(mpath, sep="\t", index_col=0)
    shared = meta.index.intersection(cnts.index)
    if len(shared) == 0:
        return None, None
    meta = meta.loc[shared]
    pos_in_cnts = {sid: i for i, sid in enumerate(cnts.index)}
    row_order = np.array([pos_in_cnts[sid] for sid in meta.index])
    counts_local = counts_local[row_order]
    coords = meta[["pixel_x", "pixel_y"]].values
    is_center = meta["neighbor_count"].values >= 7

    # k=6 nearest neighbors per spot, then center + 6 neighbors = 7-spot niche
    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1).fit(coords)
    _, idx = nn.kneighbors(coords)  # [n_spots x 7]
    center_rows = np.where(is_center)[0]
    if len(center_rows) == 0:
        return None, None

    # vectorized 7-spot sum per niche: [n_centers x n_kept_local]
    niche_counts = counts_local[idx[center_rows]].sum(axis=1)

    # build COO directly in GLOBAL column space (bounded per subarray)
    nz_row, nz_col_local = np.nonzero(niche_counts)
    if len(nz_row) == 0:
        return None, None
    rows = nz_row.astype(np.int32)
    cols = keep_global_col[nz_col_local].astype(np.int32)
    vals = niche_counts[nz_row, nz_col_local]

    X_sub = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(len(center_rows), n_global),
        dtype=np.float32,
    ).tocsr()

    metadata_rows = [
        {
            "niche_id": f"{stem}::{meta.index[ci]}",
            "subarray": stem,
            "patient": patient,
            "spot_id": str(meta.index[ci]),
            "pixel_x": float(coords[ci, 0]),
            "pixel_y": float(coords[ci, 1]),
        }
        for ci in center_rows
    ]
    return X_sub, metadata_rows


def build_niche_count_cache():
    """
    Per-subarray sparse chunk + sp.vstack pattern. Constant memory per chunk.
    Global gene universe is fixed upfront from ensembl_to_symbol.tsv (minus MT/MTRNR).
    """
    e2s = pd.read_csv(ROOT / "data" / "inputs" / "ensembl_to_symbol.tsv", sep="\t")
    e2s_map = dict(zip(e2s["ensembl_id"], e2s["gene_symbol"]))

    # build stable global HGNC universe = all unique symbols minus MT-prefixed
    all_symbols = sorted(set(s for s in e2s_map.values() if s and not s.startswith(MT_PREFIXES)))
    genes = np.array(all_symbols)
    symbol_to_global_col = {s: i for i, s in enumerate(all_symbols)}
    n_global = len(genes)
    n_mt_dropped = sum(1 for s in set(e2s_map.values()) if s and s.startswith(MT_PREFIXES))
    print(f"  global gene universe: {n_global} HGNC symbols (dropped {n_mt_dropped} MT/MTRNR genes)")

    meta_files = sorted(NICHE_DIR.glob("*_meta.tsv"))
    if PILOT_PREFIXES:
        meta_files = [m for m in meta_files if any(m.stem.startswith(p) for p in PILOT_PREFIXES)]
        print(f"  PILOT mode: {len(meta_files)} subarrays matching {PILOT_PREFIXES}")
    else:
        print(f"  found {len(meta_files)} subarray meta files (full run)")

    chunks = []
    metadata = []
    total_niches = 0
    n_ok = 0

    for fi, mpath in enumerate(meta_files):
        X_sub, meta_rows = _build_subarray_chunk(mpath, e2s_map, symbol_to_global_col, n_global)
        if X_sub is None:
            continue
        chunks.append(X_sub)
        metadata.extend(meta_rows)
        total_niches += X_sub.shape[0]
        n_ok += 1

        if (fi + 1) % 20 == 0:
            mb = sum(c.data.nbytes + c.indices.nbytes + c.indptr.nbytes for c in chunks) / 1e6
            print(f"  {fi + 1}/{len(meta_files)}  chunks={len(chunks)}  niches={total_niches:,}  chunk_mem={mb:.0f} MB")

    print(f"  stacking {len(chunks)} chunks into one csr matrix...")
    X = sp.vstack(chunks, format="csr")
    del chunks  # release per-chunk memory

    meta_df = pd.DataFrame(metadata)
    print(f"  final: {X.shape}  nnz={X.nnz:,}  density={X.nnz/(X.shape[0]*X.shape[1]):.4f}  "
          f"subarrays_used={n_ok}")

    # save csr components separately to avoid scipy save_npz zip64 corruption on >4GB matrices
    cache_dir = BIO / "niche_counts_cache"
    cache_dir.mkdir(exist_ok=True)
    np.save(cache_dir / "data.npy", X.data)
    np.save(cache_dir / "indices.npy", X.indices)
    np.save(cache_dir / "indptr.npy", X.indptr)
    np.save(cache_dir / "shape.npy", np.array(X.shape))
    meta_df.to_parquet(NICHE_META_CACHE)
    np.save(BIO / "niche_counts_cache_genes.npy", genes)
    print(f"  saved -> niche_counts_cache/{{data,indices,indptr,shape}}.npy + _meta.parquet + _genes.npy")
    return X, meta_df, genes


def load_niche_count_cache():
    cache_dir = BIO / "niche_counts_cache"
    if cache_dir.exists():
        data = np.load(cache_dir / "data.npy")
        indices = np.load(cache_dir / "indices.npy")
        indptr = np.load(cache_dir / "indptr.npy")
        shape = tuple(np.load(cache_dir / "shape.npy"))
        X = sp.csr_matrix((data, indices, indptr), shape=shape)
    else:
        # fallback to legacy npz (small matrices only)
        X = sp.load_npz(BIO / "niche_counts_cache_X.npz")
    meta_df = pd.read_parquet(NICHE_META_CACHE)
    genes = np.load(BIO / "niche_counts_cache_genes.npy", allow_pickle=True)
    return X, meta_df, genes


# ========================================================================
# phase 3: AUCell scoring
# ========================================================================

def run_aucell_pipeline(X, meta_df, genes, parent_sets):
    import anndata as ad
    import decoupler as dc

    print(f"  building AnnData: {X.shape}")
    adata = ad.AnnData(X=X.tocsr(),
                       obs=meta_df.reset_index(drop=True),
                       var=pd.DataFrame(index=genes))
    adata.obs_names_make_unique()

    # normalize: lib size to 1e4, then log1p
    print("  normalizing: target_sum=1e4 -> log1p")
    lib = np.asarray(adata.X.sum(axis=1)).ravel()
    scaling = 1e4 / np.maximum(lib, 1)
    # in place per-row scaling of sparse matrix
    adata.X = adata.X.multiply(scaling[:, None]).tocsr()
    adata.X.data = np.log1p(adata.X.data)

    # --- low-level Reactome net ---
    lp = json.loads((REACTOME / "low_level_pathways.json").read_text())["pathways"]
    low_rows = []
    for stId, p in lp.items():
        sz = len(p["genes"])
        if MIN_PATHWAY_SIZE <= sz <= MAX_PATHWAY_SIZE:
            name = p["name"]
            for g in p["genes"]:
                low_rows.append({"source": f"{stId}|{name[:50]}", "target": g})
    low_net = pd.DataFrame(low_rows)
    print(f"  low-level net: {low_net['source'].nunique()} pathways, {len(low_net)} edges")

    # --- target parent net ---
    target_rows = []
    for name, info in parent_sets.items():
        for g in info["genes"]:
            target_rows.append({"source": name, "target": g})
    target_net = pd.DataFrame(target_rows)
    print(f"  target net: {target_net['source'].nunique()} parents, {len(target_net)} edges")

    print("  running AUCell on low-level pathways (slow)...")
    dc.mt.aucell(adata, net=low_net, tmin=MIN_PATHWAY_SIZE, verbose=True)
    low_scores = adata.obsm["score_aucell"].copy()
    print(f"    low-level scores: {low_scores.shape}")

    print("  running AUCell on 5 target parent pathways...")
    dc.mt.aucell(adata, net=target_net, tmin=MIN_PATHWAY_SIZE, verbose=True)
    target_scores = adata.obsm["score_aucell"].copy()
    print(f"    target scores: {target_scores.shape}")

    return low_scores, target_scores


# ========================================================================
# main
# ========================================================================

print("=" * 60)
print("phase 1: build 5 parent gene sets via BFS")
print("=" * 60)
parent_sets = build_parent_gene_sets()

print("\n" + "=" * 60)
print("phase 2: niche count matrix (build or load cache)")
print("=" * 60)
cache_dir = BIO / "niche_counts_cache"
cache_x = BIO / "niche_counts_cache_X.npz"
if cache_dir.exists() or cache_x.exists():
    print("  loading cached niche count matrix")
    X, meta_df, genes = load_niche_count_cache()
    print(f"  X: {X.shape}  niches: {len(meta_df)}  genes: {len(genes)}")
else:
    print("  building niche count matrix from selection.RData (slow, ~10 min)")
    X, meta_df, genes = build_niche_count_cache()

print("\n" + "=" * 60)
print("phase 3: AUCell scoring")
print("=" * 60)
low_scores, target_scores = run_aucell_pipeline(X, meta_df, genes, parent_sets)

print("\n" + "=" * 60)
print("phase 4: per-subarray z-score on 5 target columns")
print("=" * 60)
target_scores.index = meta_df["niche_id"].values
target_scores.index.name = "niche_id"
# attach subarray
target_df = target_scores.copy()
target_df["subarray"] = meta_df["subarray"].values
target_df["patient"] = meta_df["patient"].values
target_df["spot_id"] = meta_df["spot_id"].values
target_df["pixel_x"] = meta_df["pixel_x"].values
target_df["pixel_y"] = meta_df["pixel_y"].values

z_cols = {}
for col in TARGET_PATHWAYS:
    if col not in target_scores.columns:
        print(f"  warning: {col} not in AUCell output columns: {target_scores.columns.tolist()}")
        continue
    z_col = f"{col}_z"
    # z-score within each subarray
    target_df[z_col] = (
        target_df.groupby("subarray")[col]
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    )
    z_cols[col] = z_col

target_df.to_parquet(TARGET_OUT)
print(f"  saved -> {TARGET_OUT.relative_to(ROOT)}")

low_out = low_scores.copy()
low_out["niche_id"] = meta_df["niche_id"].values
low_out["subarray"] = meta_df["subarray"].values
low_out["patient"] = meta_df["patient"].values
low_out["spot_id"] = meta_df["spot_id"].values
low_out.to_parquet(LOW_LEVEL_OUT)
print(f"  saved -> {LOW_LEVEL_OUT.relative_to(ROOT)}")

print("\n" + "=" * 60)
print("summary")
print("=" * 60)
print(f"total niches: {len(meta_df)}")
print(f"subarrays: {meta_df['subarray'].nunique()}")
print(f"patients: {meta_df['patient'].nunique()}")
print(f"low-level AUCell matrix: {low_scores.shape}")
print(f"target AUCell matrix:    {target_scores.shape}")

print("\nper-target global statistics (AUCell raw):")
for col in TARGET_PATHWAYS:
    if col in target_scores.columns:
        v = target_scores[col].values
        print(f"  {col:<25}  mean={v.mean():.4f}  std={v.std():.4f}  min={v.min():.4f}  max={v.max():.4f}")