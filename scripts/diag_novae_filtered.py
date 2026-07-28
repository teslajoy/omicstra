"""
diag_novae_filtered.py - test if Novae fails on Wang et al. due to graph quality.

hypothesis A: platform mismatch (100um spots too coarse for Novae)
hypothesis B: graph corruption from background/artifact spots in neighborhoods

test on CN1_C1 (TNBC1):
  1. unfiltered: existing novae_all/TNBC1_CN1_C1.npy embeddings
  2. filtered: re-run Novae on graph excluding background-dominated spots

compare silhouette vs MC labels for both.
if filtered improves meaningfully -> Novae is salvageable
if filtered stays bad -> platform mismatch confirmed
"""

import warnings
warnings.filterwarnings('ignore')

import subprocess
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parent.parent
SUBARRAY = 'CN1_C1'
SHORT_SUBARRAY = SUBARRAY  # CN1_C1
FULL_SUBARRAY = 'TNBC1_CN1_C1'

print(f'=== Novae filtered graph test on {FULL_SUBARRAY} ===\n')

# load existing unfiltered Novae embeddings
unfiltered_path = ROOT / 'data' / 'embeddings' / 'novae_all' / f'{FULL_SUBARRAY}.npy'
unfiltered = np.load(unfiltered_path)
print(f'unfiltered novae shape: {unfiltered.shape}')

# load spot order from selection.RData
rdata_path = ROOT / 'data' / 'inputs' / 'byArray' / 'CN1' / 'C1' / 'selection.RData'
r = subprocess.run(['Rscript', '-e', f'load("{rdata_path}"); cat(rownames(cnts), sep="\\n")'],
                   capture_output=True, text=True)
spot_order = [s for s in r.stdout.strip().split('\n') if s]
print(f'selection.RData spots: {len(spot_order)}')

# load mc_labels
mc_all = pd.read_csv(ROOT / 'data' / 'embeddings' / 'biological_signals' / 'mc_labels.tsv', sep='\t')
mc_sub = mc_all[mc_all['subarray'] == SHORT_SUBARRAY].set_index('spot_id')
print(f'mc_labels for {SHORT_SUBARRAY}: {len(mc_sub)} spots')

# load morphology to get filter mask (background fraction)
morph_all = pd.read_csv(ROOT / 'data' / 'embeddings' / 'biological_signals' / 'morphology_labels.tsv', sep='\t')
morph_sub = morph_all[morph_all['subarray_id'] == FULL_SUBARRAY].set_index('spot_id')
print(f'morphology for {FULL_SUBARRAY}: {len(morph_sub)} spots (background-pre-filtered)')

# the morphology table already excluded background-dominated spots
# spots in morph_sub are KEEP, spots not in morph_sub are background
clean_spot_set = set(morph_sub.index)
print(f'spots passing morphology QC: {len(clean_spot_set)} / {len(spot_order)}')

# build filtered Novae input: load counts and filter spots
print('\nloading counts + filtering...')
r = subprocess.run(['Rscript', '-e', f'''
    load("{rdata_path}")
    write.csv(cnts, stdout())
'''], capture_output=True, text=True)
counts_df = pd.read_csv(io.StringIO(r.stdout), index_col=0)
print(f'counts shape: {counts_df.shape}')

# spot coords
r = subprocess.run(['Rscript', '-e', f'''
    load("{rdata_path}")
    write.csv(spots, stdout())
'''], capture_output=True, text=True)
spots_df = pd.read_csv(io.StringIO(r.stdout), index_col=0)

# gene mapping
gene_map = pd.read_csv(ROOT / 'data' / 'inputs' / 'ensembl_to_symbol.tsv', sep='\t')
ensembl_to_symbol = dict(zip(gene_map.iloc[:, 0], gene_map.iloc[:, 1]))

# map ensembl -> symbol
symbols = []
keep_genes = []
for eid in counts_df.columns:
    base = eid.split('.')[0]
    if base in ensembl_to_symbol:
        symbols.append(ensembl_to_symbol[base])
        keep_genes.append(True)
    else:
        keep_genes.append(False)
counts_mapped = counts_df.loc[:, keep_genes].copy()
counts_mapped.columns = symbols
counts_mapped = counts_mapped.T.groupby(level=0).sum().T
print(f'gene-mapped counts: {counts_mapped.shape}')

# filter spots
spot_mask = counts_mapped.index.isin(clean_spot_set)
counts_clean = counts_mapped[spot_mask]
spots_clean = spots_df[spots_df.index.isin(clean_spot_set)]
# align order
common = list(set(counts_clean.index) & set(spots_clean.index))
counts_clean = counts_clean.loc[common]
spots_clean = spots_clean.loc[common]
print(f'after filtering: {counts_clean.shape[0]} spots')

# build AnnData and run Novae
print('\nrunning Novae on filtered graph...')
import anndata as ad
import novae

adata = ad.AnnData(
    X=csr_matrix(counts_clean.values),
    obs=pd.DataFrame(index=counts_clean.index),
    var=pd.DataFrame(index=counts_clean.columns),
)
adata.obsm['spatial'] = spots_clean[['pixel_x', 'pixel_y']].values
adata.obs['subarray'] = SHORT_SUBARRAY

novae.settings.scale_to_microns = 200.0 / 164.0
novae.spatial_neighbors(adata)

model = novae.Novae.from_pretrained('MICS-Lab/novae-human-0')
model.compute_representations(adata, zero_shot=True)

filtered_emb = adata.obsm['novae_latent']
if hasattr(filtered_emb, 'values'):
    filtered_emb = filtered_emb.values
filtered_emb = np.asarray(filtered_emb, dtype=np.float32)
print(f'filtered novae shape: {filtered_emb.shape}')

# --- comparison ---
# unfiltered: take only the spots that are also in filtered set
spot_to_idx_unfiltered = {sid: i for i, sid in enumerate(spot_order)}
unfiltered_idx = [spot_to_idx_unfiltered[s] for s in counts_clean.index if s in spot_to_idx_unfiltered]
unfiltered_aligned = unfiltered[unfiltered_idx]

# get MC labels for these spots
mc_clean = mc_sub.loc[mc_sub.index.intersection(counts_clean.index), 'megacluster']
print(f'\ncomparison spots: {len(mc_clean)} (have unfiltered + filtered + MC label)')

# align embeddings to MC label order
common_with_mc = list(set(counts_clean.index) & set(mc_clean.index))
common_with_mc.sort()  # deterministic order
print(f'common spots with all data: {len(common_with_mc)}')

# index lookups
idx_map_filtered = {s: i for i, s in enumerate(counts_clean.index)}
idx_map_unfiltered_aligned = {s: i for i, s in enumerate(counts_clean.index)}  # same order

filtered_for_eval = filtered_emb[[idx_map_filtered[s] for s in common_with_mc]]
unfiltered_for_eval = unfiltered_aligned[[idx_map_unfiltered_aligned[s] for s in common_with_mc]]
mc_for_eval = mc_clean.loc[common_with_mc].values

n_classes = len(set(mc_for_eval))
print(f'unique MCs in CN1_C1: {sorted(set(mc_for_eval))}')

# silhouette + ARI for both
print('\n=== silhouette score (cosine, vs MC labels) ===')
sil_unf = silhouette_score(unfiltered_for_eval, mc_for_eval, metric='cosine')
sil_fil = silhouette_score(filtered_for_eval, mc_for_eval, metric='cosine')
print(f'  unfiltered: {sil_unf:.4f}')
print(f'  filtered:   {sil_fil:.4f}')
print(f'  delta:      {sil_fil - sil_unf:+.4f}')

# ARI vs k-means
print('\n=== ARI (k-means K vs MC labels) ===')
km_unf = MiniBatchKMeans(n_clusters=n_classes, random_state=42, n_init=3, batch_size=512)
pred_unf = km_unf.fit_predict(unfiltered_for_eval)
ari_unf = adjusted_rand_score(mc_for_eval, pred_unf)

km_fil = MiniBatchKMeans(n_clusters=n_classes, random_state=42, n_init=3, batch_size=512)
pred_fil = km_fil.fit_predict(filtered_for_eval)
ari_fil = adjusted_rand_score(mc_for_eval, pred_fil)

print(f'  unfiltered: {ari_unf:.4f}')
print(f'  filtered:   {ari_fil:.4f}')
print(f'  delta:      {ari_fil - ari_unf:+.4f}')

# verdict
print('\n=== VERDICT ===')
if sil_fil > 0 and sil_fil - sil_unf > 0.05:
    print('FILTERING HELPS - hypothesis B confirmed (graph quality)')
    print('  Novae is salvageable with QC pre-filtering')
elif sil_fil > sil_unf + 0.02:
    print('FILTERING HELPS MARGINALLY - graph quality matters but not the only issue')
else:
    print('FILTERING DOES NOT HELP - hypothesis A confirmed (platform mismatch)')
    print('  Novae is NOT salvageable on this 100um spot platform')

# save filtered embeddings for inspection
np.save(ROOT / 'data' / 'embeddings' / 'novae_filtered_test_CN1_C1.npy', filtered_emb)
print(f'\nfiltered embeddings saved to data/embeddings/novae_filtered_test_CN1_C1.npy')
