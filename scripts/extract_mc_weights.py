"""
extract_mc_weights.py - per-spot soft megacluster weights from Wang et al. NMF.

source: data/inputs/clustering/MC deconv/TNBC{N}.RData
each patient file has:
  m: matrix [n_spots x 14] - soft NMF weights for the 14 megaclusters
  idSpot: data.frame [n_spots x 2] - slide, spot columns

output: per-spot 14-d weight vector linking H&E patches to cell program mixtures.

usage:
    python scripts/extract_mc_weights.py

output:
    data/embeddings/biological_signals/mc_weights.tsv
        columns: subarray, spot_id, mc1..mc14 (14 columns of soft weights)
"""

import warnings
warnings.filterwarnings('ignore')

import subprocess
import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'inputs'
MC_DECONV_DIR = DATA / 'clustering' / 'MC deconv'
IDS_PATH = DATA / 'Clinical' / 'ids.RDS'
OUT_DIR = ROOT / 'data' / 'embeddings' / 'biological_signals'


def load_ids_mapping():
    """build slide_idx (1..N) -> subarray name (CN1_C1) from ids.RDS.
    idSpot$slide is an integer index per patient, need to map back."""
    r = subprocess.run(['Rscript', '-e', f'''
        ids <- readRDS("{IDS_PATH}")
        # need patient_id -> ordered list of subarrays
        out <- data.frame(
            subarray = rownames(ids),
            patient_id = ids$id,
            slide = ids$slide,
            subslide = ids$subslide,
            hasAnnot = ids$hasAnnot
        )
        write.csv(out, stdout(), row.names=FALSE)
    '''], capture_output=True, text=True)
    if r.returncode != 0:
        print(f'ids load failed: {r.stderr[:200]}')
        return None
    return pd.read_csv(io.StringIO(r.stdout))


def load_per_patient_weights(patient_id):
    """load m matrix and idSpot for one patient."""
    path = MC_DECONV_DIR / f'TNBC{patient_id}.RData'
    if not path.exists():
        return None, None

    r = subprocess.run(['Rscript', '-e', f'''
        load("{path}")
        cat("===DIM===\n", paste(dim(m), collapse=","), "\n", sep="")
        cat("===IDSPOT===\n")
        write.csv(idSpot, stdout())
        cat("===M===\n")
        write.csv(m, stdout())
    '''], capture_output=True, text=True)
    if r.returncode != 0:
        return None, None

    parts = r.stdout.split('===')
    sections = {}
    for i in range(1, len(parts) - 1, 2):
        if i + 1 < len(parts):
            sections[parts[i]] = parts[i + 1]

    if 'IDSPOT' not in sections or 'M' not in sections:
        return None, None

    idspot_df = pd.read_csv(io.StringIO(sections['IDSPOT'].strip()), index_col=0)
    m_df = pd.read_csv(io.StringIO(sections['M'].strip()), index_col=0)

    return m_df, idspot_df


def extract_all():
    print('=== MC weight extraction ===')
    print('loading ids mapping...')
    ids = load_ids_mapping()
    if ids is None:
        return
    print(f'  loaded {len(ids)} subarrays from ids.RDS')

    # build patient_id -> ordered list of (slide_idx, subarray_name)
    # slide column in idSpot is an integer per patient sequentially numbering subarrays
    # we need to know the order
    patient_subarrays = {}
    for pid in ids['patient_id'].unique():
        sub_rows = ids[ids['patient_id'] == pid].sort_values('subarray')
        patient_subarrays[pid] = sub_rows['subarray'].tolist()

    print(f'  {len(patient_subarrays)} patients')

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    skipped = []
    n_total = 0
    patients = sorted(patient_subarrays.keys())

    for i, pid in enumerate(patients):
        m_df, idspot_df = load_per_patient_weights(pid)
        if m_df is None:
            skipped.append(pid)
            continue

        n_spots = len(m_df)
        if len(idspot_df) != n_spots:
            print(f'  TNBC{pid}: WARNING idspot {len(idspot_df)} vs m {n_spots}')
            continue

        # idSpot columns: slide (integer index), spot (e.g., "2x12")
        slide_indices = idspot_df['slide'].values
        spot_ids = idspot_df['spot'].values

        # map slide index -> actual subarray name using patient_subarrays order
        sub_list = patient_subarrays[pid]
        n_unique_slides = int(np.max(slide_indices))
        if n_unique_slides > len(sub_list):
            print(f'  TNBC{pid}: slide indices go to {n_unique_slides} but only '
                  f'{len(sub_list)} subarrays in ids')

        weights = m_df.values  # (n_spots, 14)
        # normalize each row to sum to 1 (so weights are mixture proportions)
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        weights_normed = weights / row_sums

        for j in range(n_spots):
            slide_idx = int(slide_indices[j])
            if 1 <= slide_idx <= len(sub_list):
                subarray = sub_list[slide_idx - 1]
            else:
                subarray = f'TNBC{pid}_slide{slide_idx}'
            row = {'subarray': subarray, 'spot_id': spot_ids[j], 'patient_id': pid}
            for k in range(14):
                row[f'mc{k+1}'] = float(weights_normed[j, k])
            all_rows.append(row)

        n_total += n_spots
        if (i + 1) % 10 == 0 or i == 0:
            print(f'  [{i+1}/{len(patients)}] TNBC{pid}: {n_spots} spots')

    out_df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / 'mc_weights.tsv'
    out_df.to_csv(out_path, sep='\t', index=False)

    print(f'\nsaved {len(out_df)} spots from {out_df["patient_id"].nunique()} patients '
          f'-> {out_path.relative_to(ROOT)}')
    if skipped:
        print(f'skipped: {skipped}')

    # quick stats
    mc_cols = [f'mc{k}' for k in range(1, 15)]
    mc_means = out_df[mc_cols].mean()
    print('\nmean weight per MC across all spots:')
    for col in mc_cols:
        print(f'  {col}: {mc_means[col]:.4f}')

    # how many spots have a dominant MC (max weight > 0.5)?
    max_w = out_df[mc_cols].max(axis=1)
    n_dominant = (max_w > 0.5).sum()
    print(f'\nspots with max MC weight > 0.5: {n_dominant}/{len(out_df)} '
          f'({n_dominant/len(out_df)*100:.1f}%) - "pure" spots')
    print(f'spots with max MC weight > 0.3: '
          f'{(max_w > 0.3).sum()/len(out_df)*100:.1f}%')


if __name__ == '__main__':
    extract_all()