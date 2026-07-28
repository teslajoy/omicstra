"""
extract_mc_labels.py - per-spot megacluster (MC) labels from Wang et al. data.

uses three Zenodo files to map every spot to one of 14 megaclusters:
  1. clustering/intraPatientClust/TNBC{N}.RData  - per-spot intra-patient cluster (km)
  2. classification/projectedSamples/TNBC{N}.RData - subarray.spot_id rownames
  3. clustering/Kmeans MC/km14.RDS - intra-cluster -> megacluster mapping

Wang et al. methodology:
  - cluster spots within each patient (k=2..10)
  - select k per patient based on stability
  - cluster the patient-level cluster prototypes globally into 14 megaclusters
  - megaclusters are cross-patient stable by construction

usage:
    python scripts/extract_mc_labels.py

outputs:
    data/embeddings/biological_signals/mc_labels.tsv
        columns: subarray_id, spot_id, patient_id, intra_cluster, megacluster
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
INTRA_DIR = DATA / 'clustering' / 'intraPatientClust'
PROJ_DIR = DATA / 'classification' / 'projectedSamples'
KM14_PATH = DATA / 'clustering' / 'Kmeans MC' / 'km14.RDS'
OUT_DIR = ROOT / 'data' / 'embeddings' / 'biological_signals'


def load_km14_mapping():
    """parse km14.RDS into a {(patient_id, intra_cluster): mc} dict."""
    r = subprocess.run(['Rscript', '-e', f'''
        x <- readRDS("{KM14_PATH}")
        # cluster vector with names like "1 1", "1 2", "10 3"
        out <- data.frame(name=names(x$cluster), mc=x$cluster)
        write.csv(out, stdout(), row.names=FALSE)
    '''], capture_output=True, text=True)
    assert r.returncode == 0, f'km14 load failed: {r.stderr}'

    df = pd.read_csv(io.StringIO(r.stdout))
    mapping = {}
    for _, row in df.iterrows():
        parts = row['name'].split()
        if len(parts) != 2:
            continue
        patient_id = int(parts[0])
        intra_cluster = int(parts[1])
        mapping[(patient_id, intra_cluster)] = int(row['mc'])

    return mapping


def load_per_patient(patient_id):
    """load km matrix and spot index for one patient.
    returns:
        km: (n_spots, 9) intra-patient cluster assignments for k=2..10
        spot_index: dataframe with rowname = subarray.spot_id
    """
    intra_path = INTRA_DIR / f'TNBC{patient_id}.RData'
    proj_path = PROJ_DIR / f'TNBC{patient_id}.RData'

    if not intra_path.exists() or not proj_path.exists():
        return None, None

    r = subprocess.run(['Rscript', '-e', f'''
        load("{intra_path}")
        load("{proj_path}")
        # km has k=2..10 in columns 1..9
        # spot has rownames like "CN1_C1.2x12"
        cat("===KM_DIM===\n", paste(dim(km), collapse=","), "\n", sep="")
        cat("===SPOT_NAMES===\n")
        cat(rownames(spot), sep="\n")
        cat("===KM_VALUES===\n")
        write.csv(km, stdout())
    '''], capture_output=True, text=True)
    if r.returncode != 0:
        return None, None

    output = r.stdout

    # parse the output
    parts = output.split('===')
    sections = {}
    for i in range(1, len(parts) - 1, 2):
        if i + 1 < len(parts):
            sections[parts[i]] = parts[i + 1]

    if 'SPOT_NAMES' not in sections or 'KM_VALUES' not in sections:
        return None, None

    spot_names = [s.strip() for s in sections['SPOT_NAMES'].strip().split('\n') if s.strip()]
    km_csv = sections['KM_VALUES'].strip()
    km_df = pd.read_csv(io.StringIO(km_csv), index_col=0)

    return km_df, spot_names


def extract_all_mc_labels():
    """build the per-spot MC label table for all patients."""
    print('=== MC label extraction ===')

    print('loading km14 mapping...')
    mapping = load_km14_mapping()
    print(f'  loaded {len(mapping)} (patient, intra_cluster) -> MC mappings')

    # find unique patients in mapping
    patients = sorted(set(p for p, _ in mapping.keys()))
    print(f'  {len(patients)} patients in MC scheme')

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    n_total = 0
    n_unmapped = 0
    skipped = []

    for i, pid in enumerate(patients):
        km_df, spot_names = load_per_patient(pid)
        if km_df is None:
            skipped.append(pid)
            continue

        n_spots = len(km_df)
        if len(spot_names) != n_spots:
            print(f'  TNBC{pid}: WARNING spot count mismatch '
                  f'{len(spot_names)} names vs {n_spots} km rows')
            continue

        # find which k value was used for this patient (from km14 mapping)
        intra_clusters_for_pid = sorted(set(c for p, c in mapping.keys() if p == pid))
        k_used = max(intra_clusters_for_pid)
        col_idx = k_used - 2  # k=2 is column 0 (1-indexed col 1)

        if col_idx < 0 or col_idx >= km_df.shape[1]:
            print(f'  TNBC{pid}: invalid k={k_used} for km shape {km_df.shape}')
            continue

        cluster_col = km_df.iloc[:, col_idx].values

        # assign MC per spot
        for j, spot_full_id in enumerate(spot_names):
            intra = int(cluster_col[j])
            mc = mapping.get((pid, intra))
            if mc is None:
                n_unmapped += 1
                continue
            # split "CN1_C1.2x12" -> subarray, spot_id
            if '.' in spot_full_id:
                subarray, spot_id = spot_full_id.split('.', 1)
            else:
                subarray, spot_id = '', spot_full_id
            all_rows.append({
                'subarray': subarray,
                'spot_id': spot_id,
                'patient_id': pid,
                'intra_cluster': intra,
                'megacluster': mc,
            })
        n_total += n_spots

        if (i + 1) % 10 == 0 or i == 0:
            print(f'  [{i+1}/{len(patients)}] TNBC{pid}: '
                  f'{n_spots} spots, k={k_used}, '
                  f'MCs: {sorted(set(mapping[(pid, c)] for c in intra_clusters_for_pid))}')

    out_df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / 'mc_labels.tsv'
    out_df.to_csv(out_path, sep='\t', index=False)

    print(f'\nsaved {len(out_df)} spots from {out_df["patient_id"].nunique()} patients '
          f'-> {out_path.relative_to(ROOT)}')
    print(f'unmapped spots: {n_unmapped}')
    if skipped:
        print(f'skipped (no data): {skipped}')

    print('\nMC distribution:')
    mc_counts = out_df['megacluster'].value_counts().sort_index()
    for mc, n in mc_counts.items():
        print(f'  MC{mc:2d}: {n:6d} ({n/len(out_df)*100:.1f}%)')

    return out_df


if __name__ == '__main__':
    extract_all_mc_labels()