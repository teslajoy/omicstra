"""
extract_biological_signals.py - extract per-spot biological labels from Wang et al. data.

step 5: inventory RDS files
step 6: morphological labels per spot (5-class collapse from annotBySpot.RDS)
step 7: patient-level biological labels from Clinical.RDS
step 8: TLS signature score per spot

usage:
    python scripts/extract_biological_signals.py
    python scripts/extract_biological_signals.py --step morphology
    python scripts/extract_biological_signals.py --step clinical
    python scripts/extract_biological_signals.py --step tls

outputs:
    data/embeddings/biological_signals/
        morphology_labels.tsv  - per-spot 5-class morphology
        clinical_labels.tsv    - per-patient labels (SA, subtype, TIME)
        tls_scores.tsv         - per-spot TLS signature score
        inventory.json         - what's available
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import subprocess
import io
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'inputs'
BY_ARRAY = DATA / 'byArray'
HD_DIR = DATA / 'Images' / 'imagesHD'
OUT_DIR = ROOT / 'data' / 'embeddings' / 'biological_signals'

# 18-category -> 5-class collapse mapping
TUMOR_COLS = ['Tumor', 'Tumor region', 'in situ']
STROMA_COLS = ['Stroma cell', 'Acellular stroma', 'Low TIL stroma']
IMMUNE_COLS = ['Lymphocyte', 'High TIL stroma', 'Lymphoid nodule']
OTHER_COLS = ['Fat tissue', 'Vessels', 'Necrosis', 'Lactiferous duct',
              'Nerve', 'Heterologous elements']
BACKGROUND_COLS = ['Nothing', 'Artefacts', 'Hole (whitespace)']

CLASS_MAP = {
    'Tumor': 'tumor', 'Tumor region': 'tumor', 'in situ': 'tumor',
    'Stroma cell': 'stroma', 'Acellular stroma': 'stroma', 'Low TIL stroma': 'stroma',
    'Lymphocyte': 'immune', 'High TIL stroma': 'immune', 'Lymphoid nodule': 'immune',
    'Fat tissue': 'other', 'Vessels': 'other', 'Necrosis': 'other',
    'Lactiferous duct': 'other', 'Nerve': 'other', 'Heterologous elements': 'other',
}

BACKGROUND_THRESHOLD = 0.5  # drop if background fraction > 0.5
MIXED_THRESHOLD = 0.5       # if dominant tissue fraction < 0.5, label = mixed


def discover_subarrays_with_annot():
    """find all subarrays that have annotBySpot.RDS."""
    hd_lookup = {}
    for p in HD_DIR.glob('*.jpg'):
        parts = p.stem.split('_')
        tnbc_id, slide, pos = parts[0], parts[1], parts[2]
        hd_lookup[(slide, pos)] = tnbc_id

    subarrays = []
    for slide_dir in sorted(BY_ARRAY.iterdir()):
        if not slide_dir.is_dir() or not slide_dir.name.startswith('CN'):
            continue
        for pos_dir in sorted(slide_dir.iterdir()):
            if not pos_dir.is_dir():
                continue
            annot_path = pos_dir / 'annotBySpot.RDS'
            if not annot_path.exists():
                continue
            slide = slide_dir.name
            pos = pos_dir.name
            tnbc_id = hd_lookup.get((slide, pos))
            key = f'{tnbc_id}_{slide}_{pos}' if tnbc_id else f'UNKNOWN_{slide}_{pos}'
            subarrays.append({
                'key': key,
                'tnbc_id': tnbc_id,
                'slide': slide,
                'pos': pos,
                'annot_path': str(annot_path),
                'rdata_path': str(pos_dir / 'selection.RData'),
            })
    return subarrays


def inventory():
    """quick inventory of biological signal files on disk."""
    print('=== inventory ===')
    info = {}

    # annotBySpot
    annot_subarrays = discover_subarrays_with_annot()
    info['annotBySpot_count'] = len(annot_subarrays)
    print(f'annotBySpot.RDS: {len(annot_subarrays)} subarrays')

    # Clinical
    clin_path = DATA / 'Clinical' / 'Clinical.RDS'
    info['clinical_path'] = str(clin_path) if clin_path.exists() else None
    print(f'Clinical.RDS: {"present" if clin_path.exists() else "MISSING"}')

    # signatures (for TLS)
    sig_path = DATA / 'misc' / 'signatures.RDS'
    info['signatures_path'] = str(sig_path) if sig_path.exists() else None
    print(f'signatures.RDS: {"present" if sig_path.exists() else "MISSING"}')

    # selection.RData (for raw counts)
    sel_count = sum(1 for s in annot_subarrays if Path(s['rdata_path']).exists())
    info['selection_RData_count'] = sel_count
    print(f'selection.RData: {sel_count} subarrays (for TLS scoring)')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'inventory.json', 'w') as f:
        json.dump(info, f, indent=2)
    return info


def extract_morphology():
    """step 6: extract 5-class morphology labels per spot."""
    print('\n=== morphology extraction ===')
    subarrays = discover_subarrays_with_annot()
    print(f'processing {len(subarrays)} subarrays with annotBySpot.RDS')

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    class_counts_total = defaultdict(int)
    background_dropped_total = 0

    for i, sub in enumerate(subarrays):
        # load via Rscript -> CSV
        r = subprocess.run(['Rscript', '-e', f'''
            d <- readRDS("{sub["annot_path"]}")
            write.csv(d, stdout())
        '''], capture_output=True, text=True)
        if r.returncode != 0:
            print(f'  [{i+1}/{len(subarrays)}] {sub["key"]}: ERROR loading: {r.stderr[:100]}')
            continue

        df = pd.read_csv(io.StringIO(r.stdout), index_col=0)
        n_total = len(df)

        # check that all 18 expected columns are present
        expected = (TUMOR_COLS + STROMA_COLS + IMMUNE_COLS + OTHER_COLS +
                    BACKGROUND_COLS)
        # also "Heterologous elements" -> may be misnamed
        missing = [c for c in expected if c not in df.columns]
        if missing:
            # try alternative spellings (R may convert special chars)
            for col in df.columns:
                stripped = col.strip('"').replace('.', ' ').replace('  ', ' ')
                if stripped != col:
                    df = df.rename(columns={col: stripped})
            missing = [c for c in expected if c not in df.columns]
            if missing:
                print(f'  [{i+1}/{len(subarrays)}] {sub["key"]}: '
                      f'WARNING missing columns: {missing}')

        # row sums (should be ~10557 - constant pixel area per spot)
        row_total = df[expected].sum(axis=1).clip(lower=1)

        # background fraction
        bg_pixels = df[BACKGROUND_COLS].sum(axis=1)
        bg_frac = bg_pixels / row_total

        # filter background-dominated spots
        keep = bg_frac <= BACKGROUND_THRESHOLD
        n_dropped = (~keep).sum()
        background_dropped_total += n_dropped
        df = df[keep].copy()
        row_total = row_total[keep]

        if len(df) == 0:
            print(f'  [{i+1}/{len(subarrays)}] {sub["key"]}: 0/{n_total} spots after filter')
            continue

        # tissue pixels (denominator excludes background)
        tissue_cols = TUMOR_COLS + STROMA_COLS + IMMUNE_COLS + OTHER_COLS
        tissue_pixels = df[tissue_cols].sum(axis=1).clip(lower=1)

        # fractions per 18-class on tissue-only denominator
        tissue_fracs = df[tissue_cols].div(tissue_pixels, axis=0)

        # dominant 18-class category and its fraction
        dominant_18class = tissue_fracs.idxmax(axis=1)
        dominant_18frac = tissue_fracs.max(axis=1)

        # collapse to 5-class
        dominant_5class = dominant_18class.map(CLASS_MAP)
        # apply mixed threshold
        dominant_5class = dominant_5class.where(dominant_18frac >= MIXED_THRESHOLD, 'mixed')

        # build output rows
        for spot_id in df.index:
            row = {
                'subarray_id': sub['key'],
                'spot_id': spot_id,
                'dominant_5class': dominant_5class[spot_id],
                'dominant_18class': dominant_18class[spot_id],
                'dominant_18class_fraction': float(dominant_18frac[spot_id]),
                'background_fraction': float(bg_frac[spot_id]),
            }
            # all 18 raw fractions on full denominator (for downstream use)
            full_fracs = (df.loc[spot_id, tissue_cols + BACKGROUND_COLS].values /
                          row_total[spot_id])
            for col, frac in zip(tissue_cols + BACKGROUND_COLS, full_fracs):
                row[f'frac_{col}'] = float(frac)
            all_rows.append(row)

        # update class counts
        for cls, n in dominant_5class.value_counts().items():
            class_counts_total[cls] += n

        if (i + 1) % 10 == 0 or i == 0:
            print(f'  [{i+1}/{len(subarrays)}] {sub["key"]}: '
                  f'{len(df)}/{n_total} spots ({n_dropped} dropped as background)')

    # save TSV
    out_df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / 'morphology_labels.tsv'
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f'\nsaved {len(out_df)} spots from {out_df["subarray_id"].nunique()} '
          f'subarrays -> {out_path.relative_to(ROOT)}')
    print(f'background-dropped: {background_dropped_total} spots')

    print(f'\n5-class distribution:')
    total = sum(class_counts_total.values())
    for cls in ['tumor', 'stroma', 'immune', 'other', 'mixed']:
        n = class_counts_total.get(cls, 0)
        print(f'  {cls:10s}: {n:6d} ({n/total*100:.1f}%)')

    # warn if class imbalance is severe
    counts = [class_counts_total.get(c, 0) for c in ['tumor', 'stroma', 'immune', 'other', 'mixed']]
    if max(counts) / max(min(counts), 1) > 50:
        print('\nWARNING: severe class imbalance detected. '
              'use stratified sampling for rank 1a.')

    return out_df


def extract_clinical():
    """step 7: extract patient-level labels from Clinical.RDS."""
    print('\n=== clinical labels ===')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(['Rscript', '-e', '''
        clin <- readRDS("data/inputs/Clinical/Clinical.RDS")
        # rename problem columns
        write.csv(clin, stdout())
    '''], capture_output=True, text=True)
    if r.returncode != 0:
        print(f'ERROR: {r.stderr[:300]}')
        return None

    df = pd.read_csv(io.StringIO(r.stdout), index_col=0)
    print(f'loaded clinical: {df.shape[0]} patients, {df.shape[1]} columns')

    # extract relevant columns
    relevant = {
        'ST_TNBC_ID': 'patient_id',
        'Spatial archetypes_defined_on_ST_global_pseudobulk': 'spatial_archetype',
        'Bareche_molecular_subtype_defined_on_global_pseudobulk': 'molecular_subtype_global',
        'Bareche_molecular_subtype_defined_on_tumor_pseudobulk': 'molecular_subtype_tumor',
        'Bareche_molecular_subtype_defined_on_stroma_pseudobulk': 'molecular_subtype_stroma',
        'TIME_classes.by.pathologist': 'TIME_pathologist',
        'TIME_classes_expression_global_pseudobulk': 'TIME_expression_global',
        'TIME_classes_expression_bulk': 'TIME_expression_bulk',
    }

    out = pd.DataFrame()
    for src, dst in relevant.items():
        if src in df.columns:
            out[dst] = df[src].values
        else:
            print(f'  WARNING: missing column {src}')

    out_path = OUT_DIR / 'clinical_labels.tsv'
    out.to_csv(out_path, sep='\t', index=False)
    print(f'saved {len(out)} patient labels -> {out_path.relative_to(ROOT)}')

    print(f'\nspatial_archetype distribution:')
    print(out['spatial_archetype'].value_counts().sort_index())
    print(f'\nmolecular_subtype_global distribution:')
    print(out['molecular_subtype_global'].value_counts())

    return out


def extract_tls_scores():
    """step 8: compute TLS signature score per spot."""
    print('\n=== TLS scores ===')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # TLS signature genes (Wang et al. supp table 13, classic 30-gene panel)
    # source: PMID 31942071, Cabrita et al. Nature 2020
    TLS_GENES = [
        'CCL19', 'CCL21', 'CXCL13', 'CCR7', 'CXCR5',
        'SELL', 'LAMP3', 'CD79A', 'CD79B', 'MS4A1',
        'CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD8B',
        'TRAC', 'TRBC1', 'TRBC2', 'BCL6', 'PAX5',
        'TCL1A', 'BANK1', 'IGHM', 'IGHD', 'CXCL12',
        'LTB', 'TNFRSF13C', 'POU2AF1', 'BLK', 'FCRL5'
    ]

    # gene mapping (Ensembl -> symbol)
    map_path = DATA / 'ensembl_to_symbol.tsv'
    if not map_path.exists():
        print(f'  ERROR: {map_path} not found')
        return None
    gene_map = pd.read_csv(map_path, sep='\t')
    symbol_to_ensembl = {}
    for _, row in gene_map.iterrows():
        sym = row.iloc[1]
        eid = row.iloc[0].split('.')[0]
        symbol_to_ensembl.setdefault(sym, []).append(eid)

    tls_ensembl = set()
    for sym in TLS_GENES:
        for eid in symbol_to_ensembl.get(sym, []):
            tls_ensembl.add(eid)
    print(f'TLS genes: {len(TLS_GENES)} symbols -> {len(tls_ensembl)} Ensembl IDs')

    subarrays = discover_subarrays_with_annot()
    print(f'computing TLS scores for {len(subarrays)} subarrays...')

    all_rows = []
    for i, sub in enumerate(subarrays):
        if not Path(sub['rdata_path']).exists():
            continue
        # load raw counts
        r = subprocess.run(['Rscript', '-e', f'''
            load("{sub["rdata_path"]}")
            write.csv(cnts, stdout())
        '''], capture_output=True, text=True)
        if r.returncode != 0:
            print(f'  [{i+1}/{len(subarrays)}] {sub["key"]}: load failed')
            continue

        try:
            df = pd.read_csv(io.StringIO(r.stdout), index_col=0)
        except Exception as e:
            print(f'  [{i+1}/{len(subarrays)}] {sub["key"]}: parse failed: {e}')
            continue

        # find TLS gene columns (versioned ensembl IDs)
        tls_cols = []
        for col in df.columns:
            base = col.split('.')[0]
            if base in tls_ensembl:
                tls_cols.append(col)

        if len(tls_cols) == 0:
            print(f'  [{i+1}/{len(subarrays)}] {sub["key"]}: no TLS genes found')
            continue

        # log1p of count-normalized expression
        counts = df[tls_cols].values.astype(np.float32)
        # normalize per spot to total counts
        spot_total = df.values.sum(axis=1, keepdims=True).clip(min=1)
        normed = counts / spot_total * 10000
        log_normed = np.log1p(normed)
        # TLS score = mean log-normalized expression of TLS genes
        tls_score = log_normed.mean(axis=1)

        for spot_id, score in zip(df.index, tls_score):
            all_rows.append({
                'subarray_id': sub['key'],
                'spot_id': spot_id,
                'tls_score': float(score),
                'n_tls_genes_present': len(tls_cols),
            })

        if (i + 1) % 20 == 0 or i == 0:
            print(f'  [{i+1}/{len(subarrays)}] {sub["key"]}: '
                  f'{len(df)} spots, {len(tls_cols)}/{len(TLS_GENES)} genes')

    out_df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / 'tls_scores.tsv'
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f'\nsaved {len(out_df)} TLS scores -> {out_path.relative_to(ROOT)}')
    print(f'TLS score stats: mean={out_df["tls_score"].mean():.3f}, '
          f'median={out_df["tls_score"].median():.3f}, '
          f'max={out_df["tls_score"].max():.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['inventory', 'morphology', 'clinical', 'tls'],
                        default=None)
    args = parser.parse_args()

    if args.step == 'inventory':
        inventory()
    elif args.step == 'morphology':
        extract_morphology()
    elif args.step == 'clinical':
        extract_clinical()
    elif args.step == 'tls':
        extract_tls_scores()
    else:
        # run all in order
        inventory()
        extract_clinical()
        extract_morphology()
        extract_tls_scores()
        print('\n=== Block 2 complete ===')
        print(f'output: {OUT_DIR.relative_to(ROOT)}/')
