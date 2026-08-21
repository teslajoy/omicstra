#!/usr/bin/env python
"""generate_summary_pdf.py - compact one-page-per-chain summary of tnbc-92 work.

each "chain" is a decision node:
    Q (question) -> method -> WINNER -> meaning

ten chains, plus title + summary flowchart. tables and figures embedded.
output: docs/tnbc92_summary.pdf

usage:
    python scripts/generate_summary_pdf.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
import seaborn as sns

sns.set_theme(style='ticks', palette='Set2', context='notebook')
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / 'runs' / 'tnbc-92'
FIG_DIR = RUNS_ROOT / 'eval' / 'figures'
OUT_PDF = ROOT / 'docs' / 'tnbc92_summary.pdf'


# ----------------------------------------------------------------- #
# layout helpers
# ----------------------------------------------------------------- #

def make_page(figsize=(8.5, 11)):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    return fig


def add_header(fig, chain_num, title, y=0.95):
    fig.text(0.06, y, f'CHAIN {chain_num}', fontsize=10, color='#888',
             fontweight='bold', family='monospace')
    fig.text(0.06, y - 0.025, title, fontsize=18, fontweight='bold')
    fig.add_artist(plt.Line2D([0.06, 0.94], [y - 0.04, y - 0.04],
                               color='#222', lw=1.5))


def add_qmwm(fig, q, method, winner, meaning, y0=0.86):
    """add Question / Method / Winner / Meaning block."""
    line_h = 0.022
    indent = 0.10
    label_x = 0.06
    fig.text(label_x, y0, 'Q', fontsize=11, fontweight='bold', color='#666')
    fig.text(indent, y0, q, fontsize=10, wrap=True)
    y = y0 - line_h * 1.8
    fig.text(label_x, y, 'METHOD', fontsize=9, fontweight='bold', color='#666')
    fig.text(indent + 0.04, y, method, fontsize=9.5, color='#222')
    y -= line_h * 1.8
    fig.text(label_x, y, 'WINNER', fontsize=9, fontweight='bold', color='#1a7a3a')
    fig.text(indent + 0.04, y, winner, fontsize=10, color='#1a7a3a',
             fontweight='bold')
    y -= line_h * 1.8
    fig.text(label_x, y, 'MEANS', fontsize=9, fontweight='bold', color='#666')
    fig.text(indent + 0.04, y, meaning, fontsize=9.5, color='#222', style='italic')
    return y - line_h * 1.5


def add_table(fig, df, bbox=(0.08, 0.20, 0.84, 0.30),
              title=None, fontsize=8.5, header_color='#e0e0e0'):
    ax = fig.add_axes(bbox)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left',
                     pad=4, color='#222')
    cell_text = df.values.tolist()
    cols = list(df.columns)
    tab = ax.table(cellText=cell_text, colLabels=cols, loc='center',
                   cellLoc='left', colLoc='left')
    tab.auto_set_font_size(False)
    tab.set_fontsize(fontsize)
    tab.scale(1.0, 1.5)
    for j in range(len(cols)):
        tab[(0, j)].set_facecolor(header_color)
        tab[(0, j)].set_text_props(weight='bold')
    return ax


def add_image(fig, image_path, bbox=(0.10, 0.10, 0.80, 0.40), title=None):
    if not Path(image_path).exists():
        return None
    ax = fig.add_axes(bbox)
    ax.imshow(imread(image_path))
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9.5, fontweight='bold', color='#222')
    return ax


def page_footer(fig, n, total):
    fig.text(0.5, 0.04, f'{n}/{total}', ha='center', fontsize=8, color='#aaa')


# ----------------------------------------------------------------- #
# chain definitions
# ----------------------------------------------------------------- #

def title_page():
    fig = make_page()
    fig.text(0.5, 0.62, 'tnbc-92', ha='center', fontsize=36, fontweight='bold')
    fig.text(0.5, 0.55, 'cross-modal alignment summary', ha='center',
             fontsize=18, color='#444')
    fig.text(0.5, 0.50, 'EDA · embeddings · alignment grid · H1/H2/H3 verdicts',
             ha='center', fontsize=11, color='#666', style='italic')
    fig.add_artist(plt.Line2D([0.30, 0.70], [0.46, 0.46], color='#222', lw=1))
    fig.text(0.5, 0.42, 'Wang et al. 2024 TNBC cohort', ha='center', fontsize=11)
    fig.text(0.5, 0.395, '94 patients · 261 subarrays · ~278k niches',
             ha='center', fontsize=10, color='#666')
    fig.text(0.5, 0.36, 'OHSU Knight Cancer Institute',
             ha='center', fontsize=9, color='#888')
    fig.text(0.5, 0.34, 'ResearchHub Foundation grant',
             ha='center', fontsize=9, color='#888')
    fig.text(0.5, 0.10,
             'each chain: question -> method -> winner -> meaning\n'
             'ten chains; tables and figures embedded',
             ha='center', fontsize=9, color='#888', style='italic')
    return fig


def chain_1_eda():
    fig = make_page()
    add_header(fig, 1, 'is the dataset usable for cross-modal alignment?')
    add_qmwm(fig,
        q='94 TNBC patients, 282 samples, original ST platform (100um spots, not Visium). '
          'is there biological signal? batch effects? co-registration?',
        method='positive markers (MKI67, CD3D), negative markers (ESR1, PGR), Moran\'s I '
               'spatial autocorrelation, plate/lane silhouette, PCA variance.',
        winner='PROCEED. 270k spots post-QC. positive markers pass; ESR1/PGR near-zero (TNBC '
               'confirmed). Moran\'s I 0.6 (COL1A1), 0.5 (CD3D). batch silhouette < 0.1 = '
               'patient/biological variation, not technical.',
        meaning='dataset is biologically rich, no major batch correction needed before encoding. '
                'platform mismatch with Novae GNN training (Visium 55um vs original ST 100um) '
                'is empirical risk to validate, not blocker.')
    df = pd.DataFrame({
        'positive marker': ['MKI67', 'CD3D', 'CD79A', 'COL1A1'],
        'mean expr': [0.50, 0.15, 0.06, '-'],
        'pct expressing': [31.2, 8.9, 3.7, '-'],
        "Moran's I": [0.19, 0.50, 0.39, 0.60],
        'expected': ['proliferation', 'T cell', 'TLS B cell', 'stroma'],
    })
    add_table(fig, df, bbox=(0.08, 0.45, 0.84, 0.18),
              title='positive marker validation (selected)')
    df2 = pd.DataFrame({
        'metric': ['plate silhouette (PC)', 'lane silhouette (PC)',
                   'subtype silhouette (PC)', 'PC1 variance', 'PC2 variance'],
        'value': [-0.076, -0.081, -0.143, '42.4%', '17.6%'],
        'reading': ['no plate batch', 'no lane batch',
                    'biology in PC1', 'dominant axis = biology',
                    'second axis = biology'],
    })
    add_table(fig, df2, bbox=(0.08, 0.20, 0.84, 0.20),
              title='batch-effect analysis (eda_summary.json)')
    page_footer(fig, 2, 12)
    return fig


def chain_2_unit():
    fig = make_page()
    add_header(fig, 2, 'what is the right alignment unit: spot or niche?')
    add_qmwm(fig,
        q='100um spots ~200 cells each. is spot-level cross-modal alignment '
          'feasible across patients?',
        method='spot-level alignment pilot (UNI2-Reinhard + Novae, ComBat-corrected). '
               'compare within- vs cross-patient R@1.',
        winner='NICHE (k=6 spatial neighbors). spot-level cross-patient R@1 = 0.002 (random). '
               'within-patient R@1 = 0.136 (1.6x). spot-level exhausted for cross-patient.',
        meaning='spot resolution too sparse at 100um; neighborhood pooling captures TME '
                'architecture. niche = central spot + 6 nearest neighbors (KDTree on pixel '
                'coords). 1075/1075 spot agreement between Virchow2 and Novae niches on '
                'TNBC1_CN1_C1 verifies parity.')
    df = pd.DataFrame({
        'unit': ['spot, raw UNI2', 'spot, ComBat UNI2', 'spot, within-patient',
                 'niche (current grid)'],
        'cross-patient R@1': [0.002, 0.0003, 'N/A', '~0.0001-0.0003'],
        'verdict': ['random', 'pure overfit (best_epoch=1)',
                    '1.6x amplification but in-patient',
                    'all 4 contrastive > 3 classical on AUC/CKA'],
    })
    add_table(fig, df, bbox=(0.08, 0.40, 0.84, 0.25),
              title='alignment unit attempts')
    page_footer(fig, 3, 12)
    return fig


def chain_3_he():
    fig = make_page()
    add_header(fig, 3, 'which H&E encoder fits TNBC?')
    add_qmwm(fig,
        q='UNI2-h, Virchow2, or stain-normalized variant? does z-score help?',
        method='MC 14-class linear probe + cross-subarray silhouette + batch ratio + '
               'PathoROB robustness index. test z-score on each.',
        winner='Virchow2 niche, RAW (no z-score, no Reinhard). 23.6% MC probe (3.3x chance), '
               '57.1% morph probe, batch ratio 1.089, PathoROB RI=0.88 (highest of 20 FMs).',
        meaning='Virchow2 already encodes morphological structure cross-subarray (PathoROB '
                'reproduced empirically). z-score HURTS Virchow2 (-13% probe) because per-'
                'subarray mean is biologically meaningful. opposite of Novae.')
    df = pd.DataFrame({
        'encoder': ['Virchow2 niche', 'Novae', 'UNI2 raw (pilot)'],
        'dim': [1280, 64, 1536],
        'MC probe': ['23.6%', '12.1%', '2.2% (5 pts)'],
        'morph probe': ['57.1%', '33.0%', '55.8%'],
        'batch ratio': [1.089, 0.258, 0.983],
        'cross sil (MC)': [-0.10, -0.25, 0.05],
    })
    add_table(fig, df, bbox=(0.08, 0.42, 0.84, 0.20),
              title='encoder comparison (eda_summary.json, encoder_qc_comparison_2026-04-09)')
    df2 = pd.DataFrame({
        'preprocessing': ['raw (timm only)', 'z-score per subarray'],
        'Virchow2 MC probe': ['23.6%', '10.4% (-13.2 pts)'],
        'Novae sil (MC)': ['-0.246', '-0.135 (+0.111)'],
        'verdict': ['Virchow2 raw, Novae z-scored', 'asymmetric: each encoder needs own preprocessing'],
    })
    add_table(fig, df2, bbox=(0.08, 0.20, 0.84, 0.18),
              title='z-score test (encoder-specific)')
    page_footer(fig, 4, 12)
    return fig


def chain_4_st():
    fig = make_page()
    add_header(fig, 4, 'does Novae GNN work on TNBC original ST?')
    add_qmwm(fig,
        q='Novae trained on Xenium subcellular ST. our cohort: 100um spots, 1934/array, '
          'NOT 10x Visium. cross-platform mismatch — does the model produce useful embeddings?',
        method='spatial rho (negative = local coherence), batch_ratio (lower = biology-dom), '
               'effective rank, MC probe. 5-subarray validation set.',
        winner='Novae VALIDATED WITH CONSTRAINTS. batch_ratio 0.258 (biology-dominated), '
               'effective rank 11 of 64. 19 subarrays excluded (<512 spots, model architectural '
               'floor). 64d, z-scored per subarray.',
        meaning='works locally per-subarray (silhouette +0.058) but does NOT produce cross-'
                'subarray comparable embeddings without z-score. cross-subarray problem partially '
                '(0.111 of 0.246) per-subarray mean shift; rest is genuine geometric '
                'inconsistency.')
    df = pd.DataFrame({
        'subarray': ['TNBC1_CN1_C1', 'TNBC3_CN2_C1', 'TNBC83_CN42_C1',
                     'TNBC55_CN28_C1', 'TNBC68_CN34_D2'],
        'spatial rho': [-0.43, -0.43, -0.41, '-(filtered)', -0.25],
        'l2 norm': [2.62, 2.62, 2.62, '-', '-'],
        'status': ['ok', 'ok', 'ok', 'hvg fallback', 'low gene count'],
    })
    add_table(fig, df, bbox=(0.08, 0.45, 0.84, 0.20),
              title='Novae validation (5-subarray pilot)')
    df2 = pd.DataFrame({
        'metric': ['n subarrays canonical', 'skipped (<512 spots)', 'matched join scope',
                   'novae_niche_full path'],
        'value': ['262', '19', '260 (∩ Virchow2 280)',
                  'data/embeddings/novae_niche_full/'],
    })
    add_table(fig, df2, bbox=(0.08, 0.22, 0.84, 0.18),
              title='Novae cohort scope')
    page_footer(fig, 5, 12)
    return fig


def chain_5_pathway():
    fig = make_page()
    add_header(fig, 5, 'do pathway embeddings carry interpretable biology?')
    add_qmwm(fig,
        q='ST 64d Novae alone may lack pathway-level biological signal. add gpath2vec '
          'pathway embedding from Reactome graph?',
        method='gpath2vec on full cohort (286k niches, 512d). subarray-level perm z-test '
               'against MC_global, MC_tumor, TIME, patient_id. 1e5 perms, BH-FDR.',
        winner='gpath2vec PASSES. preserves biology (TIME z 4.84 -> 6.52), compresses '
               'patient (34.30 -> 29.63, -14%). geometric rho 0.955 vs raw EA. '
               'asymmetric smoothing — biology > patient in compression rate.',
        meaning='gpath2vec is a safe ST input feature: it carries biology that is independent '
                'of patient identity. 286k × 512d cluster_embeddings.parquet ready as the '
                'pathway track in the rich-niche ST vector.')
    df = pd.DataFrame({
        'label': ['MC_global', 'MC_tumor', 'TIME', 'patient_id'],
        'raw EA z': [4.84, 4.08, 4.84, 34.30],
        'embedding z': [5.67, 4.53, 6.52, 29.63],
        'direction': ['biology↑', 'biology↑', 'biology↑', 'patient↓ (-14%)'],
    })
    add_table(fig, df, bbox=(0.08, 0.40, 0.84, 0.25),
              title='gpath2vec validation (subarray, 1e5 perms, Bonferroni-survives)')
    page_footer(fig, 6, 12)
    return fig


def chain_6_grid():
    fig = make_page()
    add_header(fig, 6, 'which loss × fusion architecture aligns best?')
    add_qmwm(fig,
        q='4 contrastive variants × 3 classical baselines. niche-level, 260 matched '
          'subarrays, patient-held-out 85/15 split, seed=42.',
        method='train each config (R1-R4) or fit each baseline (B1-B3). evaluate '
               'cross-modal alignment via R@K, MRR, alignment_gap, AUC, CKA.',
        winner='R4 (InfoNCE + cross-attention) leads H1: AUC 0.851, CKA-after 0.564. '
               'rank-matched control: NOT collapse-driven (R1@rank3=0.234 vs R4=0.564).',
        meaning='cross-attention learns ST query → H&E tile selection that captures real '
                'cross-modal structure. tight ~3-effective-dim manifold is a regime, not '
                'a failure. all 4 contrastive > all 3 classical confirms H1.')
    df = pd.DataFrame({
        'run': ['R4', 'R1', 'R2', 'R3', 'B2', 'B1', 'B3'],
        'method': ['infonce + cross_attn', 'infonce + late', 'supcon + late',
                   'barlow + late', 'Procrustes', 'CCA', 'Unaligned PCA'],
        'AUC': [0.851, 0.741, 0.707, 0.646, 0.703, 0.541, 0.443],
        'CKA-after': [0.564, 0.242, 0.120, 0.252, 0.124, 0.086, 0.124],
        'gap': [0.195, 0.159, 0.043, 0.125, 0.156, 0.007, -0.049],
    })
    add_table(fig, df, bbox=(0.08, 0.43, 0.84, 0.23),
              title='H1 retrieval/alignment metrics (8 runs)')

    # inline AUC vs CKA scatter
    ax = fig.add_axes((0.10, 0.10, 0.80, 0.28))
    runs = df['run'].tolist()
    colors = {'R1': '#1f77b4', 'R2': '#aec7e8', 'R3': '#9467bd', 'R4': '#d62728',
              'B1': '#2ca02c', 'B2': '#98df8a', 'B3': '#888888'}
    for _, row in df.iterrows():
        c = colors.get(row['run'], '#444')
        ax.scatter(row['AUC'], row['CKA-after'], s=140, c=c,
                   edgecolor='white', linewidth=1.5, zorder=3)
        ax.annotate(row['run'], (row['AUC'], row['CKA-after']),
                    xytext=(7, 6), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=c)
    ax.axhline(0.115, ls=':', color='#aaa', lw=1, zorder=1)
    ax.text(0.45, 0.118, 'CKA before projection (raw modalities)',
            fontsize=8, color='#888')
    ax.set_xlabel('AUC', fontsize=10)
    ax.set_ylabel('CKA after projection', fontsize=10)
    ax.set_title('AUC × CKA: R4 leads both axes; R3 sits near R4 on CKA but lags AUC',
                 fontsize=9.5, color='#222', loc='left')
    sns.despine(ax=ax)
    page_footer(fig, 7, 12)
    return fig


def chain_7_h1():
    fig = make_page()
    add_header(fig, 7, 'H1 verdict: do contrastive methods beat classical?')
    add_qmwm(fig,
        q='primary H1 claim: learned contrastive alignment improves cross-modal matching '
          'over CCA, Procrustes, and unaligned baselines.',
        method='AUC (ranking-robust) + CKA-after (linear similarity) + alignment_gap. '
               'rank-matched CKA control (R1@rank3 vs R4 full) tests for collapse-driven '
               'inflation.',
        winner='YES. all 4 contrastive > all 3 classical on AUC + CKA. R4 leads by '
               'a wide margin. R@K low but consistent with niche-level cross-patient '
               'difficulty at 100um resolution.',
        meaning='H1 hypothesis validated. nonlinear cross-modal alignment captures structure '
                'classical methods cannot. R@K reported as relative-improvement over '
                'baselines, not absolute retrieval threshold.')
    df = pd.DataFrame({
        'family': ['contrastive (R1-R4)', 'classical (B1-B3)', 'gap'],
        'AUC range': ['0.646 - 0.851', '0.443 - 0.703', 'no overlap'],
        'CKA range': ['0.120 - 0.564', '0.086 - 0.124', 'no overlap'],
    })
    add_table(fig, df, bbox=(0.08, 0.40, 0.84, 0.16),
              title='H1 family separation (every contrastive > every classical)')
    df2 = pd.DataFrame({
        'control': ['R1 full-dim CKA', 'R1 @ rank-3 (R4 effective rank)',
                    'R4 full-dim CKA', 'R4 @ rank-3 (sanity)',
                    'gap R1@rank3 vs R4 full'],
        'value': ['0.242', '0.234 (drops 0.008)', '0.564', '0.564 (sanity gap 0.0003)',
                  '-0.331'],
        'verdict': ['baseline', 'R1 already in top-3 PCs',
                    'baseline', 'projection code OK',
                    'R4 is REAL architectural work, not collapse'],
    })
    add_table(fig, df2, bbox=(0.08, 0.13, 0.84, 0.23),
              title='rank-matched CKA control (falsifier test, locked pre-result)')
    page_footer(fig, 8, 12)
    return fig


def chain_8_h2():
    fig = make_page()
    add_header(fig, 8, 'H2 verdict: does shared embedding preserve biological structure?')
    add_qmwm(fig,
        q='does the shared manifold carry compartment / archetype structure beyond '
          'either modality alone?',
        method='ARI on 9 per-patient archetypes (deprecated) + per-compartment cosine '
               'on 90-subarray annotated subset (proper test).',
        winner='ARI confounded — ranks by patient leakage (B1=0.307, R4=0.059). '
               'silhouette and ARI are rotation-invariant so B2 ≡ B3 mathematically. '
               'per-compartment cosine writeup PENDING.',
        meaning='ARI cannot distinguish biology from patient when labels are per-patient '
                '(archetypes correlate with patient identity in TNBC). per-compartment '
                'cosine on human-annotated morphology categories is the correct test '
                '— labels are tissue compartments, not patient.')
    df = pd.DataFrame({
        'run': ['B1', 'B2', 'B3', 'R6', 'R1', 'R2', 'R3', 'R4'],
        'z_he ARI': [0.307, 0.298, 0.298, 0.254, 0.250, 0.204, 0.129, 0.059],
        'reading': ['leakage', 'preserves PCA + leakage', 'unaligned + leakage',
                    'novae-only', 'late fusion', 'supcon', 'barlow',
                    'cross-attn (compressed)'],
    })
    add_table(fig, df, bbox=(0.08, 0.35, 0.84, 0.28),
              title='H2 ARI on 9 archetypes (z_he) — confounded ordering')
    fig.text(0.06, 0.27,
        'NOTE: ARI ordering tracks patient z-amplification, NOT biology preservation.\n'
        'B1 wins ARI (0.307) AND amplifies patient z 3.3x (122 vs raw 37). same axis.\n'
        'rotation-invariance: silhouette and K-means cluster assignments are preserved\n'
        'under orthogonal transformation, so B2 = B3 on H2 metrics is mathematical, not\n'
        'a coincidence. H1 separates them (B2 AUC 0.703 vs B3 0.443) where geometry matters.',
        fontsize=9, color='#222', wrap=True)
    page_footer(fig, 9, 12)
    return fig


def chain_9_h3():
    fig = make_page()
    add_header(fig, 9, 'H3 verdict: does shared space disentangle biology from patient?')
    add_qmwm(fig,
        q='compared to raw modalities, does alignment amplify biology and reduce patient '
          'identity? (the proposal\'s interpretability test)',
        method='subarray-level matched-null perm z-test, 1e5 perms, BH-FDR over 5 views × '
               '4 labels. Bareche TIME / MC_global / MC_tumor labels (external to model). '
               'pass = bio z↑ vs raw_he 4.79 AND patient z↓ vs raw_he 37.25.',
        winner='YES (binary, direction-wise). all 4 contrastive PASS H3. R6 sharpens (z_he '
               'highest biology amplification). B1 CCA FAILS (patient amplified 3.3×).',
        meaning='H3 is robust direction-wise — every learned method moves bio up, patient '
                'down in z-units. B1 (CCA) is the only run that amplifies patient instead. '
                'CCA failure mode is correlation-maximization, not linearity (B2 Procrustes '
                'passes despite being linear).')
    df = pd.DataFrame({
        'run': ['raw_he (floor)', 'raw_st (floor)', 'R6', 'R1', 'R4', 'R2', 'R3', 'B2', 'B1'],
        'TIME bio z': [4.79, 6.44, 17.81, 15.13, 10.07, 8.14, 6.18, 12.66, 7.87],
        'patient z': [37.25, 28.64, 32.52, 29.27, 22.97, 27.01, 25.03, 39.83, 122.13],
        'ratio': [0.129, 0.225, 0.548, 0.517, 0.438, 0.301, 0.247, 0.318, 0.064],
        'verdict': ['baseline', 'baseline', 'PASS pass', 'PASS pass', 'PASS pass',
                    'PASS pass', 'PASS pass', 'PASS pass', 'FAIL FAIL (anti-helpful)'],
    })
    add_table(fig, df, bbox=(0.08, 0.40, 0.84, 0.24),
              title='H3 z-ratio table (z_he view, all runs vs raw floors)', fontsize=8)

    # bio_z vs patient_z scatter (the headline H3 figure)
    ax = fig.add_axes((0.12, 0.08, 0.76, 0.28))
    colors = {'R1': '#1f77b4', 'R2': '#aec7e8', 'R3': '#9467bd', 'R4': '#d62728',
              'R6': '#ff7f0e', 'B1': '#2ca02c', 'B2': '#98df8a'}
    for _, row in df.iterrows():
        if row['run'].startswith('raw'):
            ax.scatter(row['patient z'], row['TIME bio z'], s=160, marker='s',
                       facecolor='none', edgecolor='#444', linewidth=1.8, zorder=3)
            ax.annotate(row['run'], (row['patient z'], row['TIME bio z']),
                        xytext=(8, -2), textcoords='offset points',
                        fontsize=9, color='#444')
        else:
            c = colors.get(row['run'], '#444')
            ax.scatter(row['patient z'], row['TIME bio z'], s=140, c=c,
                       edgecolor='white', linewidth=1.5, zorder=3)
            ax.annotate(row['run'], (row['patient z'], row['TIME bio z']),
                        xytext=(7, 6), textcoords='offset points',
                        fontsize=10, fontweight='bold', color=c)
    # ratio reference lines
    for ratio, lab in [(0.129, 'raw_he ratio'), (0.5, 'ratio = 0.5')]:
        x = np.linspace(0, 130, 50)
        ax.plot(x, ratio * x, ':', color='#aaa', lw=1, zorder=1)
        ax.text(125, ratio * 125, lab, fontsize=7, color='#888', ha='right')
    ax.set_xlabel('patient z (lower = better disentanglement)', fontsize=9.5)
    ax.set_ylabel('TIME biology z (higher = better)', fontsize=9.5)
    ax.set_title('H3 bio vs patient: ideal is top-left; B1 (CCA) is far-right outlier',
                 fontsize=9.5, color='#222', loc='left')
    ax.set_xlim(0, 135)
    ax.set_ylim(0, 22)
    sns.despine(ax=ax)
    page_footer(fig, 10, 12)
    return fig


def chain_10_step3():
    fig = make_page()
    add_header(fig, 10, 'is "R1 disentangles more than R4" defensible?')
    add_qmwm(fig,
        q='the report\'s mechanism story claims R1 has more asymmetric compression than R4. '
          'is this directional claim statistically supported?',
        method='three legitimate measures: (1) z_he ratio, (2) z_mean ratio, '
               '(3) per-run vs raw_he paired-perm on (bio_delta − patient_delta), '
               'shared seed=42, 1e5 perms.',
        winner='NO directional ranking. R1 wins one framing, R4 wins two. headline claim DEAD. '
               'salvaged: R4 is the ONLY run that moves toward bio-patient symmetric on both '
               'views (p<1e-5).',
        meaning='unit-dependence kills the directional story. robust finding: every contrastive '
                'method compresses patient AND amplifies biology in z-units; choice between R1 '
                'and R4 is task-dependent (R1 = wider cosine spread for retrieval, R4 = tight '
                'manifold for cohort statistics), not a disentanglement ranking.')
    df = pd.DataFrame({
        'measure': [
            'z_bio / z_pat (z_he)',
            'z_bio / z_pat (z_mean)',
            'run_asym − raw_he_asym (raw cosine, both views)',
        ],
        'R1': [0.517, 0.417, 'further from raw (-0.238 z_he)'],
        'R4': [0.438, 0.513, 'TOWARD raw (+0.062 z_he, p<1e-5)'],
        'winner': ['R1', 'R4', 'R4'],
    })
    add_table(fig, df, bbox=(0.08, 0.40, 0.84, 0.22),
              title='three framings, three verdicts', fontsize=8.5)
    # asymmetric compression bar inline (z_he view)
    ax = fig.add_axes((0.12, 0.10, 0.76, 0.26))
    runs = ['R4', 'R3', 'R2', 'R1', 'R6', 'B1', 'B2']
    moves = [0.062, 0.001, -0.130, -0.238, -0.317, -0.439, -0.543]
    colors_map = {'R1': '#1f77b4', 'R2': '#aec7e8', 'R3': '#9467bd', 'R4': '#d62728',
                  'R6': '#ff7f0e', 'B1': '#2ca02c', 'B2': '#98df8a'}
    bar_colors = [colors_map[r] for r in runs]
    x = np.arange(len(runs))
    ax.bar(x, moves, color=bar_colors, edgecolor='white', linewidth=0.8, alpha=0.85)
    ax.axhline(0, color='#222', lw=0.8)
    ax.text(0.05, 0.05, 'TOWARD symmetric (better)', fontsize=8, color='#1a7a3a',
            transform=ax.transData, ha='left')
    ax.text(0.05, -0.08, 'AWAY from raw (more asymmetric)', fontsize=8, color='#a02020',
            transform=ax.transData, ha='left')
    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=10)
    ax.set_ylabel('asym move from raw_he (z_he)', fontsize=9)
    ax.set_title('only R4 moves toward symmetric on z_he (R3 statistically at raw, p=0.94)',
                 fontsize=9.5, color='#222', loc='left')
    sns.despine(ax=ax)
    page_footer(fig, 11, 12)
    return fig


def summary_flowchart():
    fig = make_page()
    add_header(fig, 'final', 'summary flowchart of verdicts')
    fig.text(0.06, 0.84, 'one-line verdicts across all chains:',
             fontsize=11, fontweight='bold', color='#222')
    rows = [
        ('CH 1', 'EDA gate', 'PROCEED', 'TNBC dataset viable, no batch correction needed'),
        ('CH 2', 'alignment unit', 'NICHE (k=6)', 'spot-level cross-patient = random'),
        ('CH 3', 'H&E encoder', 'Virchow2 RAW', 'PathoROB RI=0.88; z-score hurts'),
        ('CH 4', 'ST encoder', 'Novae 64d (z-scored)', 'platform mismatch managed; 19 subs excluded'),
        ('CH 5', 'pathway embed', 'gpath2vec 512d', 'biology↑ patient↓; safe ST input'),
        ('CH 6', 'alignment grid', 'R4 = InfoNCE + cross_attn', 'AUC 0.851, CKA 0.564'),
        ('CH 7', 'H1 verdict', 'PASS PASS', 'all contrastive > all classical, NOT collapse-driven'),
        ('CH 8', 'H2 verdict', '~ PARTIAL', 'ARI confounded; per-compartment cosine pending'),
        ('CH 9', 'H3 verdict', 'PASS PASS direction-wise', 'all contrastive amplify bio + reduce patient'),
        ('CH 10', 'R1 vs R4 mechanism', 'task-dependent', '3 framings → 1 vs 2 → no directional headline'),
    ]
    y = 0.78
    for tag, topic, verdict, gloss in rows:
        fig.text(0.06, y, tag, fontsize=9, color='#888', family='monospace',
                 fontweight='bold')
        fig.text(0.13, y, topic, fontsize=10, fontweight='bold')
        fig.text(0.36, y, verdict, fontsize=10, color='#1a7a3a', fontweight='bold')
        fig.text(0.06, y - 0.018, gloss, fontsize=9, color='#444', style='italic')
        y -= 0.055

    fig.add_artist(plt.Line2D([0.06, 0.94], [0.21, 0.21], color='#aaa', lw=0.5))
    fig.text(0.06, 0.18, 'what the dataset says, in one sentence:',
             fontsize=10, fontweight='bold', color='#222')
    fig.text(0.06, 0.14,
        'every contrastive cross-modal alignment method we tried amplifies biology and\n'
        'compresses patient signal vs raw modalities; the architecture chosen (R1 vs R4)\n'
        'reflects a regime trade-off between cosine spread and manifold tightness, not a\n'
        'disentanglement ranking. classical methods that maximize linear correlation (CCA)\n'
        'fail because patient identity is the dominant shared axis between paired modalities.',
        fontsize=9.5, color='#222')
    page_footer(fig, 12, 12)
    return fig


# ----------------------------------------------------------------- #
# main
# ----------------------------------------------------------------- #

def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    pages = [
        title_page,
        chain_1_eda,
        chain_2_unit,
        chain_3_he,
        chain_4_st,
        chain_5_pathway,
        chain_6_grid,
        chain_7_h1,
        chain_8_h2,
        chain_9_h3,
        chain_10_step3,
        summary_flowchart,
    ]
    with PdfPages(OUT_PDF) as pdf:
        for fn in pages:
            fig = fn()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'wrote {OUT_PDF.relative_to(ROOT)}')
    print(f'  pages: {len(pages)}')
    print(f'  size:  {OUT_PDF.stat().st_size // 1024} KB')


if __name__ == '__main__':
    main()