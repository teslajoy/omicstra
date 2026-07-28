"""
extract_virchow2_niche.py - Virchow2 niche embeddings for all subarrays.

for each spot: find k=6 nearest spatial neighbors (including self),
extract HD patches for each, run Virchow2 CLS extraction,
mean pool -> (1280,) niche embedding per spot.

boundary spots with fewer than k neighbors use available neighbors only
(option B - don't discard real tissue, log neighbor_count in meta).

usage:
    python scripts/extract_virchow2_niche.py
    python scripts/extract_virchow2_niche.py --step check

outputs:
    data/embeddings/virchow2_niche/{subarray_id}.npy      - (n_spots, 1280)
    data/embeddings/virchow2_niche/{subarray_id}_meta.tsv  - spot coords + neighbor count
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import time
import subprocess
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from scipy.spatial import KDTree

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'inputs'
BY_ARRAY = DATA / 'byArray'
HD_DIR = DATA / 'Images' / 'imagesHD'
EMB_DIR = ROOT / 'data' / 'embeddings'
OUT_DIR = EMB_DIR / 'virchow2_niche'
SEED = 42
K_NEIGHBORS = 6

# calibration constants (from UNI2 pilot notebook)
HE_SMALL_SIZE = 9523
HD_SIZE = 31744
SCALE_TO_HD = HD_SIZE / HE_SMALL_SIZE  # 3.3334


def discover_subarrays():
    """find all subarrays and map to HD images."""
    hd_lookup = {}
    slide_pos_to_tnbc = {}
    for p in HD_DIR.glob('*.jpg'):
        parts = p.stem.split('_')
        tnbc_id, slide, pos = parts[0], parts[1], parts[2]
        hd_lookup[p.stem] = p
        slide_pos_to_tnbc[(slide, pos)] = tnbc_id

    subarrays = []
    for slide_dir in sorted(BY_ARRAY.iterdir()):
        if not slide_dir.is_dir() or not slide_dir.name.startswith('CN'):
            continue
        for pos_dir in sorted(slide_dir.iterdir()):
            if not pos_dir.is_dir() or not (pos_dir / 'selection.RData').exists():
                continue
            slide = slide_dir.name
            pos = pos_dir.name
            tnbc_id = slide_pos_to_tnbc.get((slide, pos))
            key = f'{tnbc_id}_{slide}_{pos}' if tnbc_id else f'UNKNOWN_{slide}_{pos}'
            hd_path = hd_lookup.get(key)
            subarrays.append({
                'key': key,
                'tnbc_id': tnbc_id,
                'slide': slide,
                'pos': pos,
                'hd_path': str(hd_path) if hd_path else None,
                'rdata_path': str(pos_dir / 'selection.RData'),
            })

    print(f'discovered {len(subarrays)} subarrays, '
          f'{sum(1 for s in subarrays if s["hd_path"])} with HD images')
    return subarrays


def load_spots(rdata_path):
    """load spot coordinates from selection.RData."""
    r = subprocess.run(['Rscript', '-e', f'''
        load("{rdata_path}")
        write.csv(spots, stdout())
    '''], capture_output=True, text=True)
    assert r.returncode == 0, f'Rscript failed: {r.stderr[:200]}'
    return pd.read_csv(io.StringIO(r.stdout), index_col=0)


def extract_patches(hd_image_path, spot_df, patch_size_hd, output_size=224):
    """extract patches centered at spot coordinates from HD H&E image.
    coordinates are in HE-small space, scaled to HD internally."""
    img = Image.open(hd_image_path)
    w, h = img.size
    half = patch_size_hd // 2
    patches = []

    for _, row in spot_df.iterrows():
        cx = int(round(row['pixel_x'] * SCALE_TO_HD))
        cy = int(round(row['pixel_y'] * SCALE_TO_HD))
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        patch = img.crop((x1, y1, x2, y2))

        if patch.size != (patch_size_hd, patch_size_hd):
            padded = Image.new('RGB', (patch_size_hd, patch_size_hd), (0, 0, 0))
            padded.paste(patch, (half - (cx - x1), half - (cy - y1)))
            patch = padded

        patch = patch.resize((output_size, output_size), Image.LANCZOS)
        patches.append(patch)

    img.close()
    return patches


def sanity_check_neighbors(spots, key):
    """verify neighbor distances are in expected range before running at scale.
    spot coordinates are in HE-small space (~9523px).
    expected inter-spot distance: ~158px in HE-small (150um center-to-center / ~0.95 um/px).
    acceptable range: 100-300px in HE-small space."""
    coords = spots[['pixel_x', 'pixel_y']].values
    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=min(K_NEIGHBORS + 1, len(coords)))

    # nearest neighbor distances (exclude self at index 0)
    nn_dists = dists[:, 1]
    median_dist = np.median(nn_dists)
    min_dist = nn_dists.min()
    max_dist = nn_dists.max()

    print(f'  neighbor check ({key}): median={median_dist:.1f}px, '
          f'min={min_dist:.1f}px, max={max_dist:.1f}px (HE-small space)')

    # expected ~158px for 150um spacing at ~0.95 um/px
    if not (50 < median_dist < 500):
        raise ValueError(
            f'{key}: unexpected median neighbor distance {median_dist:.1f}px. '
            f'expected ~158px in HE-small coords. check coordinate space.')

    # check first 5 spots in detail
    for i in range(min(5, len(coords))):
        neighbor_dists = dists[i, 1:K_NEIGHBORS+1]
        neighbor_dists = neighbor_dists[neighbor_dists < np.inf]
        if len(neighbor_dists) > 0 and not all(50 < d < 500 for d in neighbor_dists):
            print(f'  WARNING: spot {i} has unexpected neighbor distances: '
                  f'{neighbor_dists.tolist()}')

    return median_dist


def run_virchow2_niche(subarrays):
    """extract niche-level Virchow2 embeddings for all subarrays."""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # skip completed
    done = {p.stem for p in OUT_DIR.glob('*.npy')}
    todo = [s for s in subarrays if s['hd_path'] and s['key'] not in done]
    print(f'Virchow2 niche: {len(done)} done, {len(todo)} remaining, '
          f'{sum(1 for s in subarrays if not s["hd_path"])} no HD')

    if not todo:
        print('Virchow2 niche: all done')
        return

    # compute patch size from first subarray
    spots = load_spots(todo[0]['rdata_path'])
    coords_hd = spots[['pixel_x', 'pixel_y']].values * SCALE_TO_HD
    tree = KDTree(coords_hd)
    dists, _ = tree.query(coords_hd, k=2)
    hd_pixel_dist = np.median(dists[:, 1])
    um_per_pixel_hd = 200.0 / hd_pixel_dist
    patch_size_hd = int(round(128.0 / um_per_pixel_hd))
    print(f'patch size: {patch_size_hd}x{patch_size_hd} HD px -> 224x224')

    # sanity check neighbor distances on first subarray
    sanity_check_neighbors(spots, todo[0]['key'])

    # load Virchow2
    print('loading Virchow2...')
    model = timm.create_model(
        'hf-hub:paige-ai/Virchow2', pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU
    )
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model = model.to(device).eval()
    print(f'  Virchow2 on {device}, 631M params')

    t0 = time.time()
    for i, sub in enumerate(todo):
        key = sub['key']
        spots = load_spots(sub['rdata_path'])
        n_spots = len(spots)

        # extract all patches for this subarray
        patches = extract_patches(sub['hd_path'], spots, patch_size_hd, 224)

        # find k nearest spatial neighbors per spot (in HE-small coords)
        coords = spots[['pixel_x', 'pixel_y']].values
        tree = KDTree(coords)
        k_query = min(K_NEIGHBORS + 1, n_spots)  # +1 because includes self
        _, nn_idx = tree.query(coords, k=k_query)

        # run Virchow2 on all patches, extract CLS tokens (index 0)
        cls_tokens = []
        batch_size = 32
        for j in range(0, len(patches), batch_size):
            batch = torch.stack([transform(p) for p in patches[j:j+batch_size]]).to(device)
            with torch.no_grad():
                out = model(batch)
            cls_tokens.append(out[:, 0, :].cpu().numpy())
        cls_all = np.vstack(cls_tokens)  # (n_spots, 1280)

        # mean pool over neighbors (including self) for niche embedding
        # boundary spots: use however many neighbors are available (option B)
        niche_embs = np.zeros((n_spots, 1280), dtype=np.float32)
        neighbor_counts = np.zeros(n_spots, dtype=np.int32)

        for j in range(n_spots):
            if nn_idx.ndim == 1:
                # edge case: only 1 spot in subarray
                valid = np.array([nn_idx[j]])
            else:
                neighbors = nn_idx[j]
                valid = neighbors[neighbors < n_spots]

            niche_embs[j] = cls_all[valid].mean(axis=0)
            neighbor_counts[j] = len(valid)

        np.save(OUT_DIR / f'{key}.npy', niche_embs)

        # save metadata with neighbor counts
        meta = spots[['pixel_x', 'pixel_y']].copy()
        meta['neighbor_count'] = neighbor_counts
        meta.to_csv(OUT_DIR / f'{key}_meta.tsv', sep='\t')

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 3600
        remaining = (len(todo) - i - 1) / rate * 3600 if rate > 0 else 0
        boundary = (neighbor_counts < K_NEIGHBORS + 1).sum()
        print(f'  [{i+1}/{len(todo)}] {key}: {n_spots} spots -> {niche_embs.shape} '
              f'({boundary} boundary) [{elapsed/60:.0f}m, ~{remaining/60:.0f}m remaining]')

        del patches, cls_tokens, cls_all, niche_embs

    print(f'Virchow2 niche done: {time.time() - t0:.0f}s total')


def check_progress():
    """report what's done."""
    if OUT_DIR.exists():
        files = [f for f in OUT_DIR.glob('*.npy')]
        if files:
            total_spots = sum(np.load(f).shape[0] for f in files)
            # check neighbor counts from meta files
            low_neighbor = 0
            for f in files:
                meta_path = OUT_DIR / f'{f.stem}_meta.tsv'
                if meta_path.exists():
                    meta = pd.read_csv(meta_path, sep='\t', index_col=0)
                    low_neighbor += (meta['neighbor_count'] < 4).sum()
            print(f'virchow2_niche: {len(files)} subarrays, {total_spots} spots, '
                  f'{low_neighbor} with <4 neighbors')
        else:
            print('virchow2_niche: no files yet')
    else:
        print('virchow2_niche: not started')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['check'], default=None)
    args = parser.parse_args()

    if args.step == 'check':
        check_progress()
    else:
        subarrays = discover_subarrays()
        run_virchow2_niche(subarrays)