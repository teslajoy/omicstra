"""
scale_embeddings.py - extract UNI2-Reinhard and Novae embeddings for all 281 subarrays.

usage:
    python scripts/scale_embeddings.py                    # run everything
    python scripts/scale_embeddings.py --step uni2        # UNI2 only
    python scripts/scale_embeddings.py --step novae       # Novae only
    python scripts/scale_embeddings.py --step check       # check progress

outputs:
    data/embeddings/uni2_reinhard_all/   - per-subarray .npy files (1536-d)
    data/embeddings/novae_all/           - per-subarray .npy files (64-d)
    data/embeddings/scale_manifest.json  - which subarrays passed, which failed

estimated time: ~11 hours on MPS (MacBook Pro)
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
SEED = 42


def discover_subarrays():
    """find all subarrays and map to HD images."""
    # build (slide, pos) -> tnbc_id from HD filenames
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


def load_counts(rdata_path):
    """load raw counts from selection.RData."""
    r = subprocess.run(['Rscript', '-e', f'''
        load("{rdata_path}")
        write.csv(cnts, stdout())
    '''], capture_output=True, text=True)
    assert r.returncode == 0, f'Rscript failed: {r.stderr[:200]}'
    return pd.read_csv(io.StringIO(r.stdout), index_col=0)


def extract_patches(hd_image_path, spot_df, scale_factor, patch_size_hd, output_size=224):
    """extract patches centered at spot coordinates from HD H&E image."""
    img = Image.open(hd_image_path)
    w, h = img.size
    half = patch_size_hd // 2
    patches = []

    for _, row in spot_df.iterrows():
        cx = int(round(row['pixel_x'] * scale_factor))
        cy = int(round(row['pixel_y'] * scale_factor))
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


def run_uni2(subarrays):
    """extract patches, Reinhard normalize, run UNI2 inference for all subarrays."""
    import timm
    from torchstain.torch.normalizers import TorchReinhardNormalizer
    from skimage import color as skcolor

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    out_dir = EMB_DIR / 'uni2_reinhard_all'
    out_dir.mkdir(parents=True, exist_ok=True)

    # skip already completed
    done = {p.stem for p in out_dir.glob('*.npy') if p.stem != 'all_embeddings' and p.stem != 'all_labels'}
    todo = [s for s in subarrays if s['hd_path'] and s['key'] not in done]
    print(f'UNI2: {len(done)} done, {len(todo)} remaining, {sum(1 for s in subarrays if not s["hd_path"])} no HD')

    if not todo:
        print('UNI2: all done')
        return

    # calibration constants (from pilot notebook)
    HE_SMALL_SIZE = 9523
    HD_SIZE = 31744
    SCALE_TO_HD = HD_SIZE / HE_SMALL_SIZE
    UNI2_INPUT_SIZE = 224

    # compute patch size from first subarray
    spots = load_spots(todo[0]['rdata_path'])
    coords_hd = spots[['pixel_x', 'pixel_y']].values * SCALE_TO_HD
    tree = KDTree(coords_hd)
    dists, _ = tree.query(coords_hd, k=2)
    hd_pixel_dist = np.median(dists[:, 1])
    um_per_pixel_hd = 200.0 / hd_pixel_dist
    patch_size_hd = int(round(128.0 / um_per_pixel_hd))
    print(f'patch size: {patch_size_hd}x{patch_size_hd} HD px -> {UNI2_INPUT_SIZE}x{UNI2_INPUT_SIZE}')

    # load Reinhard normalizer (fit on TNBC83 target from pilot)
    # re-fit on first 100 patches from TNBC83 if available
    print('fitting Reinhard normalizer...')
    target_key = 'TNBC83_CN42_C1'
    target_sub = next((s for s in subarrays if s['key'] == target_key), None)
    if target_sub and target_sub['hd_path']:
        target_spots = load_spots(target_sub['rdata_path'])
        target_patches = extract_patches(target_sub['hd_path'], target_spots,
                                          SCALE_TO_HD, patch_size_hd, UNI2_INPUT_SIZE)
        rng = np.random.RandomState(SEED)
        n_target = min(100, len(target_patches))
        target_idx = rng.choice(len(target_patches), n_target, replace=False)
        cols = int(np.ceil(np.sqrt(n_target)))
        rows = int(np.ceil(n_target / cols))
        composite = Image.new('RGB', (cols * UNI2_INPUT_SIZE, rows * UNI2_INPUT_SIZE))
        for i, pi in enumerate(target_idx):
            r, c = divmod(i, cols)
            composite.paste(target_patches[pi], (c * UNI2_INPUT_SIZE, r * UNI2_INPUT_SIZE))

        normalizer = TorchReinhardNormalizer()
        target_tensor = torch.from_numpy(np.array(composite)).permute(2, 0, 1).contiguous()
        normalizer.fit(target_tensor)
        print(f'  fitted on {n_target} patches from {target_key}')
        del target_patches
    else:
        raise RuntimeError(f'target {target_key} not found')

    # load UNI2
    print('loading UNI2-h...')
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667 * 2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
        'reg_tokens': 8, 'dynamic_img_size': True,
    }
    model = timm.create_model('hf-hub:MahmoodLab/UNI2-h', pretrained=True, **timm_kwargs)
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    uni2_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model = model.to(device).eval()
    print(f'  UNI2-h on {device}')

    def pil_to_chw_uint8(pil_img):
        return torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).contiguous()

    t0 = time.time()
    for i, sub in enumerate(todo):
        key = sub['key']
        spots = load_spots(sub['rdata_path'])

        # extract patches
        patches = extract_patches(sub['hd_path'], spots, SCALE_TO_HD, patch_size_hd, UNI2_INPUT_SIZE)

        # Reinhard normalize
        normed = []
        for p in patches:
            try:
                result = normalizer.normalize(pil_to_chw_uint8(p))
                normed.append(Image.fromarray(result.numpy()))
            except Exception:
                normed.append(p)

        # UNI2 inference
        embeddings = []
        batch_size = 32
        for j in range(0, len(normed), batch_size):
            batch = torch.stack([uni2_transform(p) for p in normed[j:j+batch_size]]).to(device)
            with torch.no_grad():
                emb = model(batch)
            embeddings.append(emb.cpu().numpy())

        emb_array = np.vstack(embeddings)
        np.save(out_dir / f'{key}.npy', emb_array)

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 3600
        remaining = (len(todo) - i - 1) / rate * 3600 if rate > 0 else 0
        print(f'  [{i+1}/{len(todo)}] {key}: {len(patches)} patches -> {emb_array.shape} '
              f'[{elapsed/60:.0f}m, ~{remaining/60:.0f}m remaining]')

        del patches, normed, embeddings

    print(f'UNI2 done: {time.time() - t0:.0f}s total')


def run_novae(subarrays):
    """run Novae inference on all qualifying subarrays."""
    import anndata as ad
    import scanpy as sc
    import novae
    from scipy.sparse import csr_matrix

    out_dir = EMB_DIR / 'novae_all'
    out_dir.mkdir(parents=True, exist_ok=True)

    done = {p.stem for p in out_dir.glob('*.npy') if p.stem != 'all_embeddings' and p.stem != 'all_labels'}

    # load gene mapping
    mapping_file = DATA / 'ensembl_to_symbol.tsv'
    gene_map_df = pd.read_csv(mapping_file, sep='\t')
    ensembl_to_symbol = dict(zip(gene_map_df.iloc[:, 0], gene_map_df.iloc[:, 1]))

    # Novae model
    print('loading Novae...')
    model = novae.Novae.from_pretrained('MICS-Lab/novae-human-0')

    # scale_to_microns from pilot
    novae.settings.scale_to_microns = 200.0 / 164.0

    skipped = []
    t0 = time.time()
    todo = [s for s in subarrays if s['key'] not in done]
    print(f'Novae: {len(done)} done, {len(todo)} remaining')

    for i, sub in enumerate(todo):
        key = sub['key']
        try:
            counts = load_counts(sub['rdata_path'])
            coords = load_spots(sub['rdata_path'])

            # gene mapping
            ensembl_ids = counts.columns.tolist()
            symbols = []
            keep_mask = []
            for eid in ensembl_ids:
                base = eid.split('.')[0]
                if base in ensembl_to_symbol:
                    symbols.append(ensembl_to_symbol[base])
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)

            counts_mapped = counts.loc[:, keep_mask].copy()
            counts_mapped.columns = symbols
            counts_mapped = counts_mapped.T.groupby(level=0).sum().T

            # QC gate: median genes per spot
            genes_per_spot = (counts_mapped > 0).sum(axis=1)
            median_genes = genes_per_spot.median()
            if median_genes < 500:
                skipped.append({'key': key, 'reason': f'median_genes={median_genes:.0f}'})
                print(f'  [{i+1}/{len(todo)}] {key}: SKIPPED (median_genes={median_genes:.0f})')
                continue

            adata = ad.AnnData(
                X=csr_matrix(counts_mapped.values),
                obs=pd.DataFrame(index=counts_mapped.index),
                var=pd.DataFrame(index=counts_mapped.columns),
            )
            adata.obs['subarray'] = key
            adata.obsm['spatial'] = coords[['pixel_x', 'pixel_y']].values

            novae.spatial_neighbors(adata)
            model.compute_representations(adata, zero_shot=True)

            emb = adata.obsm['novae_latent'].copy()
            if hasattr(emb, 'values'):
                emb = emb.values
            emb = np.asarray(emb, dtype=np.float32)

            np.save(out_dir / f'{key}.npy', emb)

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 3600
            remaining = (len(todo) - i - 1) / rate * 3600 if rate > 0 else 0
            print(f'  [{i+1}/{len(todo)}] {key}: {emb.shape[0]} spots -> {emb.shape} '
                  f'({median_genes:.0f} median genes) [{elapsed/60:.0f}m, ~{remaining/60:.0f}m remaining]')

        except Exception as e:
            skipped.append({'key': key, 'reason': str(e)[:100]})
            print(f'  [{i+1}/{len(todo)}] {key}: ERROR {str(e)[:80]}')

    print(f'Novae done: {time.time() - t0:.0f}s, {len(skipped)} skipped')
    return skipped


def check_progress():
    """report what's done."""
    for name in ['uni2_reinhard_all', 'novae_all']:
        d = EMB_DIR / name
        if d.exists():
            files = [f for f in d.glob('*.npy') if f.stem not in ('all_embeddings', 'all_labels')]
            total_spots = sum(np.load(f).shape[0] for f in files)
            print(f'{name}: {len(files)} subarrays, {total_spots} spots')
        else:
            print(f'{name}: not started')

    # manifest
    manifest = EMB_DIR / 'scale_manifest.json'
    if manifest.exists():
        m = json.loads(manifest.read_text())
        print(f'\nmanifest: {m.get("uni2_count", "?")} UNI2, {m.get("novae_count", "?")} Novae')
        if m.get('novae_skipped'):
            print(f'  Novae skipped: {len(m["novae_skipped"])}')


def build_manifest():
    """build manifest of completed embeddings."""
    manifest = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}

    for name, mkey in [('uni2_reinhard_all', 'uni2'), ('novae_all', 'novae')]:
        d = EMB_DIR / name
        if d.exists():
            files = sorted([f.stem for f in d.glob('*.npy')
                           if f.stem not in ('all_embeddings', 'all_labels')])
            manifest[f'{mkey}_count'] = len(files)
            manifest[f'{mkey}_keys'] = files
        else:
            manifest[f'{mkey}_count'] = 0
            manifest[f'{mkey}_keys'] = []

    # matched: subarrays with both UNI2 and Novae
    uni2_set = set(manifest.get('uni2_keys', []))
    novae_set = set(manifest.get('novae_keys', []))
    matched = sorted(uni2_set & novae_set)
    manifest['matched_count'] = len(matched)
    manifest['matched_keys'] = matched

    # patient mapping
    patients = {}
    for key in matched:
        tnbc_id = key.split('_')[0]
        patients.setdefault(tnbc_id, []).append(key)
    manifest['matched_patients'] = len(patients)
    manifest['patients'] = {k: v for k, v in sorted(patients.items())}

    with open(EMB_DIR / 'scale_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nmanifest: {manifest["uni2_count"]} UNI2, {manifest["novae_count"]} Novae, '
          f'{manifest["matched_count"]} matched across {manifest["matched_patients"]} patients')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['uni2', 'novae', 'check', 'manifest'], default=None,
                        help='run specific step (default: all)')
    args = parser.parse_args()

    subarrays = discover_subarrays()

    if args.step == 'check':
        check_progress()
    elif args.step == 'manifest':
        build_manifest()
    elif args.step == 'uni2':
        run_uni2(subarrays)
        build_manifest()
    elif args.step == 'novae':
        novae_skipped = run_novae(subarrays)
        build_manifest()
    else:
        # run everything
        run_uni2(subarrays)
        novae_skipped = run_novae(subarrays)
        build_manifest()