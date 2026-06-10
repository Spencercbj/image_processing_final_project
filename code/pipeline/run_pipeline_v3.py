"""
Pipeline v3: HVI-CIDNet → Wiener deconv (per-image PSF) → Restormer → Real-ESRGAN
No diffusion, no text blending. Per-image strategy from frequency analysis.
"""
import argparse
import os
import sys
import glob
import re
import json
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
PHOTOS_DIR = os.path.join(BASE_DIR, 'photos')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'pipeline_v3')
ANALYSIS_JSON = os.path.join(BASE_DIR, 'Freq_analyze', 'results_v2.json')

sys.path.insert(0, os.path.join(CODE_DIR, '..'))


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def get_image_files(image_ids=None):
    files = sorted(glob.glob(os.path.join(PHOTOS_DIR, '*.*')))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]
    return files


def run_step1_hvi_cidnet(image_ids, output_dir):
    print('\n' + '='*60)
    print('STEP 1: HVI-CIDNet (Low-light Enhancement)')
    print('='*60)
    os.makedirs(output_dir, exist_ok=True)

    from HVI_CIDNet.inference import load_model, process_image
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = os.path.join(BASE_DIR, 'HVI-CIDNet', 'weights', 'LOLv2_syn', 'w_perc.pth')
    net = load_model(weights, device)

    files = get_image_files(image_ids)
    for fpath in files:
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue
        print(f'  [{img_id}] Processing...')
        process_image(net, fpath, out_path, tile_size=512, overlap=128, device=device)
        torch.cuda.empty_cache()

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('Step 1 done.\n')


def run_step2_wiener(image_ids, input_dir, output_dir, method='wiener', K=0.01):
    print('\n' + '='*60)
    print(f'STEP 2: Deconvolution ({method})')
    print('='*60)
    os.makedirs(output_dir, exist_ok=True)

    from wiener.inference import load_analysis, process_single

    analysis = load_analysis()
    debug_dir = os.path.join(output_dir, 'debug')

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for fpath in files:
        img_id = get_image_id(fpath)
        if img_id is None:
            continue
        if image_ids and img_id not in set(image_ids):
            continue
        if img_id not in analysis:
            print(f'  [{img_id}] No analysis — skipping')
            continue

        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue

        process_single(fpath, out_path, analysis[img_id],
                       method=method, K=K, rl_iters=15, debug_dir=debug_dir)

    print('Step 2 done.\n')


def run_step3_restormer(image_ids, input_dir, output_dir):
    print('\n' + '='*60)
    print('STEP 3: Restormer (Motion Deblurring)')
    print('='*60)
    os.makedirs(output_dir, exist_ok=True)

    from Restormer.inference import load_model, tile_inference
    import torch
    import torch.nn.functional as F
    import numpy as np
    import cv2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                           'pretrained_models', 'motion_deblurring.pth')
    net = load_model(weights, device)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for fpath in files:
        img_id = get_image_id(fpath)
        if img_id is None:
            continue
        if image_ids and img_id not in set(image_ids):
            continue

        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue

        print(f'  [{img_id}] Processing...')
        img = cv2.imread(fpath)
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        output = tile_inference(net, img_tensor, tile_size=896, overlap=128, device=device)
        output = output[:, :, :h, :w]
        output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        output_bgr = (cv2.cvtColor(output, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)

        cv2.imwrite(out_path, output_bgr)
        print(f'  [{img_id}] Saved: {out_path}')
        torch.cuda.empty_cache()

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('Step 3 done.\n')


def run_step4_realesrgan(image_ids, input_dir, output_dir):
    print('\n' + '='*60)
    print('STEP 4: Real-ESRGAN (Detail Enhancement)')
    print('='*60)
    os.makedirs(output_dir, exist_ok=True)

    ids_str = ','.join(image_ids) if image_ids else None
    sys.argv = ['inference.py', '-i', input_dir, '-o', output_dir]
    if ids_str:
        sys.argv += ['--images', ids_str]

    from RealESRGAN.inference import main as esrgan_main
    esrgan_main()
    print('Step 4 done.\n')


def write_pipeline_info(results_dir, image_ids, method, K):
    info = f"""# Pipeline v3: Per-Image Deblurring

## Flow
```
原圖 → HVI-CIDNet → Wiener/RL Deconv (per-image PSF) → Restormer → Real-ESRGAN
```

## Steps

### Step 1: HVI-CIDNet (Low-light Enhancement)
- **Model**: CIDNet (CVPR 2025)
- **Weights**: `HVI-CIDNet/weights/LOLv2_syn/w_perc.pth`
- **Settings**: gated2=True, alpha=1.0, tile_size=512, overlap=128, cosine blending

### Step 2: Wiener Deconvolution (Per-Image Motion Blur PSF)
- **Method**: {method}
- **PSF**: Motion blur line kernel, angle + length estimated per-image
- **Direction source**: Freq_analyze/results_v2.json (multi-method consensus)
- **Length estimation**: Autocorrelation FWHM (9-patch median)
- **Wiener K**: {K} (adjusted per severity)
- **Per-image strategy**:
  - sharp images: skip deconv
  - severe blur: K*0.5 (more aggressive)
  - moderate blur: K*0.8
  - mild blur: K as-is
  - no direction confidence: skip deconv

### Step 3: Restormer (Residual Deblurring)
- **Model**: Restormer
- **Weights**: `Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth`
- **Settings**: tile_size=896, overlap=64

### Step 4: Real-ESRGAN (Detail Enhancement)
- **Model**: RealESRGAN_x4plus
- **Settings**: outscale=1, tile=256

## Changes from Pipeline v2
1. **Removed MISCFilter** — replaced with targeted Wiener deconv using estimated PSF
2. **Removed SDEdit** — diffusion hallucination, resolution mismatch
3. **Removed EasyOCR blending** — text blending quality was poor
4. **Added per-image blur analysis** — direction + length from multi-method frequency analysis
5. **Added Wiener deconvolution** — targeted directional deblurring with estimated PSF

## Test Images
{', '.join(image_ids)}
"""
    with open(os.path.join(results_dir, 'pipeline_info.md'), 'w', encoding='utf-8') as f:
        f.write(info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='05,06,08,15',
                        help='Comma-separated image IDs')
    parser.add_argument('--start_from', type=int, default=1)
    parser.add_argument('--end_at', type=int, default=4)
    parser.add_argument('--method', type=str, default='wiener', choices=['wiener', 'rl'])
    parser.add_argument('--K', type=float, default=0.01)
    args = parser.parse_args()

    image_ids = [x.strip().zfill(2) for x in args.images.split(',')]

    step_dirs = {
        1: os.path.join(RESULTS_DIR, 'step1_hvi_cidnet'),
        2: os.path.join(RESULTS_DIR, 'step2_wiener'),
        3: os.path.join(RESULTS_DIR, 'step3_restormer'),
        4: os.path.join(RESULTS_DIR, 'step4_realesrgan'),
    }

    print(f'Pipeline v3 — images: {image_ids}')
    print(f'Steps: {args.start_from} to {args.end_at}')
    print(f'Deconv method: {args.method}, K={args.K}')
    t0 = time.time()

    if args.start_from <= 1 <= args.end_at:
        run_step1_hvi_cidnet(image_ids, step_dirs[1])

    if args.start_from <= 2 <= args.end_at:
        run_step2_wiener(image_ids, step_dirs[1], step_dirs[2],
                         method=args.method, K=args.K)

    if args.start_from <= 3 <= args.end_at:
        run_step3_restormer(image_ids, step_dirs[2], step_dirs[3])

    if args.start_from <= 4 <= args.end_at:
        run_step4_realesrgan(image_ids, step_dirs[3], step_dirs[4])

    write_pipeline_info(RESULTS_DIR, image_ids, args.method, args.K)

    elapsed = time.time() - t0
    print(f'\nPipeline v3 complete in {elapsed/60:.1f} minutes.')


if __name__ == '__main__':
    main()
