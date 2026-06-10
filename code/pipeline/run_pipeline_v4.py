"""
Pipeline v4: HVI-CIDNet → Restormer (cosine blending) → cv2 Denoise → Real-ESRGAN
No Wiener, no diffusion. Solid baseline with grid artifact fixes and denoising.
"""
import argparse
import os
import sys
import glob
import re
import time

import cv2
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
PHOTOS_DIR = os.path.join(BASE_DIR, 'photos')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'pipeline_v4')

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
    print('\n' + '=' * 60)
    print('STEP 1: HVI-CIDNet (Low-light Enhancement)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from HVI_CIDNet.inference import load_model, process_image
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = os.path.join(BASE_DIR, 'HVI-CIDNet', 'weights', 'LOLv2_syn', 'w_perc.pth')
    net = load_model(weights, device)

    for fpath in get_image_files(image_ids):
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue
        print(f'  [{img_id}] Processing...')
        process_image(net, fpath, out_path, tile_size=512, overlap=128, device=device)
        torch.cuda.empty_cache()

    del net
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('Step 1 done.\n')


def run_step2_restormer(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 2: Restormer (Motion Deblurring, cosine blending)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from Restormer.inference import load_model, tile_inference
    import torch
    import torch.nn.functional as F

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
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('Step 2 done.\n')


def run_step3_denoise(image_ids, input_dir, output_dir, h=10, hColor=10,
                      templateWindowSize=7, searchWindowSize=21):
    print('\n' + '=' * 60)
    print('STEP 3: Denoise (cv2.fastNlMeansDenoisingColored)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

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

        print(f'  [{img_id}] Denoising (h={h}, hColor={hColor})...')
        img = cv2.imread(fpath)
        denoised = cv2.fastNlMeansDenoisingColored(
            img, None, h, hColor, templateWindowSize, searchWindowSize)
        cv2.imwrite(out_path, denoised)
        print(f'  [{img_id}] Saved: {out_path}')

    print('Step 3 done.\n')


def run_step4_realesrgan(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 4: Real-ESRGAN (Detail Enhancement)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    ids_str = ','.join(image_ids) if image_ids else None
    sys.argv = ['inference.py', '-i', input_dir, '-o', output_dir]
    if ids_str:
        sys.argv += ['--images', ids_str]

    from RealESRGAN.inference import main as esrgan_main
    esrgan_main()
    print('Step 4 done.\n')


def write_pipeline_info(results_dir, image_ids):
    info = f"""# Pipeline v4: Solid Baseline (No Wiener, No Diffusion)

## Flow
```
原圖 → HVI-CIDNet → Restormer → cv2 Denoise → Real-ESRGAN
```

## Changes from v3
- Removed Wiener deconvolution (uniform deconv hurts sharp areas, minimal help on severe blur)
- Added cv2.fastNlMeansDenoisingColored after Restormer
- Restormer now uses cosine blending (overlap=128) to fix grid artifacts
- Simplified pipeline focusing on reliable components

## Steps

### Step 1: HVI-CIDNet (Low-light Enhancement)
- **Model**: CIDNet (CVPR 2025)
- **Weights**: `HVI-CIDNet/weights/LOLv2_syn/w_perc.pth`
- **Settings**: gated2=True, alpha=1.0, tile_size=512, overlap=128, cosine blending

### Step 2: Restormer (Motion Deblurring)
- **Model**: Restormer
- **Weights**: `Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth`
- **Settings**: tile_size=896, overlap=128, **cosine blending** (fixes grid artifacts)

### Step 3: Denoise
- **Method**: cv2.fastNlMeansDenoisingColored
- **Settings**: h=10, hColor=10, templateWindowSize=7, searchWindowSize=21

### Step 4: Real-ESRGAN (Detail Enhancement)
- **Model**: RealESRGAN_x4plus
- **Settings**: outscale=1 (no upscaling), tile=256

## Test Images
{', '.join(image_ids)}
"""
    with open(os.path.join(results_dir, 'pipeline_info.md'), 'w', encoding='utf-8') as f:
        f.write(info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default=None,
                        help='Comma-separated image IDs (e.g. "01,05,08"). Default: all 15')
    parser.add_argument('--start_from', type=int, default=1)
    parser.add_argument('--end_at', type=int, default=4)
    parser.add_argument('--denoise_h', type=int, default=10)
    parser.add_argument('--denoise_hColor', type=int, default=10)
    args = parser.parse_args()

    if args.images:
        image_ids = [x.strip().zfill(2) for x in args.images.split(',')]
    else:
        image_ids = [str(i).zfill(2) for i in range(1, 16)]

    step_dirs = {
        1: os.path.join(RESULTS_DIR, 'step1_hvi_cidnet'),
        2: os.path.join(RESULTS_DIR, 'step2_restormer'),
        3: os.path.join(RESULTS_DIR, 'step3_denoise'),
        4: os.path.join(RESULTS_DIR, 'step4_realesrgan'),
    }

    print(f'Pipeline v4 — images: {image_ids}')
    print(f'Steps: {args.start_from} to {args.end_at}')
    t0 = time.time()

    if args.start_from <= 1 <= args.end_at:
        run_step1_hvi_cidnet(image_ids, step_dirs[1])

    if args.start_from <= 2 <= args.end_at:
        run_step2_restormer(image_ids, step_dirs[1], step_dirs[2])

    if args.start_from <= 3 <= args.end_at:
        run_step3_denoise(image_ids, step_dirs[2], step_dirs[3],
                          h=args.denoise_h, hColor=args.denoise_hColor)

    if args.start_from <= 4 <= args.end_at:
        run_step4_realesrgan(image_ids, step_dirs[3], step_dirs[4])

    write_pipeline_info(RESULTS_DIR, image_ids)

    elapsed = time.time() - t0
    print(f'\nPipeline v4 complete in {elapsed / 60:.1f} minutes.')


if __name__ == '__main__':
    main()
