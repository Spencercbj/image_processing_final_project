"""
Pipeline v6: HVI-CIDNet → Restormer → Downscale → OMDNet → Real-ESRGAN (x2)

Restormer handles global camera motion blur first,
then OMDNet refines local object motion blur at reduced resolution.
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
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'pipeline_v6')

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
    print('STEP 2: Restormer (Global Motion Deblurring)')
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


def run_step3_downscale(image_ids, input_dir, output_dir, scale=2):
    print('\n' + '=' * 60)
    print(f'STEP 3: Downscale (1/{scale}x)')
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

        img = cv2.imread(fpath)
        h, w = img.shape[:2]
        new_h = (h // scale // 8) * 8
        new_w = (w // scale // 8) * 8
        img_down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(out_path, img_down)
        print(f'  [{img_id}] {w}x{h} -> {new_w}x{new_h}')

    print('Step 3 done.\n')


def run_step4_omdnet(image_ids, input_dir, output_dir, tile_size=768, overlap=192):
    print('\n' + '=' * 60)
    print('STEP 4: OMDNet (Local Motion Deblurring)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from OMDNet.inference import load_model, process_image
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1', 'model_ckpt.ckpt')
    net = load_model(weights, num_res=20, device=device)

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
        process_image(net, fpath, out_path, tile_size, overlap, device)
        torch.cuda.empty_cache()

    del net
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('Step 4 done.\n')


def run_step5_realesrgan(image_ids, input_dir, output_dir, outscale=2):
    print('\n' + '=' * 60)
    print(f'STEP 5: Real-ESRGAN (Detail Enhancement, x{outscale})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    ids_str = ','.join(image_ids) if image_ids else None
    sys.argv = ['inference.py', '-i', input_dir, '-o', output_dir,
                '--outscale', str(outscale)]
    if ids_str:
        sys.argv += ['--images', ids_str]

    from RealESRGAN.inference import main as esrgan_main
    esrgan_main()
    print('Step 5 done.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default=None)
    parser.add_argument('--start_from', type=int, default=1)
    parser.add_argument('--end_at', type=int, default=5)
    parser.add_argument('--downscale', type=int, default=2)
    parser.add_argument('--tile_size', type=int, default=768)
    parser.add_argument('--overlap', type=int, default=192)
    args = parser.parse_args()

    if args.images:
        image_ids = [x.strip().zfill(2) for x in args.images.split(',')]
    else:
        image_ids = [str(i).zfill(2) for i in range(1, 16)]

    step_dirs = {
        1: os.path.join(RESULTS_DIR, 'step1_hvi_cidnet'),
        2: os.path.join(RESULTS_DIR, 'step2_restormer'),
        3: os.path.join(RESULTS_DIR, 'step3_downscale'),
        4: os.path.join(RESULTS_DIR, 'step4_omdnet'),
        5: os.path.join(RESULTS_DIR, 'step5_realesrgan'),
    }

    print(f'Pipeline v6 — images: {image_ids}')
    print(f'Steps: {args.start_from} to {args.end_at}')
    print(f'Flow: HVI-CIDNet -> Restormer -> Downscale 1/{args.downscale} -> OMDNet -> Real-ESRGAN x{args.downscale}')
    t0 = time.time()

    if args.start_from <= 1 <= args.end_at:
        run_step1_hvi_cidnet(image_ids, step_dirs[1])

    if args.start_from <= 2 <= args.end_at:
        run_step2_restormer(image_ids, step_dirs[1], step_dirs[2])

    if args.start_from <= 3 <= args.end_at:
        run_step3_downscale(image_ids, step_dirs[2], step_dirs[3], args.downscale)

    if args.start_from <= 4 <= args.end_at:
        run_step4_omdnet(image_ids, step_dirs[3], step_dirs[4],
                         args.tile_size, args.overlap)

    if args.start_from <= 5 <= args.end_at:
        run_step5_realesrgan(image_ids, step_dirs[4], step_dirs[5],
                             outscale=args.downscale)

    elapsed = time.time() - t0
    print(f'\nPipeline v6 complete in {elapsed / 60:.1f} minutes.')


if __name__ == '__main__':
    main()
