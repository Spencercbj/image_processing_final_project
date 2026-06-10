"""
Pipeline v7: Restormer → OMDNet → Real-ESRGAN
No HVI-CIDNet, no downscale. Direct on original photos.
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
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'pipeline_v7')

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


def run_step1_restormer(image_ids, input_files, output_dir):
    print('\n' + '=' * 60)
    print('STEP 1: Restormer (Motion Deblurring)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from Restormer.inference import load_model, tile_inference
    import torch
    import torch.nn.functional as F

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                           'pretrained_models', 'motion_deblurring.pth')
    net = load_model(weights, device)

    for fpath in input_files:
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
        print(f'    {w}x{h}')
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
        print(f'  [{img_id}] Saved')
        torch.cuda.empty_cache()

    del net
    torch.cuda.empty_cache()
    print('Step 1 done.\n')


def run_step2_omdnet(image_ids, input_dir, output_dir, tile_size=512, overlap=128):
    print('\n' + '=' * 60)
    print('STEP 2: OMDNet (Local Motion Deblurring)')
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
    torch.cuda.empty_cache()
    print('Step 2 done.\n')


def run_step3_realesrgan(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 3: Real-ESRGAN (Detail Enhancement, x1)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    ids_str = ','.join(image_ids) if image_ids else None
    sys.argv = ['inference.py', '-i', input_dir, '-o', output_dir, '--outscale', '1']
    if ids_str:
        sys.argv += ['--images', ids_str]

    from RealESRGAN.inference import main as esrgan_main
    esrgan_main()
    print('Step 3 done.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default=None)
    parser.add_argument('--start_from', type=int, default=1)
    parser.add_argument('--end_at', type=int, default=3)
    parser.add_argument('--omd_tile', type=int, default=512)
    parser.add_argument('--omd_overlap', type=int, default=128)
    args = parser.parse_args()

    if args.images:
        image_ids = [x.strip().zfill(2) for x in args.images.split(',')]
    else:
        image_ids = [str(i).zfill(2) for i in range(1, 16)]

    step_dirs = {
        1: os.path.join(RESULTS_DIR, 'step1_restormer'),
        2: os.path.join(RESULTS_DIR, 'step2_omdnet'),
        3: os.path.join(RESULTS_DIR, 'step3_realesrgan'),
    }

    print(f'Pipeline v7 — images: {image_ids}')
    print(f'Steps: {args.start_from} to {args.end_at}')
    print('Flow: Original -> Restormer -> OMDNet -> Real-ESRGAN x1')
    t0 = time.time()

    if args.start_from <= 1 <= args.end_at:
        run_step1_restormer(image_ids, get_image_files(image_ids), step_dirs[1])

    if args.start_from <= 2 <= args.end_at:
        run_step2_omdnet(image_ids, step_dirs[1], step_dirs[2],
                         args.omd_tile, args.omd_overlap)

    if args.start_from <= 3 <= args.end_at:
        run_step3_realesrgan(image_ids, step_dirs[2], step_dirs[3])

    elapsed = time.time() - t0
    print(f'\nPipeline v7 complete in {elapsed / 60:.1f} minutes.')


if __name__ == '__main__':
    main()
