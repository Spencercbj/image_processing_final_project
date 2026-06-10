"""
Pipeline v9: Optimized for text clarity + detail preservation.

CLAHE → Restormer → resize 1920 → OMDNet → Sharpen (stronger) → Real-ESRGAN x3

Key differences from v8:
- CLAHE instead of HVI-CIDNet: artifact-free low-light enhancement
- Higher OMDNet resolution (1920 vs 1440): preserves more detail
- Stronger sharpen (amount=1.5): better text edge clarity
- Real-ESRGAN x3: output closer to original resolution
- OMDNet with OOM fallback to tiling
"""
import glob
import sys
import os
import argparse
import re
import time

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import functional as tf

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
PHOTOS_DIR = os.path.join(BASE_DIR, 'photos')

sys.path.insert(0, os.path.join(BASE_DIR, 'OMDNet'))
sys.path.insert(0, os.path.join(CODE_DIR, '..'))

from models.deblur_model import OMDNet
from utils.general import move_to_cuda
from utils.record import load_checkpoint, plot_img


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def collect_photos(image_ids):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_files = []
    for ext in exts:
        all_files.extend(glob.glob(os.path.join(PHOTOS_DIR, ext)))
    all_files = sorted([os.path.normpath(p) for p in set(all_files)])
    if image_ids:
        id_set = set(image_ids)
        all_files = [f for f in all_files if get_image_id(f) in id_set]
    return all_files


# ─── Step 1: CLAHE (low-light enhancement, no artifacts) ────
def run_step1_clahe(image_ids, output_dir, clip_limit=3.0, grid_size=8):
    print('\n' + '=' * 60)
    print(f'STEP 1: CLAHE (clip={clip_limit}, grid={grid_size})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

    for fpath in collect_photos(image_ids):
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue

        img = cv2.imread(fpath)
        h, w = img.shape[:2]
        print(f'  [{img_id}] {w}x{h}')

        # Apply CLAHE on L channel in LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        cv2.imwrite(out_path, result)
        print(f'  [{img_id}] Done')

    print('Step 1 done.\n')


# ─── Step 2: Restormer ──────────────────────────────────────
def run_step2_restormer(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 2: Restormer (Motion Deblurring)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from Restormer.inference import load_model, tile_inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                           'pretrained_models', 'motion_deblurring.pth')
    net = load_model(weights, device)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for fpath in files:
        img_id = get_image_id(fpath)
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
    print('Step 2 done.\n')


# ─── Step 3: Resize ─────────────────────────────────────────
def run_step3_resize(image_ids, input_dir, output_dir, target_max_dim=1920):
    print('\n' + '=' * 60)
    print(f'STEP 3: Resize (max_dim={target_max_dim})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for fpath in files:
        img_id = get_image_id(fpath)
        if image_ids and img_id not in set(image_ids):
            continue
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue

        img = cv2.imread(fpath)
        h, w = img.shape[:2]
        current_max = max(w, h)
        if current_max > target_max_dim:
            sf = target_max_dim / current_max
            new_w, new_h = int(w * sf), int(h * sf)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            print(f'  [{img_id}] {w}x{h} -> {new_w}x{new_h}')
        else:
            new_w, new_h = w, h
            print(f'  [{img_id}] {w}x{h} (no resize needed)')
        cv2.imwrite(out_path, img)

    print('Step 3 done.\n')


# ─── Step 4: OMDNet with OOM fallback to tiling ─────────────
def _make_tile_weight(h, w, overlap, device='cpu'):
    weight = torch.ones(1, 1, h, w, device=device)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap, device=device)
        weight[:, :, :overlap, :] *= ramp[None, None, :, None]
        weight[:, :, -overlap:, :] *= ramp.flip(0)[None, None, :, None]
        weight[:, :, :, :overlap] *= ramp[None, None, None, :]
        weight[:, :, :, -overlap:] *= ramp.flip(0)[None, None, None, :]
    return weight


def omd_tile_forward(model, img_tensor, tile_size=512, overlap=128):
    _, C, H, W = img_tensor.shape
    step = tile_size - overlap
    out = torch.zeros(1, C, H, W, device='cpu')
    wsum = torch.zeros(1, 1, H, W, device='cpu')
    tw = _make_tile_weight(tile_size, tile_size, overlap, device='cpu')

    tiles_y = max(1, (H - overlap + step - 1) // step)
    tiles_x = max(1, (W - overlap + step - 1) // step)
    total = tiles_y * tiles_x
    count = 0

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = min(ty * step, H - tile_size)
            x0 = min(tx * step, W - tile_size)
            tile = img_tensor[:, :, y0:y0+tile_size, x0:x0+tile_size]
            tile = move_to_cuda(tile)
            _, deblured, _, _, _, _, _ = model(tile, tile, tile, train=False)
            result = deblured[0].cpu()
            out[:, :, y0:y0+tile_size, x0:x0+tile_size] += result * tw
            wsum[:, :, y0:y0+tile_size, x0:x0+tile_size] += tw
            count += 1
            if count % 10 == 0 or count == total:
                print(f'      tile {count}/{total}')
            torch.cuda.empty_cache()

    return out / wsum.clamp(min=1e-8)


def run_step4_omdnet(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 4: OMDNet (with OOM fallback to tiling)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    model = OMDNet(num_res=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    work_dir = os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1')
    load_checkpoint(model, optimizer, None, work_dir)
    torch.set_grad_enabled(False)
    model.eval()

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for fpath in files:
        img_id = get_image_id(fpath)
        if image_ids and img_id not in set(image_ids):
            continue
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue

        print(f'  [{img_id}] Processing...')
        img_pil = Image.open(fpath)
        w, h = img_pil.size
        print(f'    {w}x{h}')

        img_np = np.uint8(np.array(img_pil))
        img_tensor = tf.to_tensor(img_np).unsqueeze_(0)

        mw = 8
        if img_tensor.shape[2] % mw != 0:
            img_tensor = F.pad(img_tensor, (0, 0, 0, mw - img_tensor.shape[2] % mw), mode='reflect')
        if img_tensor.shape[3] % mw != 0:
            img_tensor = F.pad(img_tensor, (0, mw - img_tensor.shape[3] % mw, 0, 0), mode='reflect')

        # Try whole image first, fall back to tiling on OOM
        try:
            img_cuda = move_to_cuda(img_tensor)
            _, deblured, _, _, _, _, _ = model(img_cuda, img_cuda, img_cuda, train=False)
            plot_img(deblured[0][0]).save(out_path)
            print(f'    Saved (whole image)')
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'    OOM! Falling back to tiling...')
                torch.cuda.empty_cache()
                output = omd_tile_forward(model, img_tensor, tile_size=512, overlap=128)
                plot_img(output[0]).save(out_path)
                print(f'    Saved (tiled)')
            else:
                raise
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    print('Step 4 done.\n')


# ─── Step 5: Sharpen ────────────────────────────────────────
def run_step5_sharpen(image_ids, input_dir, output_dir, amount=1.5, radius=1.0):
    print('\n' + '=' * 60)
    print(f'STEP 5: Sharpen (amount={amount}, radius={radius})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for fpath in files:
        img_id = get_image_id(fpath)
        if image_ids and img_id not in set(image_ids):
            continue
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue

        img = cv2.imread(fpath)
        ksize = int(radius * 4) | 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), radius)
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path, sharpened)
        print(f'  [{img_id}] Sharpened')

    print('Step 5 done.\n')


# ─── Step 6: Real-ESRGAN ────────────────────────────────────
def run_step6_realesrgan(image_ids, input_dir, output_dir, outscale=3):
    print('\n' + '=' * 60)
    print(f'STEP 6: Real-ESRGAN (x{outscale})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    ids_str = ','.join(image_ids) if image_ids else None
    sys.argv = ['inference.py', '-i', input_dir, '-o', output_dir,
                '--outscale', str(outscale)]
    if ids_str:
        sys.argv += ['--images', ids_str]

    from RealESRGAN.inference import main as esrgan_main
    esrgan_main()
    print('Step 6 done.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='01,05,08,11,12,15')
    parser.add_argument('--start_from', type=int, default=1)
    parser.add_argument('--max_dim', type=int, default=1920)
    parser.add_argument('--esrgan_scale', type=int, default=3)
    parser.add_argument('--sharpen_amount', type=float, default=1.5)
    args = parser.parse_args()

    image_ids = [x.strip().zfill(2) for x in args.images.split(',')]

    results_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v9')
    dirs = {
        1: os.path.join(results_dir, 'step1_clahe'),
        2: os.path.join(results_dir, 'step2_restormer'),
        3: os.path.join(results_dir, 'step3_resize_1920'),
        4: os.path.join(results_dir, 'step4_omdnet'),
        5: os.path.join(results_dir, 'step5_sharpen'),
        6: os.path.join(results_dir, 'step6_realesrgan'),
    }

    print(f'Pipeline v9: CLAHE → Restormer → resize {args.max_dim} → OMDNet → Sharpen → Real-ESRGAN x{args.esrgan_scale}')
    print(f'Images: {image_ids}')
    t0 = time.time()

    if args.start_from <= 1:
        run_step1_clahe(image_ids, dirs[1])
    if args.start_from <= 2:
        run_step2_restormer(image_ids, dirs[1], dirs[2])
    if args.start_from <= 3:
        run_step3_resize(image_ids, dirs[2], dirs[3], args.max_dim)
    if args.start_from <= 4:
        run_step4_omdnet(image_ids, dirs[3], dirs[4])
    if args.start_from <= 5:
        run_step5_sharpen(image_ids, dirs[4], dirs[5], amount=args.sharpen_amount)
    if args.start_from <= 6:
        run_step6_realesrgan(image_ids, dirs[5], dirs[6], outscale=args.esrgan_scale)

    elapsed = time.time() - t0
    print(f'\nPipeline v9 done in {elapsed / 60:.1f} min.')
    print(f'Results: {dirs[6]}')


if __name__ == '__main__':
    main()
