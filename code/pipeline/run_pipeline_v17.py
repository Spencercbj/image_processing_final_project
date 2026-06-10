"""
Pipeline v17: PIL resize 1440 → CLAHE → Restormer → OMDNet → Sharpen(0.5) → ESRGAN x4
Early resize — test if processing at lower res throughout gives better results.
"""
import glob
import sys
import os
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


def run_step1_resize(image_ids, output_dir, target_max_dim=1440):
    """PIL LANCZOS resize to target_max_dim."""
    print('\n' + '=' * 60)
    print(f'STEP 1: PIL LANCZOS resize (max_dim={target_max_dim})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    for fpath in collect_photos(image_ids):
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip')
            continue
        img = Image.open(fpath)
        w, h = img.size
        current_max = max(w, h)
        if current_max > target_max_dim:
            sf = target_max_dim / current_max
            new_w = int(w * sf)
            new_h = int(h * sf)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f'  [{img_id}] {w}x{h} → {new_w}x{new_h}')
        else:
            print(f'  [{img_id}] {w}x{h} (no resize)')
        img.save(out_path)

    print('Step 1 done.\n')


def run_step2_clahe(image_ids, input_dir, output_dir, clip_limit=2.0, grid_size=8):
    print('\n' + '=' * 60)
    print(f'STEP 2: CLAHE (clipLimit={clip_limit})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]

    for fpath in files:
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip')
            continue
        img = cv2.imread(fpath)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(out_path, result)
        print(f'  [{img_id}] done')

    print('Step 2 done.\n')


def run_step3_restormer(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 3: Restormer (on 1440 images — fewer tiles, more context)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from Restormer.inference import load_model, tile_inference
    device = 'cuda'
    weights = os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                           'pretrained_models', 'motion_deblurring.pth')
    net = load_model(weights, device)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]

    for fpath in files:
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip')
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
    print('Step 3 done.\n')


def run_step4_omdnet(image_ids, input_dir, output_dir):
    """OMDNet — input already at 1440, no resize needed. Still use PIL load."""
    print('\n' + '=' * 60)
    print('STEP 4: OMDNet (input already 1440)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    model = OMDNet(num_res=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    work_dir = os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1')
    load_checkpoint(model, optimizer, None, work_dir)
    torch.set_grad_enabled(False)
    model.eval()

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]

    for path in files:
        img_id = get_image_id(path)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip')
            continue

        print(f'  [{img_id}] Processing...')
        img_blur = Image.open(path)
        print(f'    {img_blur.width}x{img_blur.height}')

        img_blur = np.uint8(np.array(img_blur))
        img_blur = tf.to_tensor(img_blur)
        img_blur.unsqueeze_(0)

        mw = 8
        if img_blur.shape[2] % mw != 0:
            img_blur = F.pad(img_blur, (0, 0, 0, mw - img_blur.shape[2] % mw), mode='reflect')
        if img_blur.shape[3] % mw != 0:
            img_blur = F.pad(img_blur, (0, mw - img_blur.shape[3] % mw, 0, 0), mode='reflect')

        img_blur = move_to_cuda(img_blur)
        _, deblured, _, gates, _, _, _ = model(img_blur, img_blur, img_blur, train=False)
        plot_img(deblured[0][0]).save(out_path)
        print(f'    Saved')
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    print('Step 4 done.\n')


def run_step5_sharpen(image_ids, input_dir, output_dir, amount=0.5, radius=1.0):
    print('\n' + '=' * 60)
    print(f'STEP 5: Sharpen (amount={amount})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]

    for fpath in files:
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip')
            continue
        img = cv2.imread(fpath)
        ksize = int(radius * 4) | 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), radius)
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path, sharpened)
        print(f'  [{img_id}] done')

    print('Step 5 done.\n')


def run_step6_realesrgan(image_ids, input_dir, output_dir, outscale=4):
    print('\n' + '=' * 60)
    print(f'STEP 6: Real-ESRGAN x{outscale}')
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='02,03,07,11,12,13')
    parser.add_argument('--start_from', type=int, default=1)
    args = parser.parse_args()

    image_ids = [x.strip().zfill(2) for x in args.images.split(',')]
    results_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v17_earlyresize')
    step1_dir = os.path.join(results_dir, 'step1_resize')
    step2_dir = os.path.join(results_dir, 'step2_clahe')
    step3_dir = os.path.join(results_dir, 'step3_restormer')
    step4_dir = os.path.join(results_dir, 'step4_omdnet')
    step5_dir = os.path.join(results_dir, 'step5_sharpen')
    step6_dir = os.path.join(results_dir, 'step6_realesrgan')

    print('Pipeline v17: PIL resize 1440 → CLAHE → Restormer → OMDNet → Sharpen → ESRGAN x4')
    print(f'Images: {image_ids}')
    t0 = time.time()

    if args.start_from <= 1:
        run_step1_resize(image_ids, step1_dir)

    if args.start_from <= 2:
        run_step2_clahe(image_ids, step1_dir, step2_dir)

    if args.start_from <= 3:
        run_step3_restormer(image_ids, step2_dir, step3_dir)

    if args.start_from <= 4:
        run_step4_omdnet(image_ids, step3_dir, step4_dir)

    if args.start_from <= 5:
        run_step5_sharpen(image_ids, step4_dir, step5_dir)

    if args.start_from <= 6:
        run_step6_realesrgan(image_ids, step5_dir, step6_dir)

    elapsed = time.time() - t0
    print(f'\nPipeline v17 done in {elapsed / 60:.1f} min.')
    print(f'Results: {step6_dir}')


if __name__ == '__main__':
    main()
