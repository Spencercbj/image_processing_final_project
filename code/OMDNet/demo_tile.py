"""
Pipeline: Restormer → OMDNet (max_dim=1440, demo.py logic) → Real-ESRGAN
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
from PIL import Image
from torchvision.transforms import functional as tf
import cv2

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))

sys.path.insert(0, os.path.join(BASE_DIR, 'OMDNet'))
sys.path.insert(0, os.path.join(CODE_DIR, '..'))
from models.deblur_model import OMDNet
from utils.general import move_to_cuda
from utils.record import load_checkpoint, plot_img


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def collect_photos(image_ids):
    photos_dir = os.path.join(BASE_DIR, 'photos')
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_files = []
    for ext in exts:
        all_files.extend(glob.glob(os.path.join(photos_dir, ext)))
    all_files = sorted([os.path.normpath(p) for p in set(all_files)])
    if image_ids:
        id_set = set(image_ids)
        all_files = [f for f in all_files if get_image_id(f) in id_set]
    return all_files


def run_step1_restormer(input_files, output_dir):
    print('\n' + '=' * 60)
    print('STEP 1: Restormer (Motion Deblurring)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from Restormer.inference import load_model, tile_inference

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                           'pretrained_models', 'motion_deblurring.pth')
    net = load_model(weights, device)

    for fpath in input_files:
        img_id = get_image_id(fpath)
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


def run_step2_omdnet(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 2: OMDNet (demo.py logic, max_dim=1440)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    model = OMDNet(num_res=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    work_dir = os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1')
    training_step = load_checkpoint(model, optimizer, None, work_dir)
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
            print(f'  [{img_id}] Already exists, skipping')
            continue

        print(f'  [{img_id}] Processing...')

        # Load with PIL (like demo.py)
        img_blur = Image.open(path)
        print(f'    Original: {img_blur.width}x{img_blur.height}')

        # Resize to max_dim=1440 (friend's logic)
        target_max_dim = 1440
        w, h = img_blur.size
        current_max_dim = max(w, h)
        if current_max_dim > target_max_dim:
            scale_factor = target_max_dim / current_max_dim
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            img_blur = img_blur.resize((new_w, new_h), Image.LANCZOS)
            print(f'    Resized: {new_w}x{new_h}')

        # to_tensor (like demo.py)
        img_blur = np.uint8(np.array(img_blur))
        img_blur = tf.to_tensor(img_blur)
        img_blur.unsqueeze_(0)

        # Pad to multiple of 8 (like demo.py)
        multiple_width = 8
        if img_blur.shape[2] % multiple_width != 0:
            l_pad = multiple_width - img_blur.shape[2] % multiple_width
            img_blur = F.pad(img_blur, (0, 0, 0, l_pad), mode='reflect')
        if img_blur.shape[3] % multiple_width != 0:
            l_pad = multiple_width - img_blur.shape[3] % multiple_width
            img_blur = F.pad(img_blur, (0, l_pad, 0, 0), mode='reflect')

        print(f'    Tensor: {img_blur.shape}')

        # Forward (whole image, like demo.py)
        img_blur = move_to_cuda(img_blur)
        _, deblured, _, gates, _, _, _ = model(img_blur, img_blur, img_blur, train=False)

        # Save with plot_img (like demo.py)
        plot_img(deblured[0][0]).save(out_path)
        print(f'    Saved: {out_path}')
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    print('Step 2 done.\n')


def run_step3_realesrgan(image_ids, input_dir, output_dir, outscale=4):
    print('\n' + '=' * 60)
    print(f'STEP 3: Real-ESRGAN (x{outscale})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    ids_str = ','.join(image_ids) if image_ids else None
    sys.argv = ['inference.py', '-i', input_dir, '-o', output_dir,
                '--outscale', str(outscale)]
    if ids_str:
        sys.argv += ['--images', ids_str]

    from RealESRGAN.inference import main as esrgan_main
    esrgan_main()
    print('Step 3 done.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default=None)
    parser.add_argument('--esrgan_scale', type=int, default=4)
    parser.add_argument('--start_from', type=int, default=1)
    parser.add_argument('--end_at', type=int, default=3)
    args = parser.parse_args()

    image_ids = None
    if args.images:
        image_ids = [x.strip().zfill(2) for x in args.images.split(',')]

    results_dir = os.path.join(BASE_DIR, 'results', 'restormer_omd_esrgan')
    step1_dir = os.path.join(results_dir, 'step1_restormer')
    step2_dir = os.path.join(results_dir, 'step2_omdnet')
    step3_dir = os.path.join(results_dir, 'step3_realesrgan')

    print(f'Pipeline: Restormer → OMDNet (1440) → Real-ESRGAN x{args.esrgan_scale}')
    print(f'Images: {image_ids or "all"}')
    t0 = time.time()

    if args.start_from <= 1 <= args.end_at:
        input_files = collect_photos(image_ids)
        run_step1_restormer(input_files, step1_dir)

    if args.start_from <= 2 <= args.end_at:
        run_step2_omdnet(image_ids, step1_dir, step2_dir)

    if args.start_from <= 3 <= args.end_at:
        run_step3_realesrgan(image_ids, step2_dir, step3_dir, args.esrgan_scale)

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed / 60:.1f} min. Results: {step3_dir}')


if __name__ == '__main__':
    main()
