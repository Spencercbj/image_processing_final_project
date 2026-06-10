"""
Pipeline v12: Restormer → resize 1440 → OMDNet → Sharpen(1.5) → Real-ESRGAN x4
No low-light enhancement. Best combination from experiments.
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='01,05,08,11,12,15')
    parser.add_argument('--start_from', type=int, default=1)
    args = parser.parse_args()

    image_ids = [x.strip().zfill(2) for x in args.images.split(',')]
    results_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v12')
    restormer_dir = os.path.join(results_dir, 'step1_restormer')
    resize_dir = os.path.join(results_dir, 'step2_resize_1440')
    omd_dir = os.path.join(results_dir, 'step3_omdnet')
    sharp_dir = os.path.join(results_dir, 'step4_sharpen')
    esrgan_dir = os.path.join(results_dir, 'step5_realesrgan')

    print('Pipeline v12: Restormer → resize 1440 → OMDNet → Sharpen(1.5) → ESRGAN x4')
    print(f'Images: {image_ids}')
    t0 = time.time()

    # Step 1: Restormer on originals
    if args.start_from <= 1:
        print('\n' + '=' * 60)
        print('STEP 1: Restormer (on original photos)')
        print('=' * 60)
        os.makedirs(restormer_dir, exist_ok=True)

        from Restormer.inference import load_model, tile_inference
        device = 'cuda'
        weights = os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                               'pretrained_models', 'motion_deblurring.pth')
        net = load_model(weights, device)

        for fpath in collect_photos(image_ids):
            img_id = get_image_id(fpath)
            out_path = os.path.join(restormer_dir, f'{img_id}.png')
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
        print('Step 1 done.\n')

    # Step 2: Resize to 1440
    if args.start_from <= 2:
        print('=' * 60)
        print('STEP 2: Resize 1440')
        print('=' * 60)
        os.makedirs(resize_dir, exist_ok=True)
        for img_id in image_ids:
            out_path = os.path.join(resize_dir, f'{img_id}.png')
            if os.path.exists(out_path):
                print(f'  [{img_id}] skip')
                continue
            fpath = os.path.join(restormer_dir, f'{img_id}.png')
            img = cv2.imread(fpath)
            h, w = img.shape[:2]
            sf = 1440 / max(w, h)
            if sf < 1:
                nw, nh = int(w * sf), int(h * sf)
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
                print(f'  [{img_id}] {w}x{h} -> {nw}x{nh}')
            else:
                print(f'  [{img_id}] {w}x{h} (no resize)')
            cv2.imwrite(out_path, img)
        print('Step 2 done.\n')

    # Step 3: OMDNet (demo.py logic, whole image)
    if args.start_from <= 3:
        print('=' * 60)
        print('STEP 3: OMDNet (1440, whole image)')
        print('=' * 60)
        os.makedirs(omd_dir, exist_ok=True)
        model = OMDNet(num_res=20)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
        load_checkpoint(model, optimizer, None,
                        os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1'))
        torch.set_grad_enabled(False)
        model.eval()

        for img_id in image_ids:
            out_path = os.path.join(omd_dir, f'{img_id}.png')
            if os.path.exists(out_path):
                print(f'  [{img_id}] skip')
                continue
            fpath = os.path.join(resize_dir, f'{img_id}.png')
            img_pil = Image.open(fpath)
            print(f'  [{img_id}] {img_pil.width}x{img_pil.height}')
            img_np = np.uint8(np.array(img_pil))
            img_t = tf.to_tensor(img_np).unsqueeze_(0)
            mw = 8
            if img_t.shape[2] % mw != 0:
                img_t = F.pad(img_t, (0, 0, 0, mw - img_t.shape[2] % mw), mode='reflect')
            if img_t.shape[3] % mw != 0:
                img_t = F.pad(img_t, (0, mw - img_t.shape[3] % mw, 0, 0), mode='reflect')
            img_c = move_to_cuda(img_t)
            _, deblured, _, _, _, _, _ = model(img_c, img_c, img_c, train=False)
            plot_img(deblured[0][0]).save(out_path)
            print(f'    Saved')
            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)
        print('Step 3 done.\n')

    # Step 4: Sharpen (1.5)
    if args.start_from <= 4:
        print('=' * 60)
        print('STEP 4: Sharpen (amount=1.5)')
        print('=' * 60)
        os.makedirs(sharp_dir, exist_ok=True)
        for img_id in image_ids:
            out_path = os.path.join(sharp_dir, f'{img_id}.png')
            if os.path.exists(out_path):
                print(f'  [{img_id}] skip')
                continue
            img = cv2.imread(os.path.join(omd_dir, f'{img_id}.png'))
            blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
            sharp = cv2.addWeighted(img, 2.5, blurred, -1.5, 0)
            sharp = np.clip(sharp, 0, 255).astype(np.uint8)
            cv2.imwrite(out_path, sharp)
            print(f'  [{img_id}] done')
        print('Step 4 done.\n')

    # Step 5: Real-ESRGAN x4
    if args.start_from <= 5:
        print('=' * 60)
        print('STEP 5: Real-ESRGAN x4')
        print('=' * 60)
        os.makedirs(esrgan_dir, exist_ok=True)
        sys.argv = ['inference.py', '-i', sharp_dir, '-o', esrgan_dir,
                    '--outscale', '4', '--images', ','.join(image_ids)]
        from RealESRGAN.inference import main as esrgan_main
        esrgan_main()
        print('Step 5 done.\n')

    elapsed = time.time() - t0
    print(f'Pipeline v12 done in {elapsed / 60:.1f} min.')
    print(f'Results: {esrgan_dir}')


if __name__ == '__main__':
    main()
