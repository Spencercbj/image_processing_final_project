"""
Pipeline v10: HVI-CIDNet → Restormer → resize 1920 → OMDNet → Sharpen(1.5) → Real-ESRGAN x3

Reuses v8's HVI-CIDNet and Restormer results.
Combines v8's HVI-CIDNet with v9's higher resolution (1920) approach.
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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'OMDNet'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.deblur_model import OMDNet
from utils.general import move_to_cuda
from utils.record import load_checkpoint, plot_img


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def _make_tile_weight(h, w, overlap, device='cpu'):
    weight = torch.ones(1, 1, h, w, device=device)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap, device=device)
        weight[:, :, :overlap, :] *= ramp[None, None, :, None]
        weight[:, :, -overlap:, :] *= ramp.flip(0)[None, None, :, None]
        weight[:, :, :, :overlap] *= ramp[None, None, None, :]
        weight[:, :, :, -overlap:] *= ramp.flip(0)[None, None, None, :]
    return weight


def main():
    image_ids = ['01', '05', '08', '11', '12', '15']

    # Reuse v8 shared Restormer results (HVI-CIDNet → Restormer already done)
    v8_restormer = os.path.join(BASE_DIR, 'results', 'pipeline_v8_shared', 'step2_restormer')
    v10_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v10')
    resize_dir = os.path.join(v10_dir, 'step3_resize_1920')
    omd_dir = os.path.join(v10_dir, 'step4_omdnet')
    sharp_dir = os.path.join(v10_dir, 'step5_sharpen')
    esrgan_dir = os.path.join(v10_dir, 'step6_realesrgan')

    print('Pipeline v10: HVI-CIDNet(v8) + Restormer(v8) + resize 1920 + OMDNet + Sharpen + ESRGAN x3')
    print(f'Images: {image_ids}')
    t0 = time.time()

    # Step 3: Resize to 1920
    print('\n' + '=' * 60)
    print('STEP 3: Resize 1920')
    print('=' * 60)
    os.makedirs(resize_dir, exist_ok=True)
    for img_id in image_ids:
        fpath = os.path.join(v8_restormer, f'{img_id}.png')
        out_path = os.path.join(resize_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip')
            continue
        img = cv2.imread(fpath)
        h, w = img.shape[:2]
        sf = 1920 / max(w, h)
        if sf < 1:
            nw, nh = int(w * sf), int(h * sf)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            print(f'  [{img_id}] {w}x{h} -> {nw}x{nh}')
        else:
            print(f'  [{img_id}] {w}x{h} (no resize)')
        cv2.imwrite(out_path, img)
    print('Step 3 done.\n')

    # Step 4: OMDNet
    print('=' * 60)
    print('STEP 4: OMDNet')
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

        try:
            img_c = move_to_cuda(img_t)
            _, deblured, _, _, _, _, _ = model(img_c, img_c, img_c, train=False)
            plot_img(deblured[0][0]).save(out_path)
            print(f'    Saved (whole)')
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'    OOM, tiling...')
                torch.cuda.empty_cache()
                ts, olap = 512, 128
                step = ts - olap
                _, C, H, W = img_t.shape
                out = torch.zeros(1, C, H, W)
                ws = torch.zeros(1, 1, H, W)
                tw = _make_tile_weight(ts, ts, olap)
                ty_n = max(1, (H - olap + step - 1) // step)
                tx_n = max(1, (W - olap + step - 1) // step)
                total = ty_n * tx_n
                count = 0
                for ty in range(ty_n):
                    for tx in range(tx_n):
                        y0 = min(ty * step, H - ts)
                        x0 = min(tx * step, W - ts)
                        tile = move_to_cuda(img_t[:, :, y0:y0+ts, x0:x0+ts])
                        _, d, _, _, _, _, _ = model(tile, tile, tile, train=False)
                        out[:, :, y0:y0+ts, x0:x0+ts] += d[0].cpu() * tw
                        ws[:, :, y0:y0+ts, x0:x0+ts] += tw
                        torch.cuda.empty_cache()
                        count += 1
                        if count % 10 == 0 or count == total:
                            print(f'      tile {count}/{total}')
                out /= ws.clamp(min=1e-8)
                plot_img(out[0]).save(out_path)
                print(f'    Saved (tiled)')
            else:
                raise
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    print('Step 4 done.\n')

    # Step 5: Sharpen (amount=1.5)
    print('=' * 60)
    print('STEP 5: Sharpen (amount=1.5)')
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
    print('Step 5 done.\n')

    # Step 6: Real-ESRGAN x3
    print('=' * 60)
    print('STEP 6: Real-ESRGAN x3')
    print('=' * 60)
    os.makedirs(esrgan_dir, exist_ok=True)
    sys.argv = ['inference.py', '-i', sharp_dir, '-o', esrgan_dir,
                '--outscale', '3', '--images', ','.join(image_ids)]
    from RealESRGAN.inference import main as esrgan_main
    esrgan_main()
    print('Step 6 done.\n')

    elapsed = time.time() - t0
    print(f'Pipeline v10 done in {elapsed / 60:.1f} min.')
    print(f'Results: {esrgan_dir}')


if __name__ == '__main__':
    main()
