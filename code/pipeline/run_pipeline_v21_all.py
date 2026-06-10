"""
Pipeline v21 (all images): Cascade multi-model deblur.
PIL resize 1440 → UFPNet → MPRNet → Restormer → OMDNet → Real-ESRGAN x4
Face images (01,03,10,14,15) get --face_enhance in ESRGAN step.
Skips 04 and 09 (already done).
"""
import glob
import sys
import os
import re
import time
import shutil
import subprocess

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import functional as tf
from collections import OrderedDict
from runpy import run_path
from skimage import img_as_ubyte

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
PYTHON = sys.executable
ESRGAN_ROOT = os.path.join(BASE_DIR, 'Real-ESRGAN')

sys.path.insert(0, os.path.join(BASE_DIR, 'OMDNet'))
sys.path.insert(0, os.path.join(CODE_DIR, '..'))

from models.deblur_model import OMDNet
from utils.general import move_to_cuda
from utils.record import load_checkpoint, plot_img

ALL_IDS = ['01','02','03','05','06','07','08','10','11','12','13','14','15']
FACE_IDS = {'01','03','10','14','15'}
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'pipeline_v21_cascade')


def step1_ufpnet(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 1: UFPNet')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    todo = [iid for iid in image_ids
            if not os.path.exists(os.path.join(output_dir, f'{iid}.png'))]
    if not todo:
        print('  All outputs exist, skip')
        return

    tmp_in = os.path.join(output_dir, '_tmp_in')
    os.makedirs(tmp_in, exist_ok=True)
    for iid in todo:
        src = os.path.join(input_dir, f'{iid}.png')
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp_in, f'{iid}.png'))

    ufp_script = os.path.join(BASE_DIR, 'UFPDeblur', 'run_inference.py')
    cmd = [PYTHON, ufp_script, '--input', tmp_in, '--output', output_dir, '--tile_size', '512']
    print(f'  Running UFPNet on {len(todo)} images...')
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  UFPNet FAILED:\n{result.stderr[-500:]}')
    else:
        print(f'  UFPNet done ({time.time()-t0:.1f}s)')
    shutil.rmtree(tmp_in, ignore_errors=True)


def mprnet_tile_inference(model, input_tensor, tile_size=512, overlap=64):
    """Tiled inference for MPRNet to avoid OOM on large images."""
    _, c, h, w = input_tensor.shape
    stride = tile_size - overlap
    out = torch.zeros_like(input_tensor)
    count = torch.zeros(1, 1, h, w, device='cpu')

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = input_tensor[:, :, y_start:y_end, x_start:x_end].cuda()
            th, tw = tile.shape[2], tile.shape[3]
            padh = (8 - th % 8) % 8
            padw = (8 - tw % 8) % 8
            if padh > 0 or padw > 0:
                tile = F.pad(tile, (0, padw, 0, padh), 'reflect')

            with torch.no_grad():
                restored = model(tile)
            restored = torch.clamp(restored[0], 0, 1)[:, :, :th, :tw].cpu()

            out[:, :, y_start:y_end, x_start:x_end] += restored
            count[:, :, y_start:y_end, x_start:x_end] += 1
            torch.cuda.empty_cache()

    out /= count
    return out


def step2_mprnet(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 2: MPRNet (tiled)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    todo = [iid for iid in image_ids
            if not os.path.exists(os.path.join(output_dir, f'{iid}.png'))]
    if not todo:
        print('  All outputs exist, skip')
        return

    mprnet_dir = os.path.join(BASE_DIR, 'MPRNet')
    sys.path.insert(0, mprnet_dir)
    load_file = run_path(os.path.join(mprnet_dir, 'Deblurring', 'MPRNet.py'))
    model = load_file['MPRNet']()
    model.cuda()

    weights = os.path.join(mprnet_dir, 'Deblurring', 'pretrained_models', 'model_deblurring.pth')
    checkpoint = torch.load(weights, map_location='cuda')
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model.eval()
    print('  MPRNet loaded')

    for iid in todo:
        out_path = os.path.join(output_dir, f'{iid}.png')
        src = os.path.join(input_dir, f'{iid}.png')
        img = Image.open(src).convert('RGB')
        w, h = img.size
        print(f'  [{iid}] Processing ({w}x{h})...')

        input_ = tf.to_tensor(img).unsqueeze(0)  # keep on CPU
        restored = mprnet_tile_inference(model, input_, tile_size=512, overlap=64)
        restored = restored.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        restored = img_as_ubyte(restored)
        cv2.imwrite(out_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
        print(f'    Saved')
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    print('Step 2 done.\n')


def step3_restormer(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 3: Restormer')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    todo = [iid for iid in image_ids
            if not os.path.exists(os.path.join(output_dir, f'{iid}.png'))]
    if not todo:
        print('  All outputs exist, skip')
        return

    from Restormer.inference import load_model, tile_inference
    net = load_model(os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                                  'pretrained_models', 'motion_deblurring.pth'), 'cuda')

    for iid in todo:
        out_path = os.path.join(output_dir, f'{iid}.png')
        src = os.path.join(input_dir, f'{iid}.png')
        print(f'  [{iid}] Processing...')
        img = cv2.imread(src)
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        output = tile_inference(net, img_tensor, tile_size=896, overlap=128, device='cuda')
        output = output[:, :, :h, :w].squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        output_bgr = (cv2.cvtColor(output, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)
        cv2.imwrite(out_path, output_bgr)
        print(f'    Saved')
        torch.cuda.empty_cache()

    del net
    torch.cuda.empty_cache()
    print('Step 3 done.\n')


def step4_omdnet(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 4: OMDNet')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    todo = [iid for iid in image_ids
            if not os.path.exists(os.path.join(output_dir, f'{iid}.png'))]
    if not todo:
        print('  All outputs exist, skip')
        return

    model = OMDNet(num_res=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    load_checkpoint(model, optimizer, None,
                    os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1'))
    torch.set_grad_enabled(False)
    model.eval()

    for iid in todo:
        out_path = os.path.join(output_dir, f'{iid}.png')
        src = os.path.join(input_dir, f'{iid}.png')
        print(f'  [{iid}] Processing...')
        img_blur = Image.open(src)
        img_np = np.uint8(np.array(img_blur))
        img_t = tf.to_tensor(img_np).unsqueeze_(0)
        mw = 8
        if img_t.shape[2] % mw != 0:
            img_t = F.pad(img_t, (0, 0, 0, mw - img_t.shape[2] % mw), mode='reflect')
        if img_t.shape[3] % mw != 0:
            img_t = F.pad(img_t, (0, mw - img_t.shape[3] % mw, 0, 0), mode='reflect')
        img_t = move_to_cuda(img_t)
        _, deblured, _, _, _, _, _ = model(img_t, img_t, img_t, train=False)
        plot_img(deblured[0][0]).save(out_path)
        print(f'    Saved')
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    print('Step 4 done.\n')


def step5_realesrgan(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 5: Real-ESRGAN x4')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    face_ids = [iid for iid in image_ids if iid in FACE_IDS]
    noface_ids = [iid for iid in image_ids if iid not in FACE_IDS]

    def run_batch(ids, face_enhance, label):
        if not ids:
            return
        todo = [iid for iid in ids
                if not os.path.exists(os.path.join(output_dir, f'{iid}.png'))]
        if not todo:
            print(f'  [{label}] All exist, skip')
            return

        tmp_in = os.path.join(output_dir, f'_tmp_{label}')
        os.makedirs(tmp_in, exist_ok=True)
        for iid in todo:
            src = os.path.join(input_dir, f'{iid}.png')
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp_in, f'{iid}.png'))

        cmd = [
            PYTHON, os.path.join(ESRGAN_ROOT, 'inference_realesrgan.py'),
            '-n', 'RealESRGAN_x4plus',
            '-i', tmp_in, '-o', output_dir,
            '-s', '4', '--tile', '256', '--suffix', '',
        ]
        if face_enhance:
            cmd.append('--face_enhance')
        print(f'  [{label}] Running ESRGAN on {len(todo)} images (face_enhance={face_enhance})...')
        subprocess.run(cmd, cwd=ESRGAN_ROOT)
        shutil.rmtree(tmp_in, ignore_errors=True)

    run_batch(face_ids, face_enhance=True, label='face')
    run_batch(noface_ids, face_enhance=False, label='noface')
    print('Step 5 done.\n')


def main():
    t_start = time.time()
    resize_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v18', 'step1_resize')

    print(f'Pipeline v21 (all): cascade deblur for {len(ALL_IDS)} images')
    print(f'resize (reuse v18) -> UFPNet -> MPRNet -> Restormer -> OMDNet -> ESRGAN')
    print(f'Face images: {sorted(FACE_IDS)}')
    print(f'Output: {RESULTS_DIR}\n')

    step1_ufpnet(ALL_IDS, resize_dir,
                 os.path.join(RESULTS_DIR, 'step1_ufpnet'))
    step2_mprnet(ALL_IDS,
                 os.path.join(RESULTS_DIR, 'step1_ufpnet'),
                 os.path.join(RESULTS_DIR, 'step2_mprnet'))
    step3_restormer(ALL_IDS,
                    os.path.join(RESULTS_DIR, 'step2_mprnet'),
                    os.path.join(RESULTS_DIR, 'step3_restormer'))
    step4_omdnet(ALL_IDS,
                 os.path.join(RESULTS_DIR, 'step3_restormer'),
                 os.path.join(RESULTS_DIR, 'step4_omdnet'))
    step5_realesrgan(ALL_IDS,
                     os.path.join(RESULTS_DIR, 'step4_omdnet'),
                     os.path.join(RESULTS_DIR, 'step5_realesrgan'))

    dt = time.time() - t_start
    print(f'\n{"=" * 60}')
    print(f'Pipeline v21 (all) done in {dt / 60:.1f} min')
    print(f'Results: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
