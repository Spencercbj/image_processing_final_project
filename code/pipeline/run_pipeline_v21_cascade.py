"""
Pipeline v21: Cascade multi-model deblur for images 04 and 09.
PIL resize 1440 → UFPNet → MPRNet → Restormer → OMDNet → Real-ESRGAN x4
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

IMAGE_IDS = ['04', '09']
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'pipeline_v21_cascade')


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def step1_ufpnet(image_ids, input_dir, output_dir):
    """Run UFPNet via subprocess (separate basicsr namespace)."""
    print('\n' + '=' * 60)
    print('STEP 1: UFPNet')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    # Check if all outputs exist
    all_exist = all(
        os.path.exists(os.path.join(output_dir, f'{iid}.png'))
        for iid in image_ids
    )
    if all_exist:
        print('  All outputs exist, skip')
        return

    # Prepare temp input dir with only our images
    tmp_in = os.path.join(output_dir, '_tmp_in')
    os.makedirs(tmp_in, exist_ok=True)
    for iid in image_ids:
        src = os.path.join(input_dir, f'{iid}.png')
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp_in, f'{iid}.png'))

    ufp_script = os.path.join(BASE_DIR, 'UFPDeblur', 'run_inference.py')
    cmd = [
        PYTHON, ufp_script,
        '--input', tmp_in,
        '--output', output_dir,
        '--tile_size', '512',
    ]
    print(f'  Running UFPNet on {len(image_ids)} images...')
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  UFPNet FAILED:\n{result.stderr}')
    else:
        print(f'  UFPNet done ({time.time()-t0:.1f}s)')
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f'    {line}')

    shutil.rmtree(tmp_in, ignore_errors=True)


def step2_mprnet(image_ids, input_dir, output_dir):
    """Run MPRNet for deblurring."""
    print('\n' + '=' * 60)
    print('STEP 2: MPRNet')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    all_exist = all(
        os.path.exists(os.path.join(output_dir, f'{iid}.png'))
        for iid in image_ids
    )
    if all_exist:
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

    img_multiple_of = 8

    for iid in image_ids:
        out_path = os.path.join(output_dir, f'{iid}.png')
        if os.path.exists(out_path):
            print(f'  [{iid}] skip')
            continue

        src = os.path.join(input_dir, f'{iid}.png')
        img = Image.open(src).convert('RGB')
        w, h = img.size
        print(f'  [{iid}] Processing ({w}x{h})...')

        input_ = tf.to_tensor(img).unsqueeze(0).cuda()
        H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
        W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        with torch.no_grad():
            restored = model(input_)
        restored = restored[0]
        restored = torch.clamp(restored, 0, 1)
        restored = restored[:, :, :h, :w]
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        cv2.imwrite(out_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
        print(f'    Saved')
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    print('Step 2 done.\n')


def step3_restormer(image_ids, input_dir, output_dir):
    """Run Restormer motion deblurring."""
    print('\n' + '=' * 60)
    print('STEP 3: Restormer')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from Restormer.inference import load_model, tile_inference
    device = 'cuda'
    weights = os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                           'pretrained_models', 'motion_deblurring.pth')
    net = load_model(weights, device)

    for iid in image_ids:
        out_path = os.path.join(output_dir, f'{iid}.png')
        if os.path.exists(out_path):
            print(f'  [{iid}] skip')
            continue

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
        output = tile_inference(net, img_tensor, tile_size=896, overlap=128, device=device)
        output = output[:, :, :h, :w]
        output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        output_bgr = (cv2.cvtColor(output, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)
        cv2.imwrite(out_path, output_bgr)
        print(f'    Saved')
        torch.cuda.empty_cache()

    del net
    torch.cuda.empty_cache()
    print('Step 3 done.\n')


def step4_omdnet(image_ids, input_dir, output_dir):
    """Run OMDNet local motion deblur."""
    print('\n' + '=' * 60)
    print('STEP 4: OMDNet')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    model = OMDNet(num_res=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    work_dir = os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1')
    load_checkpoint(model, optimizer, None, work_dir)
    torch.set_grad_enabled(False)
    model.eval()

    for iid in image_ids:
        out_path = os.path.join(output_dir, f'{iid}.png')
        if os.path.exists(out_path):
            print(f'  [{iid}] skip')
            continue

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
        _, deblured, _, gates, _, _, _ = model(img_t, img_t, img_t, train=False)
        plot_img(deblured[0][0]).save(out_path)
        print(f'    Saved')
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    print('Step 4 done.\n')


def step5_realesrgan(image_ids, input_dir, output_dir):
    """Run Real-ESRGAN x4 upscale."""
    print('\n' + '=' * 60)
    print('STEP 5: Real-ESRGAN x4')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    tmp_in = os.path.join(output_dir, '_tmp_in')
    os.makedirs(tmp_in, exist_ok=True)

    for iid in image_ids:
        src = os.path.join(input_dir, f'{iid}.png')
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp_in, f'{iid}.png'))

    cmd = [
        PYTHON, os.path.join(ESRGAN_ROOT, 'inference_realesrgan.py'),
        '-n', 'RealESRGAN_x4plus',
        '-i', tmp_in, '-o', output_dir,
        '-s', '4', '--tile', '256', '--suffix', '',
    ]
    print(f'  Running ESRGAN...')
    subprocess.run(cmd, cwd=ESRGAN_ROOT)
    shutil.rmtree(tmp_in, ignore_errors=True)
    print('Step 5 done.\n')


def main():
    t_start = time.time()
    resize_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v18', 'step1_resize')

    print(f'Pipeline v21: cascade deblur for images {IMAGE_IDS}')
    print(f'resize (reuse v18) → UFPNet → MPRNet → Restormer → OMDNet → ESRGAN')
    print(f'Output: {RESULTS_DIR}\n')

    step1_ufpnet(IMAGE_IDS, resize_dir,
                 os.path.join(RESULTS_DIR, 'step1_ufpnet'))

    step2_mprnet(IMAGE_IDS,
                 os.path.join(RESULTS_DIR, 'step1_ufpnet'),
                 os.path.join(RESULTS_DIR, 'step2_mprnet'))

    step3_restormer(IMAGE_IDS,
                    os.path.join(RESULTS_DIR, 'step2_mprnet'),
                    os.path.join(RESULTS_DIR, 'step3_restormer'))

    step4_omdnet(IMAGE_IDS,
                 os.path.join(RESULTS_DIR, 'step3_restormer'),
                 os.path.join(RESULTS_DIR, 'step4_omdnet'))

    step5_realesrgan(IMAGE_IDS,
                     os.path.join(RESULTS_DIR, 'step4_omdnet'),
                     os.path.join(RESULTS_DIR, 'step5_realesrgan'))

    dt = time.time() - t_start
    print(f'\n{"=" * 60}')
    print(f'Pipeline v21 done in {dt / 60:.1f} min')
    print(f'Results: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
