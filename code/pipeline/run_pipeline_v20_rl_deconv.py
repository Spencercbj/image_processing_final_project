"""
Pipeline v20: Richardson-Lucy deconv experiment for images 04 and 09.
PIL resize 1440 → RL deconv (multiple kernel lengths & iterations) → Restormer → OMDNet → Real-ESRGAN

Freq analysis results:
  04: blur angle ~165°, gradient_aniso=7.85, autocorr_aspect=3.47
  09: blur angle ~8°,   gradient_aniso=3.51, autocorr_aspect=8.46
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
from skimage.restoration import richardson_lucy

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
PHOTOS_DIR = os.path.join(BASE_DIR, 'photos')
PYTHON = sys.executable
ESRGAN_ROOT = os.path.join(BASE_DIR, 'Real-ESRGAN')

sys.path.insert(0, os.path.join(BASE_DIR, 'OMDNet'))
sys.path.insert(0, os.path.join(CODE_DIR, '..'))

from models.deblur_model import OMDNet
from utils.general import move_to_cuda
from utils.record import load_checkpoint, plot_img

RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'pipeline_v20_rl')

IMAGE_CONFIGS = {
    '04': {'angle': 165, 'kernel_lengths': [30, 50, 80], 'iterations': [10, 20, 30]},
    '09': {'angle': 8,   'kernel_lengths': [30, 50, 80], 'iterations': [10, 20, 30]},
}


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def make_motion_kernel(length, angle_deg):
    """Create a 1D motion blur PSF kernel at given angle and length."""
    kernel = np.zeros((length, length), dtype=np.float64)
    center = length // 2
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    for i in range(length):
        t = i - center
        x = int(round(center + t * cos_a))
        y = int(round(center - t * sin_a))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    kernel /= kernel.sum()
    return kernel


def apply_rl_deconv(img_rgb_float, kernel, iterations):
    """Apply Richardson-Lucy deconvolution channel by channel."""
    result = np.zeros_like(img_rgb_float)
    for c in range(3):
        ch = img_rgb_float[:, :, c]
        ch = np.clip(ch, 1e-7, 1.0)
        deconv = richardson_lucy(ch, kernel, num_iter=iterations, clip=True)
        result[:, :, c] = deconv
    return np.clip(result, 0, 1)


def step1_rl_deconv_grid(img_id, config, resize_dir, output_base):
    """Run RL deconv with multiple kernel_length x iterations combos. Returns list of (tag, path)."""
    src = os.path.join(resize_dir, f'{img_id}.png')
    if not os.path.exists(src):
        print(f'  [{img_id}] resize not found at {src}')
        return []

    img = cv2.imread(src)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    results = []
    for klen in config['kernel_lengths']:
        kernel = make_motion_kernel(klen, config['angle'])
        for iters in config['iterations']:
            tag = f'k{klen}_i{iters}'
            out_dir = os.path.join(output_base, f'step1_rl_{tag}')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'{img_id}.png')

            if os.path.exists(out_path):
                print(f'  [{img_id}] RL {tag} skip')
                results.append((tag, out_path))
                continue

            print(f'  [{img_id}] RL deconv angle={config["angle"]}° klen={klen} iters={iters} ...')
            t0 = time.time()
            deconv = apply_rl_deconv(img_rgb, kernel, iters)
            out_bgr = (cv2.cvtColor(deconv.astype(np.float32), cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)
            cv2.imwrite(out_path, out_bgr)
            dt = time.time() - t0
            print(f'    done ({dt:.1f}s)')
            results.append((tag, out_path))

    return results


def step2_restormer(img_id, rl_results, output_base):
    """Run Restormer on each RL result."""
    from Restormer.inference import load_model, tile_inference
    device = 'cuda'
    weights = os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                           'pretrained_models', 'motion_deblurring.pth')
    net = load_model(weights, device)

    out_paths = []
    for tag, rl_path in rl_results:
        out_dir = os.path.join(output_base, f'step2_restormer_{tag}')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{img_id}.png')

        if os.path.exists(out_path):
            print(f'  [{img_id}] Restormer {tag} skip')
            out_paths.append((tag, out_path))
            continue

        print(f'  [{img_id}] Restormer on {tag} ...')
        img = cv2.imread(rl_path)
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
        out_paths.append((tag, out_path))

    del net
    torch.cuda.empty_cache()
    return out_paths


def step3_omdnet(img_id, rest_results, output_base):
    """Run OMDNet on each Restormer result."""
    model = OMDNet(num_res=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    work_dir = os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1')
    load_checkpoint(model, optimizer, None, work_dir)
    torch.set_grad_enabled(False)
    model.eval()

    out_paths = []
    for tag, rest_path in rest_results:
        out_dir = os.path.join(output_base, f'step3_omdnet_{tag}')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{img_id}.png')

        if os.path.exists(out_path):
            print(f'  [{img_id}] OMDNet {tag} skip')
            out_paths.append((tag, out_path))
            continue

        print(f'  [{img_id}] OMDNet on {tag} ...')
        img_blur = Image.open(rest_path)
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
        out_paths.append((tag, out_path))

    del model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    return out_paths


def step4_realesrgan(img_id, omd_results, output_base):
    """Run Real-ESRGAN x4 on each OMDNet result."""
    for tag, omd_path in omd_results:
        out_dir = os.path.join(output_base, f'step4_realesrgan_{tag}')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{img_id}.png')

        if os.path.exists(out_path):
            print(f'  [{img_id}] ESRGAN {tag} skip')
            continue

        tmp_in = os.path.join(out_dir, '_tmp_in')
        os.makedirs(tmp_in, exist_ok=True)
        shutil.copy2(omd_path, os.path.join(tmp_in, f'{img_id}.png'))

        cmd = [
            PYTHON, os.path.join(ESRGAN_ROOT, 'inference_realesrgan.py'),
            '-n', 'RealESRGAN_x4plus',
            '-i', tmp_in, '-o', out_dir,
            '-s', '4', '--tile', '256', '--suffix', '',
        ]
        print(f'  [{img_id}] ESRGAN on {tag} ...')
        subprocess.run(cmd, cwd=ESRGAN_ROOT)
        shutil.rmtree(tmp_in, ignore_errors=True)
        print(f'    Saved')


def main():
    t_start = time.time()
    resize_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v18', 'step1_resize')

    for img_id, config in IMAGE_CONFIGS.items():
        print(f'\n{"#" * 60}')
        print(f'# IMAGE {img_id}  (angle={config["angle"]}°)')
        print(f'{"#" * 60}')

        # Step 1: RL deconv grid search
        rl_results = step1_rl_deconv_grid(img_id, config, resize_dir, RESULTS_DIR)
        if not rl_results:
            print(f'  [{img_id}] No RL results, skipping')
            continue

        # Step 2: Restormer on each RL result
        rest_results = step2_restormer(img_id, rl_results, RESULTS_DIR)

        # Step 3: OMDNet on each Restormer result
        omd_results = step3_omdnet(img_id, rest_results, RESULTS_DIR)

        # Step 4: Real-ESRGAN on each OMDNet result
        step4_realesrgan(img_id, omd_results, RESULTS_DIR)

    dt = time.time() - t_start
    print(f'\n{"=" * 60}')
    print(f'Pipeline v20 done in {dt / 60:.1f} min')
    print(f'Results: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
