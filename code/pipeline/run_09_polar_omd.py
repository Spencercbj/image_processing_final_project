"""
Image 09 polar coordinate deblur experiment.
PIL resize 1440 → warpPolar → PIL resize 1440 → OMDNet → warpPolar inverse →
NLMeans denoise → Sharpen → Real-ESRGAN x4

Try multiple zoom center candidates.
"""
import os
import sys
import time
import shutil
import subprocess

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import functional as tf

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
PYTHON = sys.executable
ESRGAN_ROOT = os.path.join(BASE_DIR, 'Real-ESRGAN')

sys.path.insert(0, os.path.join(BASE_DIR, 'OMDNet'))
sys.path.insert(0, os.path.join(CODE_DIR, '..'))

from models.deblur_model import OMDNet
from utils.general import move_to_cuda
from utils.record import load_checkpoint, plot_img

RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'pipeline_09_polar')
SRC = os.path.join(BASE_DIR, 'results', 'pipeline_v18', 'step1_resize', '09.png')

# Zoom center candidates (x, y) in 1440x960 image
CENTERS = {
    'center':      (720, 480),
    'right_upper': (960, 360),
    'right_mid':   (900, 480),
}


def run_omdnet(img_pil):
    """Run OMDNet on a PIL image. Returns PIL image."""
    model = OMDNet(num_res=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    load_checkpoint(model, optimizer, None,
                    os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1'))
    torch.set_grad_enabled(False)
    model.eval()

    img_np = np.uint8(np.array(img_pil))
    img_t = tf.to_tensor(img_np).unsqueeze_(0)
    mw = 8
    if img_t.shape[2] % mw != 0:
        img_t = F.pad(img_t, (0, 0, 0, mw - img_t.shape[2] % mw), mode='reflect')
    if img_t.shape[3] % mw != 0:
        img_t = F.pad(img_t, (0, mw - img_t.shape[3] % mw, 0, 0), mode='reflect')

    img_t = move_to_cuda(img_t)
    _, deblured, _, _, _, _, _ = model(img_t, img_t, img_t, train=False)
    result = plot_img(deblured[0][0])

    del model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    return result


def process_center(tag, center, src_img_bgr, out_dir):
    """Full pipeline for one zoom center candidate."""
    os.makedirs(out_dir, exist_ok=True)
    h, w = src_img_bgr.shape[:2]
    cx, cy = center

    # Max radius = distance from center to farthest corner
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    max_radius = max(np.sqrt((cx - x)**2 + (cy - y)**2) for x, y in corners)
    max_radius = int(np.ceil(max_radius))

    # Step 1: warpPolar
    # dsize = (width_for_r, height_for_theta)
    # Use generous angular resolution
    polar_h = int(2 * np.pi * max_radius)  # full circumference
    polar_h = min(polar_h, 2880)  # cap for memory
    polar_w = max_radius
    polar = cv2.warpPolar(src_img_bgr, (polar_w, polar_h),
                          (cx, cy), max_radius,
                          cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    polar_path = os.path.join(out_dir, f'01_polar_{tag}.png')
    cv2.imwrite(polar_path, polar)
    print(f'  [{tag}] Polar: {polar.shape[1]}x{polar.shape[0]} (center=({cx},{cy}), R={max_radius})')

    # Step 2: PIL resize polar image to max_dim=1440
    polar_pil = Image.fromarray(cv2.cvtColor(polar, cv2.COLOR_BGR2RGB))
    pw, ph = polar_pil.size
    sf = 1440 / max(pw, ph)
    if sf < 1:
        new_w, new_h = int(pw * sf), int(ph * sf)
        polar_pil = polar_pil.resize((new_w, new_h), Image.LANCZOS)
        print(f'  [{tag}] Polar resize: {pw}x{ph} -> {new_w}x{new_h}')
    else:
        new_w, new_h = pw, ph
        print(f'  [{tag}] Polar no resize needed: {pw}x{ph}')

    # Step 3: OMDNet in polar space
    print(f'  [{tag}] OMDNet in polar space...')
    omd_result = run_omdnet(polar_pil)
    omd_path = os.path.join(out_dir, f'02_omd_polar_{tag}.png')
    omd_result.save(omd_path)

    # Step 4: Resize back to original polar dimensions
    omd_result = omd_result.resize((polar_w, polar_h), Image.LANCZOS)
    omd_bgr = cv2.cvtColor(np.array(omd_result), cv2.COLOR_RGB2BGR)

    # Step 5: warpPolar inverse → back to Cartesian
    cart = cv2.warpPolar(omd_bgr, (w, h),
                         (cx, cy), max_radius,
                         cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    cart_path = os.path.join(out_dir, f'03_cart_{tag}.png')
    cv2.imwrite(cart_path, cart)
    print(f'  [{tag}] Back to Cartesian: {w}x{h}')

    # Step 6: NLMeans denoise
    denoised = cv2.fastNlMeansDenoisingColored(cart, None, 6, 6, 7, 21)
    dn_path = os.path.join(out_dir, f'04_denoise_{tag}.png')
    cv2.imwrite(dn_path, denoised)

    # Step 7: Unsharp mask sharpen
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    sharp_path = os.path.join(out_dir, f'05_sharp_{tag}.png')
    cv2.imwrite(sharp_path, sharpened)

    # Step 8: Real-ESRGAN x4
    esrgan_out = os.path.join(out_dir, f'esrgan_{tag}')
    os.makedirs(esrgan_out, exist_ok=True)
    tmp_in = os.path.join(esrgan_out, '_tmp')
    os.makedirs(tmp_in, exist_ok=True)
    shutil.copy2(sharp_path, os.path.join(tmp_in, '09.png'))
    cmd = [
        PYTHON, os.path.join(ESRGAN_ROOT, 'inference_realesrgan.py'),
        '-n', 'RealESRGAN_x4plus',
        '-i', tmp_in, '-o', esrgan_out,
        '-s', '4', '--tile', '256', '--suffix', '',
    ]
    print(f'  [{tag}] ESRGAN...')
    subprocess.run(cmd, cwd=ESRGAN_ROOT)
    shutil.rmtree(tmp_in, ignore_errors=True)
    print(f'  [{tag}] Done: {os.path.join(esrgan_out, "09.png")}')


def main():
    t_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    src = cv2.imread(SRC)
    h, w = src.shape[:2]
    print(f'Source: {SRC} ({w}x{h})')

    for tag, center in CENTERS.items():
        print(f'\n{"#" * 60}')
        print(f'# Center: {tag} = {center}')
        print(f'{"#" * 60}')
        process_center(tag, center, src, RESULTS_DIR)

    dt = time.time() - t_start
    print(f'\nAll done in {dt / 60:.1f} min')
    print(f'Results: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
