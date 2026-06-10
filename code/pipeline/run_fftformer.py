"""
FFTformer (CVPR 2023, frequency-domain transformer) deblur — the last untried
model in ideas.md. Theoretically strong on long motion blur. OOM-risky on 8GB
(FFT attention), so small tiles + optional low-res.

Loads fftformer_arch.py via importlib to avoid the basicsr package conflict.
resize(maxdim) -> FFTformer(GoPro) -> ESRGAN x4.
"""
import os
import sys
import math
import shutil
import subprocess
import importlib.util

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
PYTHON = sys.executable
ESRGAN_ROOT = os.path.join(BASE_DIR, 'Real-ESRGAN')
FFT_ROOT = os.path.join(BASE_DIR, 'FFTformer')

_spec = importlib.util.spec_from_file_location(
    'fftformer_arch',
    os.path.join(FFT_ROOT, 'basicsr', 'models', 'archs', 'fftformer_arch.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fftformer = _mod.fftformer

PHOTOS = {
    '04': '04_Cyclist_Passing_Warm_Storefront_Lights.jpg',
    '06': '06_Shaking_Neon_Signs_Overhead_Night.jpg',
    '07': '07_Yellow_Taxi_Neon_Rain_Street.jpg',
    '09': '09_White_Truck_Zoom_Blur_Rain.jpg',
}
OUT = os.path.join(BASE_DIR, 'results', 'pipeline_fftformer')


def load_model():
    net = fftformer(inp_channels=3, out_channels=3, dim=48,
                    num_blocks=[6, 6, 12], num_refinement_blocks=4,
                    ffn_expansion_factor=3, bias=False)
    ckpt = torch.load(os.path.join(FFT_ROOT, 'pretrain_model', 'fftformer_GoPro.pth'),
                      map_location='cpu')
    state = ckpt.get('params', ckpt)
    net.load_state_dict(state, strict=True)
    return net.eval().cuda()


def tile_infer(net, t, tile=256, overlap=32):
    _, _, h, w = t.shape
    stride = tile - overlap
    out = torch.zeros_like(t)
    cnt = torch.zeros((1, 1, h, w))
    for i in range(max(1, math.ceil((h - overlap) / stride))):
        for j in range(max(1, math.ceil((w - overlap) / stride))):
            top = max(0, min(i * stride, h - tile))
            left = max(0, min(j * stride, w - tile))
            tl = t[:, :, top:top + tile, left:left + tile].cuda()
            with torch.no_grad():
                pr = net(tl).cpu()
            out[:, :, top:top + tile, left:left + tile] += pr
            cnt[:, :, top:top + tile, left:left + tile] += 1
            torch.cuda.empty_cache()
    return out / cnt


def resize_pil(path, maxdim):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    sf = maxdim / max(w, h)
    return img.resize((int(w * sf), int(h * sf)), Image.LANCZOS)


def run_one(net, pil, out_path):
    rgb = np.array(pil).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    h, w = t.shape[2], t.shape[3]
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    out = tile_infer(net, t)[:, :, :h, :w]
    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    cv2.imwrite(out_path, (cv2.cvtColor(out, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8))


def main():
    pre = os.path.join(OUT, 'pre_esrgan')
    os.makedirs(pre, exist_ok=True)
    net = load_model()
    print('FFTformer loaded')
    for iid, fn in PHOTOS.items():
        src = os.path.join(BASE_DIR, 'photos', fn)
        # 1440 for all
        run_one(net, resize_pil(src, 1440), os.path.join(pre, f'{iid}_1440.png'))
        print(f'  [{iid}] 1440 done')
    # 04 also at low-res 720 (kernel-shrink hypothesis)
    run_one(net, resize_pil(os.path.join(BASE_DIR, 'photos', PHOTOS['04']), 720),
            os.path.join(pre, '04_720.png'))
    print('  [04] 720 done')
    del net
    torch.cuda.empty_cache()

    # ESRGAN x4
    esr_in = os.path.join(OUT, '_esr_in')
    os.makedirs(esr_in, exist_ok=True)
    for f in os.listdir(pre):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(pre, f), os.path.join(esr_in, f))
    subprocess.run([PYTHON, os.path.join(ESRGAN_ROOT, 'inference_realesrgan.py'),
                    '-n', 'RealESRGAN_x4plus', '-i', esr_in, '-o', OUT,
                    '-s', '4', '--tile', '400', '--suffix', ''], cwd=ESRGAN_ROOT)
    shutil.rmtree(esr_in, ignore_errors=True)
    print(f'Done. Results in {OUT}')


if __name__ == '__main__':
    main()
