"""
FFTformer (CVPR 2023) deblur experiment.

Loads fftformer_arch.py via importlib to avoid basicsr package conflicts.
Default flow: resize(max_dim) -> FFTformer(GoPro) -> optional Real-ESRGAN x4.
"""
import argparse
import glob
import os
import sys
import math
import shutil
import subprocess
import importlib.util
import re

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

OUT = os.path.join(BASE_DIR, 'results', 'pipeline_fftformer')


def get_image_id(path):
    m = re.match(r'^(\d+)', os.path.basename(path))
    return m.group(1) if m else None


def default_input_dir():
    img_dir = os.path.join(BASE_DIR, 'img')
    photos_dir = os.path.join(BASE_DIR, 'photos')
    return img_dir if os.path.isdir(img_dir) else photos_dir


def collect_inputs(input_dir, image_ids):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files = sorted(os.path.normpath(p) for p in set(files))
    if image_ids:
        wanted = set(image_ids)
        files = [p for p in files if get_image_id(p) in wanted]
    return files


def load_fftformer_arch(fft_root):
    arch_path = os.path.join(fft_root, 'basicsr', 'models', 'archs', 'fftformer_arch.py')
    if not os.path.exists(arch_path):
        raise FileNotFoundError(f'Cannot find FFTformer arch at {arch_path}')
    spec = importlib.util.spec_from_file_location('fftformer_arch', arch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fftformer


def load_model(fft_root=FFT_ROOT, device='cuda'):
    fftformer = load_fftformer_arch(fft_root)
    net = fftformer(inp_channels=3, out_channels=3, dim=48,
                    num_blocks=[6, 6, 12], num_refinement_blocks=4,
                    ffn_expansion_factor=3, bias=False)
    ckpt = torch.load(os.path.join(fft_root, 'pretrain_model', 'fftformer_GoPro.pth'),
                      map_location='cpu')
    state = ckpt.get('params', ckpt)
    net.load_state_dict(state, strict=True)
    return net.eval().to(device)


def tile_infer(net, t, tile=256, overlap=32, device='cuda'):
    _, _, h, w = t.shape
    stride = tile - overlap
    out = torch.zeros_like(t)
    cnt = torch.zeros((1, 1, h, w))
    for i in range(max(1, math.ceil((h - overlap) / stride))):
        for j in range(max(1, math.ceil((w - overlap) / stride))):
            top = max(0, min(i * stride, h - tile))
            left = max(0, min(j * stride, w - tile))
            tl = t[:, :, top:top + tile, left:left + tile].to(device)
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


def run_one(net, pil, out_path, tile=256, overlap=32, device='cuda'):
    rgb = np.array(pil).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    h, w = t.shape[2], t.shape[3]
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    out = tile_infer(net, t, tile=tile, overlap=overlap, device=device)[:, :, :h, :w]
    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    cv2.imwrite(out_path, (cv2.cvtColor(out, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=default_input_dir())
    parser.add_argument('--out_dir', type=str, default=OUT)
    parser.add_argument('--fft_root', type=str, default=FFT_ROOT)
    parser.add_argument('--images', type=str, default='04,06,07,09',
                        help='Comma-separated image IDs. Use 08 for the smoke test.')
    parser.add_argument('--max_dim', type=int, default=1440)
    parser.add_argument('--tile', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--skip_esrgan', action='store_true')
    parser.add_argument('--extra_04_720', action='store_true',
                        help='Also run image 04 at max_dim=720 for the kernel-shrink check.')
    args = parser.parse_args()

    image_ids = [x.strip().zfill(2) for x in args.images.split(',') if x.strip()]
    files = collect_inputs(os.path.abspath(args.input_dir), image_ids)
    if not files:
        raise FileNotFoundError(f'No matching inputs in {args.input_dir} for {image_ids}')

    pre = os.path.join(args.out_dir, 'pre_esrgan')
    os.makedirs(pre, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_model(os.path.abspath(args.fft_root), device=device)
    print('FFTformer loaded')
    for src in files:
        iid = get_image_id(src) or os.path.splitext(os.path.basename(src))[0]
        run_one(
            net,
            resize_pil(src, args.max_dim),
            os.path.join(pre, f'{iid}_{args.max_dim}.png'),
            tile=args.tile,
            overlap=args.overlap,
            device=device,
        )
        print(f'  [{iid}] {args.max_dim} done')

    if args.extra_04_720:
        img04 = next((p for p in files if get_image_id(p) == '04'), None)
        if img04:
            run_one(
                net,
                resize_pil(img04, 720),
                os.path.join(pre, '04_720.png'),
                tile=args.tile,
                overlap=args.overlap,
                device=device,
            )
            print('  [04] 720 done')

    del net
    torch.cuda.empty_cache()

    if args.skip_esrgan:
        print(f'Done. Pre-ESRGAN results in {pre}')
        return

    # ESRGAN x4
    esr_in = os.path.join(args.out_dir, '_esr_in')
    os.makedirs(esr_in, exist_ok=True)
    for f in os.listdir(pre):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(pre, f), os.path.join(esr_in, f))
    subprocess.run([PYTHON, os.path.join(ESRGAN_ROOT, 'inference_realesrgan.py'),
                    '-n', 'RealESRGAN_x4plus', '-i', esr_in, '-o', args.out_dir,
                    '-s', '4', '--tile', '400', '--suffix', ''], cwd=ESRGAN_ROOT)
    shutil.rmtree(esr_in, ignore_errors=True)
    print(f'Done. Results in {args.out_dir}')


if __name__ == '__main__':
    main()
