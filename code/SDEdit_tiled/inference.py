"""
Tile-based SDEdit: process full-resolution images in overlapping tiles
through Stable Diffusion img2img, with cosine blending to avoid seams.

Solves the v2 SDEdit problem: instead of shrinking 6K→768px (10x upscale
amplifies artifacts), each tile is processed at native SD resolution.
"""
import argparse
import os
import sys
import glob
import re
import gc
import math

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler

SD15_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'PASD', 'checkpoints', 'stable-diffusion-v1-5'))


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def load_pipeline(sd_path, device='cuda'):
    scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder='scheduler')
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        sd_path,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def _make_tile_weight_np(tile_h, tile_w, overlap):
    """2D cosine ramp blending weight (numpy, HWC-compatible)."""
    def _ramp(length, ramp_len):
        w = np.ones(length, dtype=np.float32)
        if ramp_len > 0:
            ramp = np.linspace(0, 1, ramp_len + 2)[1:-1].astype(np.float32)
            w[:ramp_len] = ramp
            w[-ramp_len:] = ramp[::-1]
        return w
    wy = _ramp(tile_h, overlap)
    wx = _ramp(tile_w, overlap)
    return wy[:, None] * wx[None, :]


def process_tiled(pipe, img_bgr, tile_size=768, overlap=128,
                  prompt='', negative_prompt='',
                  strength=0.20, steps=25, guidance=7.5, seed=42):
    """Process image in overlapping tiles through SDEdit."""
    h, w = img_bgr.shape[:2]
    stride = tile_size - overlap

    num_h = max(1, math.ceil((h - overlap) / stride))
    num_w = max(1, math.ceil((w - overlap) / stride))

    output = np.zeros((h, w, 3), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)
    tile_weight = _make_tile_weight_np(tile_size, tile_size, overlap)

    total = num_h * num_w
    count = 0

    for i in range(num_h):
        for j in range(num_w):
            top = min(i * stride, h - tile_size)
            left = min(j * stride, w - tile_size)
            top = max(top, 0)
            left = max(left, 0)
            th = min(tile_size, h - top)
            tw = min(tile_size, w - left)

            tile_bgr = img_bgr[top:top + th, left:left + tw].copy()

            pad_h = (8 - th % 8) % 8
            pad_w = (8 - tw % 8) % 8
            if pad_h > 0 or pad_w > 0:
                tile_bgr = cv2.copyMakeBorder(
                    tile_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            tile_pil = Image.fromarray(cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB))

            generator = torch.Generator(device='cuda').manual_seed(seed + i * num_w + j)
            result_pil = pipe(
                prompt=prompt,
                image=tile_pil,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]

            result_np = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
            result_np = result_np[:th, :tw].astype(np.float64)

            tw_crop = tile_weight[:th, :tw]
            for c in range(3):
                output[top:top + th, left:left + tw, c] += result_np[:, :, c] * tw_crop
            weight[top:top + th, left:left + tw] += tw_crop

            count += 1
            if count % 5 == 0 or count == total:
                print(f'    tile {count}/{total}')

            torch.cuda.empty_cache()

    weight = np.maximum(weight, 1e-8)
    for c in range(3):
        output[:, :, c] /= weight

    return np.clip(output, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--images', type=str, default=None)
    parser.add_argument('--tile_size', type=int, default=768)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--strength', type=float, default=0.20)
    parser.add_argument('--num_inference_steps', type=int, default=25)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--prompt', type=str,
                        default='a high quality sharp photograph, clean, detailed')
    parser.add_argument('--negative_prompt', type=str,
                        default='blurry, noisy, low quality, distorted, artifacts')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sd_path', type=str, default=None)
    args = parser.parse_args()

    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)
    os.makedirs(args.output, exist_ok=True)

    sd_path = args.sd_path or SD15_PATH
    filter_ids = None
    if args.images:
        filter_ids = set(x.strip().zfill(2) for x in args.images.split(','))

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, '*.*')))
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    else:
        files = [args.input]

    print(f'Loading SD 1.5 from {sd_path} ...')
    pipe = load_pipeline(sd_path)
    print('Pipeline loaded.')
    print(f'Settings: tile={args.tile_size}, overlap={args.overlap}, '
          f'strength={args.strength}, steps={args.num_inference_steps}')

    for fpath in files:
        img_id = get_image_id(fpath)
        if img_id is None:
            continue
        if filter_ids and img_id not in filter_ids:
            continue

        out_path = os.path.join(args.output, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue

        img = cv2.imread(fpath)
        h, w = img.shape[:2]
        stride = args.tile_size - args.overlap
        n_tiles = max(1, math.ceil((h - args.overlap) / stride)) * \
                  max(1, math.ceil((w - args.overlap) / stride))
        print(f'  [{img_id}] {w}x{h}, ~{n_tiles} tiles')

        result = process_tiled(
            pipe, img,
            tile_size=args.tile_size,
            overlap=args.overlap,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            strength=args.strength,
            steps=args.num_inference_steps,
            guidance=args.guidance_scale,
            seed=args.seed,
        )

        cv2.imwrite(out_path, result)
        print(f'  [{img_id}] Saved: {out_path}')

        gc.collect()
        torch.cuda.empty_cache()

    print('Tiled SDEdit done.')


if __name__ == '__main__':
    main()
