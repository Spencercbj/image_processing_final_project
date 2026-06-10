"""
SDEdit inference: img2img diffusion refinement using Stable Diffusion 1.5.
Adds controlled noise to the input and denoises, producing a refined image
that retains structure while enhancing details.

Supports tiled processing for large images (resize to max_edge, run SDEdit,
upscale back).
"""
import argparse
import os
import sys
import glob
import re
import shutil
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
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    return pipe


def process_single(pipe, img_pil, prompt, negative_prompt, strength, steps, guidance, seed):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    w, h = img_pil.size
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    if (new_w, new_h) != (w, h):
        img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)

    result = pipe(
        prompt=prompt,
        image=img_pil,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        negative_prompt=negative_prompt,
        generator=generator,
    ).images[0]

    if (new_w, new_h) != (w, h):
        result = result.resize((w, h), Image.LANCZOS)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--images', type=str, default=None)
    parser.add_argument('--max_edge', type=int, default=768)
    parser.add_argument('--strength', type=float, default=0.35,
                        help='Noise strength (0=no change, 1=full denoise)')
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--prompt', type=str,
                        default='a high quality sharp photograph, clean, detailed, 8k')
    parser.add_argument('--negative_prompt', type=str,
                        default='blurry, noisy, low quality, distorted, artifacts, oversmoothed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sd_path', type=str, default=None)
    args = parser.parse_args()

    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)
    os.makedirs(args.output, exist_ok=True)

    sd_path = args.sd_path or SD15_PATH
    filter_ids = set(args.images.split(',')) if args.images else None

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, '*.*')))
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    else:
        files = [args.input]

    print(f'Loading SD 1.5 pipeline from {sd_path} ...')
    pipe = load_pipeline(sd_path)
    print('Pipeline loaded.')

    for fpath in files:
        img_id = get_image_id(fpath)
        if filter_ids and img_id not in filter_ids:
            continue
        if img_id is None:
            continue

        img = cv2.imread(fpath)
        h, w = img.shape[:2]
        out_name = f'{img_id}.png'

        resized = False
        if max(h, w) > args.max_edge:
            scale = args.max_edge / max(h, w)
            new_w = int(w * scale) // 8 * 8
            new_h = int(h * scale) // 8 * 8
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            print(f'  {img_id}: resized {w}x{h} -> {new_w}x{new_h} for SDEdit')
            resized = True

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        print(f'  {img_id}: running SDEdit (strength={args.strength}, steps={args.num_inference_steps}) ...')
        result_pil = process_single(
            pipe, img_pil, args.prompt, args.negative_prompt,
            args.strength, args.num_inference_steps, args.guidance_scale, args.seed)

        result_np = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

        if resized:
            result_np = cv2.resize(result_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
            print(f'  {img_id}: upscaled back to {w}x{h}')

        out_path = os.path.join(args.output, out_name)
        cv2.imwrite(out_path, result_np)
        print(f'  Saved: {out_path}')

        torch.cuda.empty_cache()
        gc.collect()

    print('SDEdit done.')


if __name__ == '__main__':
    main()
