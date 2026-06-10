"""
Pipeline v8: HVI-CIDNet → Restormer → resize 1440 → OMD → [SDEdit] → Sharpen → Real-ESRGAN x4

Shared steps (1-4): HVI-CIDNet → Restormer → resize 1440 → OMDNet
Variant A (v8a): + SDEdit (strength 0.25) → Sharpen → Real-ESRGAN x4
Variant B (v8b): + Sharpen → Real-ESRGAN x4
"""
import glob
import sys
import os
import argparse
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


# ─── Step 1: HVI-CIDNet ─────────────────────────────────────
def run_step1_hvi(image_ids, output_dir):
    print('\n' + '=' * 60)
    print('STEP 1: HVI-CIDNet (Low-light Enhancement)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from HVI_CIDNet.inference import load_model, process_image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = os.path.join(BASE_DIR, 'HVI-CIDNet', 'weights', 'LOLv2_syn', 'w_perc.pth')
    net = load_model(weights, device)

    for fpath in collect_photos(image_ids):
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue
        print(f'  [{img_id}] Processing...')
        process_image(net, fpath, out_path, tile_size=512, overlap=128, device=device)
        torch.cuda.empty_cache()

    del net
    torch.cuda.empty_cache()
    print('Step 1 done.\n')


# ─── Step 2: Restormer ──────────────────────────────────────
def run_step2_restormer(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 2: Restormer (Motion Deblurring)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from Restormer.inference import load_model, tile_inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                           'pretrained_models', 'motion_deblurring.pth')
    net = load_model(weights, device)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for fpath in files:
        img_id = get_image_id(fpath)
        if image_ids and img_id not in set(image_ids):
            continue
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
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
    print('Step 2 done.\n')


# ─── Step 3: Resize 1440 ────────────────────────────────────
def run_step3_resize(image_ids, input_dir, output_dir, target_max_dim=1440):
    print('\n' + '=' * 60)
    print(f'STEP 3: Resize (max_dim={target_max_dim})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for fpath in files:
        img_id = get_image_id(fpath)
        if image_ids and img_id not in set(image_ids):
            continue
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue

        img = cv2.imread(fpath)
        h, w = img.shape[:2]
        current_max = max(w, h)
        if current_max > target_max_dim:
            sf = target_max_dim / current_max
            new_w, new_h = int(w * sf), int(h * sf)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            print(f'  [{img_id}] {w}x{h} -> {new_w}x{new_h}')
        else:
            new_w, new_h = w, h
            print(f'  [{img_id}] {w}x{h} (no resize needed)')
        cv2.imwrite(out_path, img)

    print('Step 3 done.\n')


# ─── Step 4: OMDNet (demo.py logic) ─────────────────────────
def run_step4_omdnet(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 4: OMDNet (demo.py logic, whole image)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    model = OMDNet(num_res=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    work_dir = os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1')
    load_checkpoint(model, optimizer, None, work_dir)
    torch.set_grad_enabled(False)
    model.eval()

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for fpath in files:
        img_id = get_image_id(fpath)
        if image_ids and img_id not in set(image_ids):
            continue
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue

        print(f'  [{img_id}] Processing...')
        img_blur = Image.open(fpath)
        print(f'    {img_blur.width}x{img_blur.height}')

        img_blur = np.uint8(np.array(img_blur))
        img_blur = tf.to_tensor(img_blur)
        img_blur.unsqueeze_(0)

        mw = 8
        if img_blur.shape[2] % mw != 0:
            img_blur = F.pad(img_blur, (0, 0, 0, mw - img_blur.shape[2] % mw), mode='reflect')
        if img_blur.shape[3] % mw != 0:
            img_blur = F.pad(img_blur, (0, mw - img_blur.shape[3] % mw, 0, 0), mode='reflect')

        img_blur = move_to_cuda(img_blur)
        _, deblured, _, gates, _, _, _ = model(img_blur, img_blur, img_blur, train=False)

        plot_img(deblured[0][0]).save(out_path)
        print(f'    Saved: {out_path}')
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    print('Step 4 done.\n')


# ─── Step 5a: SDEdit (diffusion) ────────────────────────────
def run_sdedit(image_ids, input_dir, output_dir, strength=0.25):
    print('\n' + '=' * 60)
    print(f'SDEdit (Stable Diffusion img2img, strength={strength})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for fpath in files:
        img_id = get_image_id(fpath)
        if image_ids and img_id not in set(image_ids):
            continue
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue

        print(f'  [{img_id}] Processing...')
        img = Image.open(fpath).convert('RGB')
        w, h = img.size
        # Resize to SD-friendly resolution (multiple of 8, max ~1024)
        sd_max = 1024
        if max(w, h) > sd_max:
            sf = sd_max / max(w, h)
            sd_w = int(w * sf) // 8 * 8
            sd_h = int(h * sf) // 8 * 8
        else:
            sd_w = w // 8 * 8
            sd_h = h // 8 * 8
        img_sd = img.resize((sd_w, sd_h), Image.LANCZOS)
        print(f'    SD input: {sd_w}x{sd_h}')

        result = pipe(
            prompt="high quality sharp detailed photograph, nighttime urban scene",
            negative_prompt="blurry, noise, artifacts, distortion",
            image=img_sd,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=30,
        ).images[0]

        # Resize back to original OMD output size
        result = result.resize((w, h), Image.LANCZOS)
        result.save(out_path)
        print(f'    Saved: {out_path}')
        torch.cuda.empty_cache()

    del pipe
    torch.cuda.empty_cache()
    print('SDEdit done.\n')


# ─── Sharpen ─────────────────────────────────────────────────
def run_sharpen(image_ids, input_dir, output_dir, amount=1.0, radius=1.0):
    print('\n' + '=' * 60)
    print(f'Sharpen (Unsharp Mask, amount={amount}, radius={radius})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for fpath in files:
        img_id = get_image_id(fpath)
        if image_ids and img_id not in set(image_ids):
            continue
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue

        img = cv2.imread(fpath)
        # Unsharp mask: sharpen = original + amount * (original - blur)
        ksize = int(radius * 4) | 1  # ensure odd
        blurred = cv2.GaussianBlur(img, (ksize, ksize), radius)
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path, sharpened)
        print(f'  [{img_id}] Sharpened')

    print('Sharpen done.\n')


# ─── Real-ESRGAN ────────────────────────────────────────────
def run_realesrgan(image_ids, input_dir, output_dir, outscale=4):
    print('\n' + '=' * 60)
    print(f'Real-ESRGAN (x{outscale})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    ids_str = ','.join(image_ids) if image_ids else None
    sys.argv = ['inference.py', '-i', input_dir, '-o', output_dir,
                '--outscale', str(outscale)]
    if ids_str:
        sys.argv += ['--images', ids_str]

    from RealESRGAN.inference import main as esrgan_main
    esrgan_main()
    print('Real-ESRGAN done.\n')


# ─── Main ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='01,05,08,11,12,15')
    parser.add_argument('--variant', type=str, choices=['shared', 'a', 'b', 'all'],
                        default='all',
                        help='shared=steps 1-4 only, a=with SDEdit, b=no SDEdit, all=shared+b+a')
    parser.add_argument('--strength', type=float, default=0.25)
    parser.add_argument('--start_from', type=int, default=1)
    args = parser.parse_args()

    image_ids = [x.strip().zfill(2) for x in args.images.split(',')]

    shared_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v8_shared')
    dirs_shared = {
        1: os.path.join(shared_dir, 'step1_hvi_cidnet'),
        2: os.path.join(shared_dir, 'step2_restormer'),
        3: os.path.join(shared_dir, 'step3_resize_1440'),
        4: os.path.join(shared_dir, 'step4_omdnet'),
    }

    va_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v8a')
    dirs_a = {
        'sdedit': os.path.join(va_dir, 'step5_sdedit'),
        'sharpen': os.path.join(va_dir, 'step6_sharpen'),
        'esrgan': os.path.join(va_dir, 'step7_realesrgan'),
    }

    vb_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v8b')
    dirs_b = {
        'sharpen': os.path.join(vb_dir, 'step5_sharpen'),
        'esrgan': os.path.join(vb_dir, 'step6_realesrgan'),
    }

    t0 = time.time()

    # ─── Shared steps ───
    if args.variant in ('shared', 'all'):
        print('\n' + '#' * 60)
        print('  SHARED STEPS (HVI-CIDNet → Restormer → Resize → OMDNet)')
        print('#' * 60)

        if args.start_from <= 1:
            run_step1_hvi(image_ids, dirs_shared[1])
        if args.start_from <= 2:
            run_step2_restormer(image_ids, dirs_shared[1], dirs_shared[2])
        if args.start_from <= 3:
            run_step3_resize(image_ids, dirs_shared[2], dirs_shared[3])
        if args.start_from <= 4:
            run_step4_omdnet(image_ids, dirs_shared[3], dirs_shared[4])

    # ─── Variant B (no diffusion) ───
    if args.variant in ('b', 'all'):
        print('\n' + '#' * 60)
        print('  VARIANT B: OMD → Sharpen → Real-ESRGAN x4')
        print('#' * 60)
        run_sharpen(image_ids, dirs_shared[4], dirs_b['sharpen'])
        run_realesrgan(image_ids, dirs_b['sharpen'], dirs_b['esrgan'], outscale=4)

    # ─── Variant A (with SDEdit) ───
    if args.variant in ('a', 'all'):
        print('\n' + '#' * 60)
        print(f'  VARIANT A: OMD → SDEdit ({args.strength}) → Sharpen → Real-ESRGAN x4')
        print('#' * 60)
        run_sdedit(image_ids, dirs_shared[4], dirs_a['sdedit'], strength=args.strength)
        run_sharpen(image_ids, dirs_a['sdedit'], dirs_a['sharpen'])
        run_realesrgan(image_ids, dirs_a['sharpen'], dirs_a['esrgan'], outscale=4)

    elapsed = time.time() - t0
    print(f'\n{"=" * 60}')
    print(f'All done in {elapsed / 60:.1f} min.')
    if args.variant in ('b', 'all'):
        print(f'  v8b (no diffusion): {dirs_b["esrgan"]}')
    if args.variant in ('a', 'all'):
        print(f'  v8a (with SDEdit):  {dirs_a["esrgan"]}')
    print('=' * 60)


if __name__ == '__main__':
    main()
