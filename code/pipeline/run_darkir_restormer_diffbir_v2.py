"""
Improved DarkIR -> Restormer -> DiffBIR pipeline.

This version applies the all-pipelines findings as routing defaults:
- resize to 1440 before neural restoration to reduce VRAM and shorten blur kernels;
- run DarkIR only on selected low-light/glass images instead of forcing it globally;
- insert OMDNet after Restormer by default because it was the strongest handoff;
- skip DiffBIR on text-heavy, zoom/ghosting, or known hallucination-prone images.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
PYTHON = sys.executable

sys.path.insert(0, os.path.join(CODE_DIR, '..'))

DEFAULT_DARKIR_IDS = '13'
DEFAULT_NO_DIFFBIR_IDS = '02,04,05,07,08,09,11,12,13'


def get_image_id(path):
    m = re.match(r'^(\d+)', os.path.basename(path))
    return m.group(1) if m else None


def parse_ids(value):
    if not value:
        return []
    return [x.strip().zfill(2) for x in value.split(',') if x.strip()]


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


def files_for_ids(input_dir, image_ids):
    files = collect_inputs(input_dir, image_ids)
    found = {get_image_id(p) for p in files}
    missing = [iid for iid in image_ids if iid not in found]
    if missing:
        raise FileNotFoundError(f'Missing inputs for image IDs: {missing}')
    return files


def run_step1_resize(input_dir, image_ids, output_dir, max_dim=1440):
    print('\n' + '=' * 60)
    print(f'STEP 1: PIL resize to max_dim={max_dim}')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    for src in files_for_ids(input_dir, image_ids):
        iid = get_image_id(src)
        out_path = os.path.join(output_dir, f'{iid}.png')
        if os.path.exists(out_path):
            print(f'  [{iid}] skip')
            continue

        img = Image.open(src).convert('RGB')
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            print(f'  [{iid}] {w}x{h} -> {img.width}x{img.height}')
        else:
            print(f'  [{iid}] {w}x{h} (no resize)')
        img.save(out_path)


def run_step2_darkir_gate(image_ids, input_dir, output_dir, darkir_ids,
                          tile_size=384, overlap=128):
    print('\n' + '=' * 60)
    print(f'STEP 2: DarkIR gate, DarkIR IDs={sorted(darkir_ids)}')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    darkir_todo = []
    for iid in image_ids:
        src = os.path.join(input_dir, f'{iid}.png')
        dst = os.path.join(output_dir, f'{iid}.png')
        if os.path.exists(dst):
            print(f'  [{iid}] skip')
            continue
        if iid in darkir_ids:
            darkir_todo.append((iid, src, dst))
        else:
            shutil.copy2(src, dst)
            print(f'  [{iid}] copied, DarkIR skipped')

    if not darkir_todo:
        return

    from DarkIR.inference import load_model, process_image

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = os.path.join(BASE_DIR, 'DarkIR', 'models', 'DarkIR_384.pt')
    net = load_model(weights, width=32, device=device)

    for iid, src, dst in darkir_todo:
        print(f'  [{iid}] DarkIR processing')
        process_image(net, src, dst, tile_size=tile_size, overlap=overlap, device=device)
        torch.cuda.empty_cache()

    del net
    torch.cuda.empty_cache()


def run_step3_restormer(image_ids, input_dir, output_dir, tile_size=896, overlap=128):
    print('\n' + '=' * 60)
    print('STEP 3: Restormer motion deblur')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from Restormer.inference import load_model, tile_inference

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = os.path.join(
        BASE_DIR, 'Restormer', 'Motion_Deblurring', 'pretrained_models',
        'motion_deblurring.pth')
    net = load_model(weights, device)

    for iid in image_ids:
        src = os.path.join(input_dir, f'{iid}.png')
        dst = os.path.join(output_dir, f'{iid}.png')
        if os.path.exists(dst):
            print(f'  [{iid}] skip')
            continue

        img = cv2.imread(src)
        if img is None:
            raise FileNotFoundError(src)
        h, w = img.shape[:2]
        print(f'  [{iid}] {w}x{h}')
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode='reflect')
        out = tile_inference(net, t, tile_size=tile_size, overlap=overlap, device=device)
        out = out[:, :, :h, :w].squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        out_bgr = (cv2.cvtColor(out, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)
        cv2.imwrite(dst, out_bgr)
        torch.cuda.empty_cache()

    del net
    torch.cuda.empty_cache()


def run_step4_omdnet_or_copy(image_ids, input_dir, output_dir, skip_omdnet):
    print('\n' + '=' * 60)
    print('STEP 4: OMDNet gate')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    if skip_omdnet:
        for iid in image_ids:
            src = os.path.join(input_dir, f'{iid}.png')
            dst = os.path.join(output_dir, f'{iid}.png')
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        print('  OMDNet skipped; copied Restormer output.')
        return

    import run_pipeline_v18 as v18
    v18.run_step3_omdnet(image_ids, input_dir, output_dir)


def sharpen_image(src, dst, amount=0.5, radius=1.0):
    img = cv2.imread(src)
    if img is None:
        raise FileNotFoundError(src)
    ksize = int(radius * 4) | 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), radius)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    cv2.imwrite(dst, np.clip(sharpened, 0, 255).astype(np.uint8))


def run_step5_sharpen_pre(image_ids, input_dir, output_dir, no_diffbir_ids):
    print('\n' + '=' * 60)
    print('STEP 5: pre-DiffBIR sharpen or copy')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    for iid in image_ids:
        src = os.path.join(input_dir, f'{iid}.png')
        dst = os.path.join(output_dir, f'{iid}.png')
        if os.path.exists(dst):
            print(f'  [{iid}] skip')
            continue
        if iid in no_diffbir_ids:
            shutil.copy2(src, dst)
            print(f'  [{iid}] copied, DiffBIR skipped')
        else:
            sharpen_image(src, dst, amount=0.3)
            print(f'  [{iid}] sharpened before DiffBIR')


def run_step6_diffbir_gate(image_ids, input_dir, output_dir, no_diffbir_ids, strength):
    print('\n' + '=' * 60)
    print(f'STEP 6: DiffBIR gate, strength={strength}')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    todo = []
    for iid in image_ids:
        src = os.path.join(input_dir, f'{iid}.png')
        dst = os.path.join(output_dir, f'{iid}.png')
        if os.path.exists(dst):
            print(f'  [{iid}] skip')
        elif iid in no_diffbir_ids:
            shutil.copy2(src, dst)
            print(f'  [{iid}] copied, no DiffBIR')
        else:
            todo.append((iid, src))

    if not todo:
        return

    diffbir_root = os.path.join(BASE_DIR, 'DiffBIR')
    tmp_in = tempfile.mkdtemp(prefix='darkir_diffbir_in_')
    tmp_out = tempfile.mkdtemp(prefix='darkir_diffbir_out_')
    for iid, src in todo:
        shutil.copy2(src, os.path.join(tmp_in, f'{iid}.png'))

    pos_prompt = (
        'Sharp nighttime street photography, crisp details, high resolution, '
        'well-defined edges, photo-realistic.'
    )
    neg_prompt = (
        'motion blur, blurry, smeared, hallucinated text, wrong letters, '
        'painting, illustration, 3d render, worst quality, low quality, '
        'watermark, jpeg artifacts, over-smooth.'
    )
    cmd = [
        PYTHON, os.path.join(diffbir_root, 'inference.py'),
        '--task', 'sr',
        '--version', 'v2.1',
        '--upscale', '1',
        '--input', tmp_in,
        '--output', tmp_out,
        '--captioner', 'none',
        '--precision', 'fp16',
        '--start_point_type', 'cond',
        '--strength', str(strength),
        '--noise_aug', '0',
        '--pos_prompt', pos_prompt,
        '--neg_prompt', neg_prompt,
        '--cleaner_tiled',
        '--cleaner_tile_size', '512',
        '--cleaner_tile_stride', '256',
        '--vae_encoder_tiled',
        '--vae_encoder_tile_size', '256',
        '--vae_decoder_tiled',
        '--vae_decoder_tile_size', '256',
        '--cldm_tiled',
        '--cldm_tile_size', '512',
        '--cldm_tile_stride', '256',
        '--steps', '5',
        '--cfg_scale', '4.0',
        '--seed', '231',
    ]
    result = subprocess.run(cmd, cwd=diffbir_root)
    shutil.rmtree(tmp_in, ignore_errors=True)
    if result.returncode != 0:
        shutil.rmtree(tmp_out, ignore_errors=True)
        raise RuntimeError(f'DiffBIR failed with exit code {result.returncode}')

    for iid, _src in todo:
        produced = os.path.join(tmp_out, f'{iid}.png')
        if not os.path.exists(produced):
            raise FileNotFoundError(f'DiffBIR did not produce {produced}')
        shutil.move(produced, os.path.join(output_dir, f'{iid}.png'))
        print(f'  [{iid}] DiffBIR saved')
    shutil.rmtree(tmp_out, ignore_errors=True)


def run_step7_sharpen_post(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 7: post sharpen')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    for iid in image_ids:
        src = os.path.join(input_dir, f'{iid}.png')
        dst = os.path.join(output_dir, f'{iid}.png')
        if os.path.exists(dst):
            print(f'  [{iid}] skip')
            continue
        sharpen_image(src, dst, amount=0.5)
        print(f'  [{iid}] done')


def run_step8_realesrgan(image_ids, input_dir, output_dir, outscale=4):
    print('\n' + '=' * 60)
    print(f'STEP 8: Real-ESRGAN x{outscale}')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    ids_arg = ','.join(image_ids)
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            'inference.py',
            '-i', input_dir,
            '-o', output_dir,
            '--outscale', str(outscale),
            '--images', ids_arg,
        ]
        from RealESRGAN.inference import main as esrgan_main
        esrgan_main()
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=default_input_dir())
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(BASE_DIR, 'results', 'pipeline_darkir_restormer_diffbir_v2'))
    parser.add_argument('--images', type=str,
                        default='01,02,03,04,05,06,07,08,09,10,11,12,13,14,15')
    parser.add_argument('--darkir_ids', type=str, default=DEFAULT_DARKIR_IDS,
                        help='IDs that should actually pass through DarkIR.')
    parser.add_argument('--no_diffbir', type=str, default=DEFAULT_NO_DIFFBIR_IDS,
                        help='IDs copied around DiffBIR to protect text and failure cases.')
    parser.add_argument('--start_from', type=int, default=1)
    parser.add_argument('--max_dim', type=int, default=1440)
    parser.add_argument('--diffbir_strength', type=float, default=0.8)
    parser.add_argument('--skip_omdnet', action='store_true')
    parser.add_argument('--skip_esrgan', action='store_true')
    args = parser.parse_args()

    image_ids = parse_ids(args.images)
    darkir_ids = set(parse_ids(args.darkir_ids))
    no_diffbir_ids = set(parse_ids(args.no_diffbir))

    s1 = os.path.join(args.out_dir, 'step1_resize')
    s2 = os.path.join(args.out_dir, 'step2_darkir_gate')
    s3 = os.path.join(args.out_dir, 'step3_restormer')
    s4 = os.path.join(args.out_dir, 'step4_omdnet')
    s5 = os.path.join(args.out_dir, 'step5_sharpen_pre')
    s6 = os.path.join(args.out_dir, 'step6_diffbir_gate')
    s7 = os.path.join(args.out_dir, 'step7_sharpen_post')
    s8 = os.path.join(args.out_dir, 'step8_realesrgan')

    print('Improved DarkIR/Restormer/DiffBIR pipeline')
    print(f'Input: {os.path.abspath(args.input_dir)}')
    print(f'Images: {image_ids}')
    print(f'DarkIR IDs: {sorted(darkir_ids)}')
    print(f'No DiffBIR IDs: {sorted(no_diffbir_ids)}')
    t0 = time.time()

    if args.start_from <= 1:
        run_step1_resize(args.input_dir, image_ids, s1, args.max_dim)
    if args.start_from <= 2:
        run_step2_darkir_gate(image_ids, s1, s2, darkir_ids)
    if args.start_from <= 3:
        run_step3_restormer(image_ids, s2, s3)
    if args.start_from <= 4:
        run_step4_omdnet_or_copy(image_ids, s3, s4, args.skip_omdnet)
    if args.start_from <= 5:
        run_step5_sharpen_pre(image_ids, s4, s5, no_diffbir_ids)
    if args.start_from <= 6:
        run_step6_diffbir_gate(image_ids, s5, s6, no_diffbir_ids, args.diffbir_strength)
    if args.start_from <= 7:
        run_step7_sharpen_post(image_ids, s6, s7)
    if args.start_from <= 8 and not args.skip_esrgan:
        run_step8_realesrgan(image_ids, s7, s8)

    print(f'\nDone in {(time.time() - t0) / 60:.1f} min.')
    print(f'Output root: {args.out_dir}')


if __name__ == '__main__':
    main()
