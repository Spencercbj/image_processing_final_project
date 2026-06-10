"""
Pipeline v19 (face): resize 1440 → Restormer → OMDNet → DiffBIR SDEdit (s=0.85) → GFPGAN face enhance
Face images only: 01, 03, 10, 14, 15
Improved negative prompt to reduce ghosting/artifacts around faces.
"""
import glob
import sys
import os
import re
import time
import shutil
import subprocess
import tempfile

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import functional as tf

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
PHOTOS_DIR = os.path.join(BASE_DIR, 'photos')
PYTHON = sys.executable
ESRGAN_ROOT = os.path.join(BASE_DIR, 'Real-ESRGAN')
DIFFBIR_ROOT = os.path.join(BASE_DIR, 'DiffBIR')

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


def run_step1_resize(image_ids, output_dir, target_max_dim=1440):
    print('\n' + '=' * 60)
    print(f'STEP 1: PIL LANCZOS resize (max_dim={target_max_dim})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    # Reuse from v18 if available
    v18_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v18', 'step1_resize')
    reused = 0
    for img_id in image_ids:
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            continue
        v18_path = os.path.join(v18_dir, f'{img_id}.png')
        if os.path.exists(v18_path):
            shutil.copy2(v18_path, out_path)
            reused += 1
            print(f'  [{img_id}] reused from v18')

    if reused == len(image_ids):
        print('  All reused.')
        print('Step 1 done.\n')
        return

    for fpath in collect_photos(image_ids):
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            continue
        img = Image.open(fpath)
        w, h = img.size
        current_max = max(w, h)
        if current_max > target_max_dim:
            sf = target_max_dim / current_max
            new_w = int(w * sf)
            new_h = int(h * sf)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f'  [{img_id}] {w}x{h} -> {new_w}x{new_h}')
        else:
            print(f'  [{img_id}] {w}x{h} (no resize)')
        img.save(out_path)

    print('Step 1 done.\n')


def run_step2_restormer(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 2: Restormer')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    # Reuse from v18
    v18_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v18', 'step2_restormer')
    for img_id in image_ids:
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            continue
        v18_path = os.path.join(v18_dir, f'{img_id}.png')
        if os.path.exists(v18_path):
            shutil.copy2(v18_path, out_path)
            print(f'  [{img_id}] reused from v18')

    remaining = [img_id for img_id in image_ids
                 if not os.path.exists(os.path.join(output_dir, f'{img_id}.png'))]
    if not remaining:
        print('  All reused.')
        print('Step 2 done.\n')
        return

    from Restormer.inference import load_model, tile_inference
    device = 'cuda'
    weights = os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                           'pretrained_models', 'motion_deblurring.pth')
    net = load_model(weights, device)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    remaining_set = set(remaining)
    files = [f for f in files if get_image_id(f) in remaining_set]

    for fpath in files:
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        print(f'  [{img_id}] Processing...')
        img = cv2.imread(fpath)
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
        print(f'  [{img_id}] Saved')
        torch.cuda.empty_cache()

    del net
    torch.cuda.empty_cache()
    print('Step 2 done.\n')


def run_step3_omdnet(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 3: OMDNet (input already 1440)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    # Reuse from v18
    v18_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v18', 'step3_omdnet')
    for img_id in image_ids:
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            continue
        v18_path = os.path.join(v18_dir, f'{img_id}.png')
        if os.path.exists(v18_path):
            shutil.copy2(v18_path, out_path)
            print(f'  [{img_id}] reused from v18')

    remaining = [img_id for img_id in image_ids
                 if not os.path.exists(os.path.join(output_dir, f'{img_id}.png'))]
    if not remaining:
        print('  All reused.')
        print('Step 3 done.\n')
        return

    model = OMDNet(num_res=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    work_dir = os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1')
    load_checkpoint(model, optimizer, None, work_dir)
    torch.set_grad_enabled(False)
    model.eval()

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    remaining_set = set(remaining)
    files = [f for f in files if get_image_id(f) in remaining_set]

    for path in files:
        img_id = get_image_id(path)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        print(f'  [{img_id}] Processing...')
        img_blur = Image.open(path)
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
        print(f'    Saved')
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    print('Step 3 done.\n')


def run_step4_diffbir(image_ids, input_dir, output_dir, strength=0.85):
    """DiffBIR SDEdit with improved negative prompt to reduce ghosting."""
    print('\n' + '=' * 60)
    print(f'STEP 4: DiffBIR SDEdit (strength={strength})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    tmpdir = tempfile.mkdtemp(prefix='diffbir_face_in_')
    tmp_out = tempfile.mkdtemp(prefix='diffbir_face_out_')

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]

    todo = []
    for fpath in files:
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip')
            continue
        shutil.copy2(fpath, os.path.join(tmpdir, f'{img_id}.png'))
        todo.append(img_id)

    if not todo:
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)
        print('  All done already.')
        print('Step 4 done.\n')
        return

    neg_prompt = (
        "painting, oil painting, illustration, drawing, art, sketch, cartoon, "
        "CG Style, 3D render, unreal engine, blurring, dirty, messy, "
        "worst quality, low quality, frames, watermark, signature, jpeg artifacts, "
        "deformed, lowres, over-smooth, "
        "ghost, ghosting, motion trail, double exposure, transparent overlay, "
        "duplicate face, blurred edges, halo artifact, chromatic aberration"
    )

    pos_prompt = (
        "sharp photograph, high resolution, highly detailed, taken with Canon EOS R, "
        "hyper detailed photo-realistic, 32k, Color Grading, ultra HD, "
        "clear face, sharp facial features, well-defined edges, "
        "skin pore detailing, hyper sharpness, perfect without deformations, no ghosting"
    )

    cmd = [
        PYTHON, os.path.join(DIFFBIR_ROOT, 'inference.py'),
        '--task', 'sr',
        '--version', 'v2.1',
        '--upscale', '1',
        '--start_point_type', 'cond',
        '--strength', str(strength),
        '--noise_aug', '0',
        '--steps', '5',
        '--cfg_scale', '4.0',
        '--cleaner_tiled',
        '--cleaner_tile_size', '512',
        '--vae_encoder_tiled',
        '--vae_encoder_tile_size', '512',
        '--vae_decoder_tiled',
        '--vae_decoder_tile_size', '512',
        '--cldm_tiled',
        '--cldm_tile_size', '512',
        '--captioner', 'none',
        '--neg_prompt', neg_prompt,
        '--pos_prompt', pos_prompt,
        '--input', tmpdir,
        '--output', tmp_out,
    ]

    print(f'  Running DiffBIR on {todo} ...')
    print(f'  neg_prompt: {neg_prompt[:80]}...')
    result = subprocess.run(cmd, cwd=DIFFBIR_ROOT)

    if result.returncode != 0:
        print('  WARNING: DiffBIR had errors')
    else:
        print('  DiffBIR done.')

    # Collect outputs
    for img_id in todo:
        candidates = [
            os.path.join(tmp_out, f'{img_id}.png'),
            os.path.join(tmp_out, 'samples', f'{img_id}.png'),
        ]
        # Also check subdirectories
        for root, dirs, fnames in os.walk(tmp_out):
            for fn in fnames:
                if fn.startswith(img_id):
                    candidates.append(os.path.join(root, fn))

        found = False
        for c in candidates:
            if os.path.exists(c):
                shutil.copy2(c, os.path.join(output_dir, f'{img_id}.png'))
                print(f'  [{img_id}] saved from {os.path.relpath(c, tmp_out)}')
                found = True
                break
        if not found:
            print(f'  [{img_id}] WARNING: output not found!')

    shutil.rmtree(tmpdir, ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)
    print('Step 4 done.\n')


def run_step5_face_enhance(image_ids, input_dir, output_dir, outscale=4):
    """ESRGAN x4 with GFPGAN face enhancement."""
    print('\n' + '=' * 60)
    print(f'STEP 5: Real-ESRGAN x{outscale} + GFPGAN face enhance')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]

    tmp_in = os.path.join(output_dir, '_tmp_esrgan_in')
    os.makedirs(tmp_in, exist_ok=True)
    todo = []
    for fpath in files:
        img_id = get_image_id(fpath)
        dst = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(dst):
            print(f'  [{img_id}] skip')
            continue
        shutil.copy2(fpath, os.path.join(tmp_in, f'{img_id}.png'))
        todo.append(img_id)

    if not todo:
        shutil.rmtree(tmp_in, ignore_errors=True)
        print('  All done already.')
        print('Step 5 done.\n')
        return

    cmd = [
        PYTHON, os.path.join(ESRGAN_ROOT, 'inference_realesrgan.py'),
        '-n', 'RealESRGAN_x4plus',
        '-i', tmp_in,
        '-o', output_dir,
        '-s', str(outscale),
        '--tile', '256',
        '--suffix', '',
        '--face_enhance',
    ]

    print(f'  Running ESRGAN + face_enhance on {todo} ...')
    result = subprocess.run(cmd, cwd=ESRGAN_ROOT)
    shutil.rmtree(tmp_in, ignore_errors=True)

    if result.returncode != 0:
        print('  WARNING: ESRGAN had errors')
    else:
        print('  Done.')

    print('Step 5 done.\n')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='01,03,10,14,15')
    parser.add_argument('--start_from', type=int, default=1)
    parser.add_argument('--strength', type=float, default=0.85)
    args = parser.parse_args()

    image_ids = [x.strip().zfill(2) for x in args.images.split(',')]
    results_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v19_face')
    step1_dir = os.path.join(results_dir, 'step1_resize')
    step2_dir = os.path.join(results_dir, 'step2_restormer')
    step3_dir = os.path.join(results_dir, 'step3_omdnet')
    step4_dir = os.path.join(results_dir, 'step4_diffbir')
    step5_dir = os.path.join(results_dir, 'step5_realesrgan_face')

    print(f'Pipeline v19 (face): resize -> Restormer -> OMDNet -> DiffBIR(s={args.strength}) -> ESRGAN+GFPGAN')
    print(f'Images: {image_ids}')
    t0 = time.time()

    if args.start_from <= 1:
        run_step1_resize(image_ids, step1_dir)

    if args.start_from <= 2:
        run_step2_restormer(image_ids, step1_dir, step2_dir)

    if args.start_from <= 3:
        run_step3_omdnet(image_ids, step2_dir, step3_dir)

    if args.start_from <= 4:
        run_step4_diffbir(image_ids, step3_dir, step4_dir, strength=args.strength)

    if args.start_from <= 5:
        run_step5_face_enhance(image_ids, step4_dir, step5_dir)

    elapsed = time.time() - t0
    print(f'\nPipeline v19 done in {elapsed / 60:.1f} min.')
    print(f'Results: {step5_dir}')


if __name__ == '__main__':
    main()
