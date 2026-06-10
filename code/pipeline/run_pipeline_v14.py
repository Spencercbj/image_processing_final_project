"""
Pipeline v14: CLAHE → Restormer → OMDNet (PIL 1440) → [Sharpen(0.3) → DiffBIR(0.85)] → Sharpen(0.5) → ESRGAN x4
Text-heavy images skip DiffBIR (and pre-DiffBIR sharpen).
"""
import glob
import sys
import os
import re
import time
import subprocess
import shutil
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


# ─── Step 1: CLAHE ─────────────────────────────────────────
def run_step1_clahe(image_ids, output_dir, clip_limit=2.0, grid_size=8):
    print('\n' + '=' * 60)
    print(f'STEP 1: CLAHE (clipLimit={clip_limit}, grid={grid_size}x{grid_size})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

    for fpath in collect_photos(image_ids):
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip')
            continue
        print(f'  [{img_id}] Processing...')
        img = cv2.imread(fpath)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(out_path, result)
        print(f'  [{img_id}] Saved')

    print('Step 1 done.\n')


# ─── Step 2: Restormer ─────────────────────────────────────
def run_step2_restormer(image_ids, input_dir, output_dir):
    print('\n' + '=' * 60)
    print('STEP 2: Restormer (Motion Deblurring)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from Restormer.inference import load_model, tile_inference
    device = 'cuda'
    weights = os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                           'pretrained_models', 'motion_deblurring.pth')
    net = load_model(weights, device)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]

    for fpath in files:
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip')
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


# ─── Step 3: OMDNet (PIL resize 1440) ──────────────────────
def run_step3_omdnet(image_ids, input_dir, output_dir, target_max_dim=1440):
    print('\n' + '=' * 60)
    print(f'STEP 3: OMDNet (PIL resize {target_max_dim}, whole image)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    model = OMDNet(num_res=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    work_dir = os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1')
    load_checkpoint(model, optimizer, None, work_dir)
    torch.set_grad_enabled(False)
    model.eval()

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]

    for path in files:
        img_id = get_image_id(path)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip')
            continue

        print(f'  [{img_id}] Processing...')
        img_blur = Image.open(path)
        w, h = img_blur.size
        print(f'    Original: {w}x{h}')

        current_max_dim = max(w, h)
        if current_max_dim > target_max_dim:
            scale_factor = target_max_dim / current_max_dim
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            img_blur = img_blur.resize((new_w, new_h), Image.LANCZOS)
            print(f'    PIL resize: {new_w}x{new_h}')

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


# ─── Step 4: Sharpen (pre-DiffBIR, only for non-text images) ──
def run_step4_sharpen_pre(image_ids, input_dir, output_dir, no_diffbir_ids, amount=0.3, radius=1.0):
    """For DiffBIR images: sharpen(0.3). For text images: just copy (they skip this + DiffBIR)."""
    print('\n' + '=' * 60)
    print(f'STEP 4: Pre-DiffBIR Sharpen (amount={amount}) + copy text images')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]

    for fpath in files:
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip')
            continue

        if img_id in no_diffbir_ids:
            # Text image: copy directly (will go straight to step 6 sharpen)
            shutil.copy2(fpath, out_path)
            print(f'  [{img_id}] copied (text image, skip DiffBIR)')
        else:
            img = cv2.imread(fpath)
            ksize = int(radius * 4) | 1
            blurred = cv2.GaussianBlur(img, (ksize, ksize), radius)
            sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            cv2.imwrite(out_path, sharpened)
            print(f'  [{img_id}] sharpened (amount={amount})')

    print('Step 4 done.\n')


# ─── Step 5: DiffBIR (batch, skip text images) ─────────────
def run_step5_diffbir(image_ids, input_dir, output_dir, no_diffbir_ids, strength=0.85):
    print('\n' + '=' * 60)
    print(f'STEP 5: DiffBIR (strength={strength}, skip text: {sorted(no_diffbir_ids)})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    diffbir_root = os.path.join(BASE_DIR, 'DiffBIR')

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]

    todo_diffbir = []
    for fpath in files:
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip (exists)')
        elif img_id in no_diffbir_ids:
            # Copy from step 4 output (which is the OMDNet output, unsharpened)
            shutil.copy2(fpath, out_path)
            print(f'  [{img_id}] copied (text image, no DiffBIR)')
        else:
            todo_diffbir.append(fpath)

    if not todo_diffbir:
        print('  No images need DiffBIR.')
        print('Step 5 done.\n')
        return

    tmpdir = tempfile.mkdtemp(prefix='diffbir_in_batch_')
    tmp_out = tempfile.mkdtemp(prefix='diffbir_out_batch_')
    for fpath in todo_diffbir:
        img_id = get_image_id(fpath)
        shutil.copy2(fpath, os.path.join(tmpdir, f'{img_id}.png'))
        print(f'  [{img_id}] queued for DiffBIR')

    pos_prompt = (
        "Sharp nighttime street photography, crisp details, high resolution, "
        "well-defined edges, clear text on signs, no motion blur, "
        "hyper detailed photo-realistic maximum detail, 32k, ultra HD."
    )
    neg_prompt = (
        "motion blur, blurry, streaky, smeared, out of focus, "
        "painting, oil painting, illustration, drawing, art, sketch, "
        "CG Style, 3D render, worst quality, low quality, "
        "watermark, signature, jpeg artifacts, deformed, lowres, over-smooth."
    )

    cmd = [
        PYTHON, os.path.join(diffbir_root, 'inference.py'),
        '--task', 'sr',
        '--version', 'v2.1',
        '--upscale', '1',
        '--input', tmpdir,
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

    print(f'  Running DiffBIR on {len(todo_diffbir)} images (single model load)...')
    result = subprocess.run(cmd, cwd=diffbir_root,
                            stdout=sys.stdout, stderr=sys.stderr)

    shutil.rmtree(tmpdir, ignore_errors=True)

    count = 0
    for fpath in todo_diffbir:
        img_id = get_image_id(fpath)
        result_img = os.path.join(tmp_out, f'{img_id}.png')
        if os.path.exists(result_img):
            shutil.move(result_img, os.path.join(output_dir, f'{img_id}.png'))
            print(f'  [{img_id}] Saved')
            count += 1
        else:
            print(f'  [{img_id}] FAILED')

    shutil.rmtree(tmp_out, ignore_errors=True)
    print(f'Step 5 done. ({count}/{len(todo_diffbir)} DiffBIR images)\n')


# ─── Step 6: Sharpen (post-DiffBIR) ────────────────────────
def run_step6_sharpen(image_ids, input_dir, output_dir, amount=0.5, radius=1.0):
    print('\n' + '=' * 60)
    print(f'STEP 6: Post Sharpen (amount={amount})')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]

    for fpath in files:
        img_id = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{img_id}.png')
        if os.path.exists(out_path):
            print(f'  [{img_id}] skip')
            continue
        img = cv2.imread(fpath)
        ksize = int(radius * 4) | 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), radius)
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path, sharpened)
        print(f'  [{img_id}] done')

    print('Step 6 done.\n')


# ─── Step 7: Real-ESRGAN ───────────────────────────────────
def run_step7_realesrgan(image_ids, input_dir, output_dir, outscale=4):
    print('\n' + '=' * 60)
    print(f'STEP 7: Real-ESRGAN x{outscale}')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    ids_str = ','.join(image_ids) if image_ids else None
    sys.argv = ['inference.py', '-i', input_dir, '-o', output_dir,
                '--outscale', str(outscale)]
    if ids_str:
        sys.argv += ['--images', ids_str]

    from RealESRGAN.inference import main as esrgan_main
    esrgan_main()
    print('Step 7 done.\n')


# ─── Main ──────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='01,04,05,06,07,08,09,10,14,15')
    parser.add_argument('--no_diffbir', type=str, default='02,03,08,11,12,13',
                        help='Image IDs to skip DiffBIR (text-heavy)')
    parser.add_argument('--start_from', type=int, default=1)
    parser.add_argument('--strength', type=float, default=0.85)
    args = parser.parse_args()

    image_ids = [x.strip().zfill(2) for x in args.images.split(',')]
    no_diffbir_ids = set(x.strip().zfill(2) for x in args.no_diffbir.split(','))

    results_dir = os.path.join(BASE_DIR, 'results', 'pipeline_v14')
    step1_dir = os.path.join(results_dir, 'step1_clahe')
    step2_dir = os.path.join(results_dir, 'step2_restormer')
    step3_dir = os.path.join(results_dir, 'step3_omdnet')
    step4_dir = os.path.join(results_dir, 'step4_sharpen_pre')
    step5_dir = os.path.join(results_dir, 'step5_diffbir')
    step6_dir = os.path.join(results_dir, 'step6_sharpen_post')
    step7_dir = os.path.join(results_dir, 'step7_realesrgan')

    print('Pipeline v14: CLAHE → Restormer → OMDNet (PIL 1440) → [Sharpen → DiffBIR] → Sharpen → ESRGAN x4')
    print(f'Images: {image_ids}')
    print(f'No DiffBIR (text): {sorted(no_diffbir_ids & set(image_ids))}')
    print(f'DiffBIR strength: {args.strength}')
    t0 = time.time()

    if args.start_from <= 1:
        run_step1_clahe(image_ids, step1_dir)

    if args.start_from <= 2:
        run_step2_restormer(image_ids, step1_dir, step2_dir)

    if args.start_from <= 3:
        run_step3_omdnet(image_ids, step2_dir, step3_dir)

    if args.start_from <= 4:
        run_step4_sharpen_pre(image_ids, step3_dir, step4_dir, no_diffbir_ids)

    if args.start_from <= 5:
        run_step5_diffbir(image_ids, step4_dir, step5_dir, no_diffbir_ids, strength=args.strength)

    if args.start_from <= 6:
        run_step6_sharpen(image_ids, step5_dir, step6_dir)

    if args.start_from <= 7:
        run_step7_realesrgan(image_ids, step6_dir, step7_dir)

    elapsed = time.time() - t0
    print(f'\nPipeline v14 done in {elapsed / 60:.1f} min.')
    print(f'Results: {step7_dir}')


if __name__ == '__main__':
    main()
