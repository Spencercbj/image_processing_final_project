"""
Pipeline v13: Restormer → OMDNet (PIL resize 1440 in-memory) → DiffBIR (strength=0.8) → Real-ESRGAN x4
Fixed: PIL LANCZOS resize inside OMDNet step (not cv2 as separate step).
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


def run_step1_restormer(image_ids, output_dir):
    print('\n' + '=' * 60)
    print('STEP 1: Restormer (Motion Deblurring)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    from Restormer.inference import load_model, tile_inference
    device = 'cuda'
    weights = os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                           'pretrained_models', 'motion_deblurring.pth')
    net = load_model(weights, device)

    for fpath in collect_photos(image_ids):
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
    print('Step 1 done.\n')


def run_step2_omdnet(image_ids, input_dir, output_dir, target_max_dim=1440):
    """OMDNet with PIL LANCZOS resize in-memory (NOT cv2, NOT saved to disk)."""
    print('\n' + '=' * 60)
    print(f'STEP 2: OMDNet (PIL resize {target_max_dim}, whole image)')
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
    print('Step 2 done.\n')


def run_step3_diffbir(image_ids, input_dir, output_dir, strength=0.8):
    """DiffBIR refinement — single subprocess for all images (one model load)."""
    print('\n' + '=' * 60)
    print(f'STEP 3: DiffBIR (strength={strength}, start_point=cond)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    diffbir_root = os.path.join(BASE_DIR, 'DiffBIR')

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        id_set = set(image_ids)
        files = [f for f in files if get_image_id(f) in id_set]

    # Filter out already-done images
    todo = []
    for fpath in files:
        img_id = get_image_id(fpath)
        if os.path.exists(os.path.join(output_dir, f'{img_id}.png')):
            print(f'  [{img_id}] skip')
        else:
            todo.append(fpath)

    if not todo:
        print('  All images already done.')
        print('Step 3 done.\n')
        return

    # Copy all pending images into one temp dir → single DiffBIR call
    tmpdir = tempfile.mkdtemp(prefix='diffbir_in_batch_')
    tmp_out = tempfile.mkdtemp(prefix='diffbir_out_batch_')
    for fpath in todo:
        img_id = get_image_id(fpath)
        shutil.copy2(fpath, os.path.join(tmpdir, f'{img_id}.png'))
        print(f'  [{img_id}] queued')

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

    print(f'  Running DiffBIR on {len(todo)} images (single model load)...')
    result = subprocess.run(cmd, cwd=diffbir_root,
                            stdout=sys.stdout, stderr=sys.stderr)

    shutil.rmtree(tmpdir, ignore_errors=True)

    # Move results
    count = 0
    for fpath in todo:
        img_id = get_image_id(fpath)
        result_img = os.path.join(tmp_out, f'{img_id}.png')
        if os.path.exists(result_img):
            shutil.move(result_img, os.path.join(output_dir, f'{img_id}.png'))
            print(f'  [{img_id}] Saved')
            count += 1
        else:
            print(f'  [{img_id}] FAILED')

    shutil.rmtree(tmp_out, ignore_errors=True)
    print(f'Step 3 done. ({count}/{len(todo)} images)\n')


def run_step4_sharpen(image_ids, input_dir, output_dir, amount=0.5, radius=1.0):
    print('\n' + '=' * 60)
    print(f'STEP 4: Sharpen (amount={amount}, radius={radius})')
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

    print('Step 4 done.\n')


def run_step5_realesrgan(image_ids, input_dir, output_dir, outscale=4):
    print('\n' + '=' * 60)
    print(f'STEP 5: Real-ESRGAN x{outscale}')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)

    ids_str = ','.join(image_ids) if image_ids else None
    sys.argv = ['inference.py', '-i', input_dir, '-o', output_dir,
                '--outscale', str(outscale)]
    if ids_str:
        sys.argv += ['--images', ids_str]

    from RealESRGAN.inference import main as esrgan_main
    esrgan_main()
    print('Step 5 done.\n')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='01,05,08,11,12,15')
    parser.add_argument('--start_from', type=int, default=1)
    parser.add_argument('--strength', type=float, default=0.8)
    parser.add_argument('--name', type=str, default=None,
                        help='Results folder name. Default: pipeline_v13_s{strength}')
    parser.add_argument('--shared_dir', type=str, default=None,
                        help='Reuse step1/step2 from this dir instead of recomputing')
    args = parser.parse_args()

    image_ids = [x.strip().zfill(2) for x in args.images.split(',')]
    s_tag = str(args.strength).replace('.', '')
    folder_name = args.name or f'pipeline_v13_s{s_tag}'
    results_dir = os.path.join(BASE_DIR, 'results', folder_name)

    # If shared_dir provided, reuse its step1/step2
    if args.shared_dir:
        shared = os.path.join(BASE_DIR, 'results', args.shared_dir)
        step1_dir = os.path.join(shared, 'step1_restormer')
        step2_dir = os.path.join(shared, 'step2_omdnet')
    else:
        step1_dir = os.path.join(results_dir, 'step1_restormer')
        step2_dir = os.path.join(results_dir, 'step2_omdnet')

    step3_dir = os.path.join(results_dir, 'step3_diffbir')
    step4_dir = os.path.join(results_dir, 'step4_sharpen')
    step5_dir = os.path.join(results_dir, 'step5_realesrgan')

    print(f'Pipeline v13: Restormer → OMDNet (PIL 1440) → DiffBIR (s={args.strength}) → Sharpen → ESRGAN x4')
    print(f'Images: {image_ids}, Output: {folder_name}')
    t0 = time.time()

    if not args.shared_dir:
        # Reuse Restormer results from previous pipelines if available
        prev_sources = [
            os.path.join(BASE_DIR, 'results', 'pipeline_v13', 'step1_restormer'),
            os.path.join(BASE_DIR, 'results', 'restormer_omd_esrgan', 'step1_restormer'),
            os.path.join(BASE_DIR, 'results', 'pipeline_v12', 'step1_restormer'),
        ]
        if args.start_from <= 1:
            os.makedirs(step1_dir, exist_ok=True)
            for img_id in image_ids:
                dst = os.path.join(step1_dir, f'{img_id}.png')
                if os.path.exists(dst):
                    continue
                for prev_dir in prev_sources:
                    src = os.path.join(prev_dir, f'{img_id}.png')
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                        print(f'  [{img_id}] Reused Restormer from {os.path.basename(os.path.dirname(prev_dir))}')
                        break

        if args.start_from <= 1:
            run_step1_restormer(image_ids, step1_dir)

        if args.start_from <= 2:
            run_step2_omdnet(image_ids, step1_dir, step2_dir)

    if args.start_from <= 3:
        run_step3_diffbir(image_ids, step2_dir, step3_dir, strength=args.strength)

    if args.start_from <= 4:
        run_step4_sharpen(image_ids, step3_dir, step4_dir, amount=0.5)

    if args.start_from <= 5:
        run_step5_realesrgan(image_ids, step4_dir, step5_dir)

    elapsed = time.time() - t0
    print(f'\nPipeline done in {elapsed / 60:.1f} min.')
    print(f'Results: {step5_dir}')


if __name__ == '__main__':
    main()
