"""
Run DiffBIR SDEdit on all Restormer results.
Skips images that already exist in output dir.
Resizes large images (long edge > MAX_EDGE) to avoid OOM on 8GB VRAM.
"""
import subprocess
import sys
import os
import re
import shutil
import glob
import tempfile
import cv2

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DIFFBIR_ROOT = os.path.join(BASE, 'DiffBIR')
RESTORMER_RESULTS = os.path.join(BASE, 'results', 'Restormer')
OUTPUT_DIR = os.path.join(BASE, 'results', 'pipeline_Restormer_DiffBIR_sdedit')
PYTHON = sys.executable

MAX_EDGE = 4096

# Set to None to process all, or a list like [4, 7, 11] to process specific images only
ONLY_IMAGES = [4, 7, 11]


def get_short_name(filename):
    m = re.match(r'^(\d+)', filename)
    return m.group(1) if m else None


def resize_if_needed(src_path, dst_path):
    """Resize image if its long edge exceeds MAX_EDGE. Returns (orig_h, orig_w, was_resized)."""
    img = cv2.imread(src_path)
    h, w = img.shape[:2]
    long_edge = max(h, w)

    if long_edge > MAX_EDGE:
        scale = MAX_EDGE / long_edge
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(dst_path, img_resized)
        print(f'  Resized {w}x{h} -> {new_w}x{new_h} (long edge capped at {MAX_EDGE})')
        return h, w, True
    else:
        shutil.copy2(src_path, dst_path)
        print(f'  Size OK: {w}x{h}')
        return h, w, False


def upscale_to_original(img_path, orig_h, orig_w):
    """Resize output back to original dimensions."""
    img = cv2.imread(img_path)
    if img is None:
        return
    cur_h, cur_w = img.shape[:2]
    if cur_h != orig_h or cur_w != orig_w:
        img_up = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(img_path, img_up)
        print(f'  Upscaled back to {orig_w}x{orig_h}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    src_files = sorted(glob.glob(os.path.join(RESTORMER_RESULTS, '*.png')))

    # Copy existing test result (01.png from test_sdedit_v2)
    test_result = os.path.join(BASE, 'results', 'test_sdedit_v2', '01.png')
    dst_01 = os.path.join(OUTPUT_DIR, '01.png')
    if os.path.exists(test_result) and not os.path.exists(dst_01):
        shutil.copy2(test_result, dst_01)
        print('Copied existing 01.png from test_sdedit_v2')

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

    for src in src_files:
        fname = os.path.basename(src)
        short = get_short_name(fname)
        if short is None:
            continue

        if ONLY_IMAGES is not None and int(short) not in ONLY_IMAGES:
            continue

        out_path = os.path.join(OUTPUT_DIR, f'{short}.png')
        if os.path.exists(out_path):
            print(f'[skip] {short}.png (already exists)')
            continue

        print(f'\n=== Processing {short} ({fname}) ===')

        tmpdir = tempfile.mkdtemp(prefix=f'sdedit_{short}_')
        tmp_out = tempfile.mkdtemp(prefix=f'sdedit_out_{short}_')

        tmp_input_path = os.path.join(tmpdir, f'{short}.png')
        orig_h, orig_w, was_resized = resize_if_needed(src, tmp_input_path)

        cmd = [
            PYTHON, os.path.join(DIFFBIR_ROOT, 'inference.py'),
            '--task', 'sr',
            '--version', 'v2.1',
            '--upscale', '1',
            '--input', tmpdir,
            '--output', tmp_out,
            '--captioner', 'none',
            '--precision', 'fp16',
            '--start_point_type', 'cond',
            '--strength', '0.7',
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

        result = subprocess.run(
            cmd, cwd=DIFFBIR_ROOT,
            stdout=sys.stdout, stderr=sys.stderr,
        )

        shutil.rmtree(tmpdir, ignore_errors=True)

        result_img = os.path.join(tmp_out, f'{short}.png')
        if os.path.exists(result_img):
            shutil.move(result_img, out_path)
            if was_resized:
                upscale_to_original(out_path, orig_h, orig_w)
            print(f'Saved {out_path}')
        else:
            print(f'FAILED for {short}')

        shutil.rmtree(tmp_out, ignore_errors=True)

    print('\nAll done!')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f'  {f}')


if __name__ == '__main__':
    main()
