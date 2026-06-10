"""
PASD inference wrapper:
  1. Resize large images to MAX_EDGE for VRAM/speed
  2. Run PASD test_pasd.py on all images at once (single model load)
  3. Upscale results back to original dimensions
"""
import argparse
import os
import sys
import glob
import re
import shutil
import subprocess
import cv2

PASD_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'PASD')


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--images', type=str, default=None,
                        help='Comma-separated image IDs to process, e.g. "01,08,12"')
    parser.add_argument('--max_edge', type=int, default=2048)
    parser.add_argument('--decoder_tiled_size', type=int, default=160)
    parser.add_argument('--encoder_tiled_size', type=int, default=512)
    parser.add_argument('--latent_tiled_size', type=int, default=192)
    parser.add_argument('--latent_tiled_overlap', type=int, default=4)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--process_size', type=int, default=768)
    parser.add_argument('--added_noise_level', type=int, default=400)
    args = parser.parse_args()

    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)
    os.makedirs(args.output, exist_ok=True)

    filter_ids = None
    if args.images:
        filter_ids = set(args.images.split(','))

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, '*.*')))
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    else:
        files = [args.input]

    tmp_in = os.path.join(args.output, '_tmp_pasd_in')
    pasd_out = os.path.join(args.output, '_tmp_pasd_out')
    os.makedirs(tmp_in, exist_ok=True)
    os.makedirs(pasd_out, exist_ok=True)

    orig_sizes = {}

    for fpath in files:
        img_id = get_image_id(fpath)
        if filter_ids and img_id not in filter_ids:
            continue
        if img_id is None:
            continue

        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]
        out_name = f'{img_id}.png'
        orig_sizes[img_id] = (w, h)

        if max(h, w) > args.max_edge:
            scale = args.max_edge / max(h, w)
            new_w = int(w * scale) // 8 * 8
            new_h = int(h * scale) // 8 * 8
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            print(f'  {img_id}: resized {w}x{h} -> {new_w}x{new_h}')

        cv2.imwrite(os.path.join(tmp_in, out_name), img)

    if not os.listdir(tmp_in):
        print('No images to process')
        return

    pasd_light_path = os.path.join(PASD_ROOT, 'runs', 'pasd_light')
    subdirs = [d for d in os.listdir(pasd_light_path)
               if os.path.isdir(os.path.join(pasd_light_path, d)) and d.startswith('checkpoint')]
    if subdirs:
        pasd_model = os.path.join(pasd_light_path, sorted(subdirs)[-1])
    else:
        pasd_model = pasd_light_path

    cmd = [
        sys.executable, os.path.join(PASD_ROOT, 'test_pasd.py'),
        '--pretrained_model_path', os.path.join(PASD_ROOT, 'checkpoints', 'stable-diffusion-v1-5'),
        '--pasd_model_path', pasd_model,
        '--image_path', tmp_in,
        '--output_dir', pasd_out,
        '--use_pasd_light',
        '--mixed_precision', 'fp16',
        '--num_inference_steps', str(args.num_inference_steps),
        '--guidance_scale', str(args.guidance_scale),
        '--process_size', str(args.process_size),
        '--decoder_tiled_size', str(args.decoder_tiled_size),
        '--encoder_tiled_size', str(args.encoder_tiled_size),
        '--latent_tiled_size', str(args.latent_tiled_size),
        '--latent_tiled_overlap', str(args.latent_tiled_overlap),
        '--added_noise_level', str(args.added_noise_level),
        '--control_type', 'realisr',
        '--high_level_info', 'caption',
        '--added_prompt', 'clean, high-resolution, 8k, sharp',
        '--negative_prompt', 'blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed',
    ]

    print(f'\nRunning PASD on {len(os.listdir(tmp_in))} images ...')
    print(f'Model path: {pasd_model}')
    result = subprocess.run(cmd, cwd=PASD_ROOT)

    for pasd_file in sorted(glob.glob(os.path.join(pasd_out, '*.png'))):
        img_id = get_image_id(pasd_file)
        if img_id is None:
            continue
        out_path = os.path.join(args.output, f'{img_id}.png')
        if img_id in orig_sizes:
            orig_w, orig_h = orig_sizes[img_id]
            result_img = cv2.imread(pasd_file)
            rh, rw = result_img.shape[:2]
            if (rw, rh) != (orig_w, orig_h):
                result_img = cv2.resize(result_img, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
                print(f'  {img_id}: upscaled {rw}x{rh} -> {orig_w}x{orig_h}')
            cv2.imwrite(out_path, result_img)
        else:
            shutil.move(pasd_file, out_path)
        print(f'  Saved: {out_path}')

    shutil.rmtree(tmp_in, ignore_errors=True)
    shutil.rmtree(pasd_out, ignore_errors=True)
    print('PASD done.')


if __name__ == '__main__':
    main()
