import argparse
import os
import sys
import glob
import re
import shutil
import subprocess

ESRGAN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Real-ESRGAN'))


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--images', type=str, default=None,
                        help='Comma-separated image IDs to process, e.g. "01,08,12"')
    parser.add_argument('--model_name', type=str, default='RealESRGAN_x4plus')
    parser.add_argument('--outscale', type=int, default=1)
    parser.add_argument('--tile', type=int, default=256)
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

    tmp_in = os.path.join(args.output, '_tmp_esrgan_in')
    os.makedirs(tmp_in, exist_ok=True)

    selected = []
    for fpath in files:
        img_id = get_image_id(fpath)
        if filter_ids and img_id not in filter_ids:
            continue
        out_name = f'{img_id}.png' if img_id else os.path.splitext(os.path.basename(fpath))[0] + '.png'
        tmp_path = os.path.join(tmp_in, out_name)
        shutil.copy2(fpath, tmp_path)
        selected.append(img_id)

    if not selected:
        print('No images to process')
        return

    cmd = [
        sys.executable, os.path.join(ESRGAN_ROOT, 'inference_realesrgan.py'),
        '-n', args.model_name,
        '-i', tmp_in,
        '-o', args.output,
        '-s', str(args.outscale),
        '--tile', str(args.tile),
        '--suffix', '',
    ]

    print(f'Running Real-ESRGAN (scale={args.outscale}, tile={args.tile}) on {selected} ...')
    result = subprocess.run(cmd, cwd=ESRGAN_ROOT)

    shutil.rmtree(tmp_in, ignore_errors=True)

    if result.returncode != 0:
        print('Real-ESRGAN had errors, check output')
    else:
        print('Real-ESRGAN done.')


if __name__ == '__main__':
    main()
