"""
Multi-stage image deblurring pipeline:
  原圖 → HVI-CIDNet → MISCFilter → Restormer → PASD → Real-ESRGAN → EasyOCR blending

Each step saves intermediate results to ./results/{step_name}/.
Each step runs as a subprocess to cleanly free GPU memory between steps.
"""
import argparse
import os
import sys
import time
import subprocess

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CODE_DIR = os.path.join(BASE_DIR, 'code')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PHOTOS_DIR = os.path.join(BASE_DIR, 'photos')

STEPS = [
    {
        'name': '01_HVI-CIDNet',
        'script': os.path.join(CODE_DIR, 'HVI_CIDNet', 'inference.py'),
        'input_from': None,  # uses original photos
    },
    {
        'name': '02_MISCFilter',
        'script': os.path.join(CODE_DIR, 'MISCFilter', 'inference.py'),
        'input_from': '01_HVI-CIDNet',
    },
    {
        'name': '03_Restormer',
        'script': os.path.join(CODE_DIR, 'Restormer', 'inference.py'),
        'input_from': '02_MISCFilter',
        'per_image': True,
    },
    {
        'name': '04_PASD',
        'script': os.path.join(CODE_DIR, 'PASD', 'inference.py'),
        'input_from': '03_Restormer',
    },
    {
        'name': '05_RealESRGAN',
        'script': os.path.join(CODE_DIR, 'RealESRGAN', 'inference.py'),
        'input_from': '04_PASD',
    },
    {
        'name': '06_blended',
        'script': os.path.join(CODE_DIR, 'blend_text', 'blend_v2.py'),
        'input_from': None,  # special: reads from 03_Restormer and 05_RealESRGAN
    },
]


def run_step(step, images_arg, python=sys.executable, start_from_step=None):
    name = step['name']
    script = step['script']
    output_dir = os.path.join(RESULTS_DIR, name)

    if step.get('input_from'):
        input_dir = os.path.join(RESULTS_DIR, step['input_from'])
    else:
        input_dir = PHOTOS_DIR

    print(f'\n{"="*60}')
    print(f'  Step: {name}')
    print(f'  Input: {input_dir}')
    print(f'  Output: {output_dir}')
    print(f'{"="*60}\n')

    if name == '06_blended':
        cmd = [
            python, script,
            '--restormer_dir', os.path.join(RESULTS_DIR, '03_Restormer'),
            '--diffusion_dir', os.path.join(RESULTS_DIR, '05_RealESRGAN'),
            '--output', output_dir,
        ]
        if images_arg:
            cmd += ['--images', images_arg]
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        print(f'\n  {name} finished in {elapsed:.1f}s (exit code {result.returncode})')
        return elapsed, result.returncode

    if step.get('per_image'):
        import glob
        import re
        files = sorted(glob.glob(os.path.join(input_dir, '*.*')))
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        os.makedirs(output_dir, exist_ok=True)

        filter_ids = None
        if images_arg:
            filter_ids = set(images_arg.split(','))

        total_elapsed = 0
        for fpath in files:
            m = re.match(r'^(\d+)', os.path.basename(fpath))
            img_id = m.group(1) if m else None
            if filter_ids and img_id not in filter_ids:
                continue
            out_path = os.path.join(output_dir, f'{img_id}.png')
            cmd = [python, script, '-i', fpath, '-o', out_path]
            t0 = time.time()
            result = subprocess.run(cmd)
            elapsed = time.time() - t0
            total_elapsed += elapsed
            print(f'  {img_id}: {elapsed:.1f}s')
        print(f'\n  {name} total: {total_elapsed:.1f}s')
        return total_elapsed, 0

    cmd = [python, script, '-i', input_dir, '-o', output_dir]
    if images_arg:
        cmd += ['--images', images_arg]
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    print(f'\n  {name} finished in {elapsed:.1f}s (exit code {result.returncode})')
    return elapsed, result.returncode


def main():
    parser = argparse.ArgumentParser(description='Multi-stage image deblurring pipeline')
    parser.add_argument('--images', type=str, default=None,
                        help='Comma-separated image IDs, e.g. "01,08,12"')
    parser.add_argument('--start_from', type=int, default=1,
                        help='Start from step N (1-6)')
    parser.add_argument('--end_at', type=int, default=6,
                        help='End at step N (1-6)')
    parser.add_argument('--skip_steps', type=str, default=None,
                        help='Comma-separated step numbers to skip, e.g. "1,4"')
    args = parser.parse_args()

    skip = set()
    if args.skip_steps:
        skip = set(int(x) for x in args.skip_steps.split(','))

    print(f'Pipeline: images={args.images or "all"}, steps {args.start_from}-{args.end_at}')
    if skip:
        print(f'Skipping steps: {skip}')
    print()

    total_time = 0
    step_times = {}

    for i, step in enumerate(STEPS, 1):
        if i < args.start_from or i > args.end_at:
            continue
        if i in skip:
            print(f'\n  [SKIP] Step {i}: {step["name"]}')
            continue

        elapsed, code = run_step(step, args.images)
        step_times[step['name']] = elapsed
        total_time += elapsed

        if code != 0:
            print(f'\n  WARNING: Step {step["name"]} failed (exit code {code})')
            resp = input('  Continue to next step? (y/n): ').strip().lower()
            if resp != 'y':
                break

    print(f'\n{"="*60}')
    print(f'  Pipeline complete!')
    print(f'  Total time: {total_time:.1f}s ({total_time/60:.1f}min)')
    print(f'\n  Per-step times:')
    for name, t in step_times.items():
        print(f'    {name}: {t:.1f}s')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
