"""
Pipeline v2: HVI-CIDNet → MISCFilter → Restormer → SDEdit → Real-ESRGAN → EasyOCR blending

Changes from v1:
  - HVI-CIDNet: fixed tile blending (cosine ramp), overlap=128
  - Replaced PASD with SDEdit (img2img, strength=0.35)
  - All results stored under results/pipeline_v2/{step_name}/
"""
import argparse
import os
import sys
import subprocess
import time

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))


STEPS = [
    {
        'name': '01_HVI-CIDNet',
        'script': os.path.join(CODE_ROOT, 'HVI_CIDNet', 'inference.py'),
        'input_from': None,  # photos dir
        'args': ['--tile_size', '512', '--overlap', '128'],
    },
    {
        'name': '02_MISCFilter',
        'script': os.path.join(CODE_ROOT, 'MISCFilter', 'inference.py'),
        'input_from': '01_HVI-CIDNet',
        'args': [],
    },
    {
        'name': '03_Restormer',
        'script': os.path.join(CODE_ROOT, 'Restormer', 'inference.py'),
        'input_from': '02_MISCFilter',
        'args': [],
        'per_image': True,
    },
    {
        'name': '04_SDEdit',
        'script': os.path.join(CODE_ROOT, 'SDEdit', 'inference.py'),
        'input_from': '03_Restormer',
        'args': ['--strength', '0.35', '--num_inference_steps', '30',
                 '--guidance_scale', '7.5', '--max_edge', '768'],
    },
    {
        'name': '05_RealESRGAN',
        'script': os.path.join(CODE_ROOT, 'RealESRGAN', 'inference.py'),
        'input_from': '04_SDEdit',
        'args': [],
    },
    {
        'name': '06_blend_text',
        'script': os.path.join(CODE_ROOT, 'blend_text', 'blend_v2.py'),
        'input_from': None,  # special: reads from 03 and 05
        'args': [],
        'blend': True,
    },
]


def run_step(step, pipeline_dir, photos_dir, image_ids, python_exe):
    step_out = os.path.join(pipeline_dir, step['name'])
    os.makedirs(step_out, exist_ok=True)

    images_arg = ['--images', image_ids] if image_ids else []

    if step.get('blend'):
        restormer_dir = os.path.join(pipeline_dir, '03_Restormer')
        diffusion_dir = os.path.join(pipeline_dir, '05_RealESRGAN')
        cmd = [python_exe, step['script'],
               '--restormer_dir', restormer_dir,
               '--diffusion_dir', diffusion_dir,
               '--output', step_out] + images_arg + step.get('args', [])
    elif step.get('per_image'):
        import glob, re
        input_dir = os.path.join(pipeline_dir, step['input_from'])
        files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
        filter_set = set(image_ids.split(',')) if image_ids else None
        for fpath in files:
            m = re.match(r'^(\d+)', os.path.basename(fpath))
            if not m:
                continue
            fid = m.group(1)
            if filter_set and fid not in filter_set:
                continue
            out_path = os.path.join(step_out, f'{fid}.png')
            cmd = [python_exe, step['script'], '-i', fpath, '-o', out_path] + step.get('args', [])
            print(f'  Running: {os.path.basename(step["script"])} on {fid}')
            t0 = time.time()
            result = subprocess.run(cmd)
            elapsed = time.time() - t0
            print(f'  {fid} done in {elapsed:.1f}s (exit={result.returncode})')
        return
    else:
        input_dir = photos_dir if step['input_from'] is None else os.path.join(pipeline_dir, step['input_from'])
        cmd = [python_exe, step['script'],
               '--input', input_dir,
               '--output', step_out] + images_arg + step.get('args', [])

    print(f'  Command: {" ".join(cmd[:4])} ...')
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    print(f'  Finished in {elapsed:.1f}s (exit={result.returncode})')
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--photos', type=str, default=os.path.join(RESULTS_ROOT, '..', 'photos'))
    parser.add_argument('--output', type=str, default=os.path.join(RESULTS_ROOT, 'pipeline_v2'))
    parser.add_argument('--images', type=str, default=None)
    parser.add_argument('--start_from', type=int, default=1)
    parser.add_argument('--end_at', type=int, default=6)
    parser.add_argument('--skip_steps', type=str, default=None,
                        help='Comma-separated step numbers to skip, e.g. "5,6"')
    args = parser.parse_args()

    args.photos = os.path.abspath(args.photos)
    args.output = os.path.abspath(args.output)
    os.makedirs(args.output, exist_ok=True)

    python_exe = sys.executable
    skip = set()
    if args.skip_steps:
        skip = set(int(x) for x in args.skip_steps.split(','))

    total_time = 0
    for idx, step in enumerate(STEPS, 1):
        if idx < args.start_from or idx > args.end_at:
            continue
        if idx in skip:
            print(f'\n=== SKIP Step {idx}: {step["name"]} ===')
            continue
        print(f'\n{"="*60}')
        print(f'=== Step {idx}/{len(STEPS)}: {step["name"]} ===')
        print(f'{"="*60}')
        elapsed = run_step(step, args.output, args.photos, args.images, python_exe)
        if elapsed:
            total_time += elapsed

    print(f'\n{"="*60}')
    print(f'Pipeline v2 complete. Total time: {total_time:.0f}s ({total_time/60:.1f}min)')
    print(f'Results in: {args.output}')


if __name__ == '__main__':
    main()
