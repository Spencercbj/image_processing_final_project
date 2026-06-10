"""Batch inference: run all photos through all methods."""
import subprocess
import sys
import os
import glob
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTOS = os.path.join(BASE, 'photos')
RESULTS = os.path.join(BASE, 'results')
PYTHON = sys.executable

METHODS = {
    'DarkIR': {
        'script': os.path.join(BASE, 'code', 'DarkIR', 'inference.py'),
        'extra_args': [],
    },
    'Restormer': {
        'script': os.path.join(BASE, 'code', 'Restormer', 'inference.py'),
        'extra_args': [],
    },
    'pipeline_DarkIR_Restormer': {
        'script': os.path.join(BASE, 'code', 'pipeline_DarkIR_Restormer', 'inference.py'),
        'extra_args': [],
    },
}

def main():
    images = sorted(glob.glob(os.path.join(PHOTOS, '*.*')))
    print(f'Found {len(images)} images in {PHOTOS}\n')

    for method, cfg in METHODS.items():
        out_dir = os.path.join(RESULTS, method)
        os.makedirs(out_dir, exist_ok=True)

        print(f'=== {method} ===')
        for img_path in images:
            name = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f'{name}.png')

            if os.path.exists(out_path):
                print(f'  [skip] {name} (already exists)')
                continue

            print(f'  [{method}] {name} ...', end=' ', flush=True)
            t0 = time.time()
            cmd = [PYTHON, cfg['script'],
                   '-i', img_path, '-o', out_path] + cfg['extra_args']
            result = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - t0

            if result.returncode == 0:
                print(f'done ({elapsed:.1f}s)')
            else:
                print(f'FAILED ({elapsed:.1f}s)')
                print(result.stderr[-500:] if result.stderr else 'no error output')
        print()

    print('All done!')


if __name__ == '__main__':
    main()
