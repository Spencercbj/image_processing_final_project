"""
Pipeline: Restormer (already done) -> DiffBIR (diffusion refinement)
Input: ./results/Restormer/*.png
Output: ./results/pipeline_Restormer_DiffBIR/01.png, 02.png, ...
"""
import subprocess
import sys
import os
import re
import shutil
import glob
import tempfile

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DIFFBIR_ROOT = os.path.join(BASE, 'DiffBIR')
RESTORMER_RESULTS = os.path.join(BASE, 'results', 'Restormer')
OUTPUT_DIR = os.path.join(BASE, 'results', 'pipeline_Restormer_DiffBIR')
PYTHON = sys.executable


def get_short_name(filename):
    """Extract leading number from filename: '01_Urban_...' -> '01'"""
    m = re.match(r'^(\d+)', filename)
    return m.group(1) if m else None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare temp input folder with short names so DiffBIR outputs short names
    tmpdir = tempfile.mkdtemp(prefix='diffbir_input_')
    src_files = sorted(glob.glob(os.path.join(RESTORMER_RESULTS, '*.png')))
    name_map = {}

    for src in src_files:
        fname = os.path.basename(src)
        short = get_short_name(fname)
        if short is None:
            continue
        dst = os.path.join(tmpdir, f'{short}.png')
        shutil.copy2(src, dst)
        name_map[short] = fname
        print(f'  {fname} -> {short}.png')

    print(f'\nPrepared {len(name_map)} images in temp folder')
    print(f'Running DiffBIR v2.1 (this will take a while)...\n')

    cmd = [
        PYTHON, os.path.join(DIFFBIR_ROOT, 'inference.py'),
        '--task', 'sr',
        '--version', 'v2.1',
        '--upscale', '1',
        '--input', tmpdir,
        '--output', OUTPUT_DIR,
        # skip captioner to save VRAM
        '--captioner', 'none',
        '--precision', 'fp16',
        # tiled inference for all stages (8GB VRAM)
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
        '--cfg_scale', '6.0',
        '--seed', '231',
    ]

    result = subprocess.run(
        cmd, cwd=DIFFBIR_ROOT,
        stdout=sys.stdout, stderr=sys.stderr,
    )

    # Cleanup temp folder
    shutil.rmtree(tmpdir, ignore_errors=True)

    if result.returncode != 0:
        print(f'\nDiffBIR failed with exit code {result.returncode}')
        sys.exit(1)

    print(f'\nResults saved to {OUTPUT_DIR}')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f'  {f}')


if __name__ == '__main__':
    main()
