"""Quick single-image test: Restormer result -> DiffBIR SDEdit mode."""
import subprocess
import sys
import os
import shutil
import tempfile

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DIFFBIR_ROOT = os.path.join(BASE, 'DiffBIR')
PYTHON = sys.executable

INPUT_IMG = os.path.join(BASE, 'results', 'Restormer',
                         '01_Urban_Light_Trails_Walking_Figure.png')
OUTPUT_DIR = os.path.join(BASE, 'results', 'test_sdedit_v2')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# DiffBIR expects a folder as input
tmpdir = tempfile.mkdtemp(prefix='sdedit_test_')
shutil.copy2(INPUT_IMG, os.path.join(tmpdir, '01.png'))

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
    PYTHON, os.path.join(DIFFBIR_ROOT, 'inference.py'),
    '--task', 'sr',
    '--version', 'v2.1',
    '--upscale', '1',
    '--input', tmpdir,
    '--output', OUTPUT_DIR,
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

print(f'Input: {INPUT_IMG}')
print(f'Output: {OUTPUT_DIR}')
print('Running DiffBIR SDEdit mode...\n')

result = subprocess.run(cmd, cwd=DIFFBIR_ROOT, stdout=sys.stdout, stderr=sys.stderr)
shutil.rmtree(tmpdir, ignore_errors=True)

if result.returncode == 0:
    print(f'\nDone! Check {OUTPUT_DIR}/01.png')
else:
    print(f'\nFailed with exit code {result.returncode}')
