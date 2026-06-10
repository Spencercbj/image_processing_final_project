"""
Pipeline: Deconvolution -> DiffBIR (SDEdit mode with low strength)
Uses start_point_type=cond for SDEdit-like behavior.
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
DECONV_SCRIPT = os.path.join(BASE, 'code', 'deconvolution', 'inference.py')
PHOTOS = os.path.join(BASE, 'photos')
OUTPUT_DIR = os.path.join(BASE, 'results', 'pipeline_deconv_DiffBIR')
PYTHON = sys.executable


def get_short_name(filename):
    m = re.match(r'^(\d+)', filename)
    return m.group(1) if m else None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Stage 1: Deconvolution on original photos
    deconv_tmp = tempfile.mkdtemp(prefix='deconv_out_')
    src_files = sorted(glob.glob(os.path.join(PHOTOS, '*.*')))

    print('=== Stage 1: Wiener Deconvolution ===')
    for src in src_files:
        fname = os.path.basename(src)
        short = get_short_name(fname)
        if short is None:
            continue
        dst = os.path.join(deconv_tmp, f'{short}.png')
        print(f'  [{short}] {fname}')
        result = subprocess.run(
            [PYTHON, DECONV_SCRIPT, '-i', src, '-o', dst,
             '--kernel_size', '21', '--snr', '0.005'],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f'    FAILED: {result.stderr[-300:]}')

    print(f'\nDeconvolution done. {len(os.listdir(deconv_tmp))} images.')

    # Stage 2: DiffBIR with SDEdit mode
    print('\n=== Stage 2: DiffBIR (SDEdit mode) ===')

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
        '--input', deconv_tmp,
        '--output', OUTPUT_DIR,
        '--captioner', 'none',
        '--precision', 'fp16',
        '--start_point_type', 'cond',
        '--strength', '0.4',
        '--noise_aug', '50',
        '--pos_prompt', pos_prompt,
        '--neg_prompt', neg_prompt,
        # tiled inference
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

    shutil.rmtree(deconv_tmp, ignore_errors=True)

    if result.returncode != 0:
        print(f'\nDiffBIR failed with exit code {result.returncode}')
        sys.exit(1)

    print(f'\nResults saved to {OUTPUT_DIR}')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f'  {f}')


if __name__ == '__main__':
    main()
