"""
Pipeline v22: DiffBIR SDEdit with low strength for images 04 and 09.
PIL resize 1440 (reuse v18) → DiffBIR SDEdit (s=0.4, 0.5, 0.6) → Real-ESRGAN x4
"""
import os
import sys
import time
import shutil
import subprocess

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PYTHON = sys.executable
DIFFBIR = os.path.join(BASE_DIR, 'DiffBIR')
ESRGAN_ROOT = os.path.join(BASE_DIR, 'Real-ESRGAN')

IMAGE_IDS = ['04', '09']
STRENGTHS = [0.4, 0.5, 0.6]
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'pipeline_v22_diffbir_low')

RESIZE_DIR = os.path.join(BASE_DIR, 'results', 'pipeline_v18', 'step1_resize')

neg_prompt = (
    "painting, oil painting, illustration, drawing, art, sketch, cartoon, "
    "CG Style, 3D render, unreal engine, blurring, dirty, messy, "
    "worst quality, low quality, frames, watermark, signature, jpeg artifacts, "
    "deformed, lowres, over-smooth"
)
pos_prompt = (
    "sharp photograph, high resolution, highly detailed, taken with Canon EOS R, "
    "hyper detailed photo-realistic, 32k, Color Grading, ultra HD, "
    "well-defined edges, hyper sharpness, perfect without deformations"
)


def run_diffbir(strength, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        PYTHON, os.path.join(DIFFBIR, 'inference.py'),
        '--task', 'sr', '--version', 'v2.1', '--upscale', '1',
        '--start_point_type', 'cond', '--strength', str(strength),
        '--noise_aug', '0', '--steps', '5', '--cfg_scale', '4.0',
        '--cleaner_tiled', '--cleaner_tile_size', '512',
        '--vae_encoder_tiled', '--vae_encoder_tile_size', '512',
        '--vae_decoder_tiled', '--vae_decoder_tile_size', '512',
        '--cldm_tiled', '--cldm_tile_size', '512',
        '--captioner', 'none',
        '--neg_prompt', neg_prompt,
        '--pos_prompt', pos_prompt,
        '--input', input_dir,
        '--output', output_dir,
    ]
    print(f'  DiffBIR s={strength} ...')
    t0 = time.time()
    result = subprocess.run(cmd, cwd=DIFFBIR)
    print(f'    exit={result.returncode} ({time.time()-t0:.1f}s)')
    return result.returncode


def run_esrgan(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tmp_in = os.path.join(output_dir, '_tmp_in')
    os.makedirs(tmp_in, exist_ok=True)

    for fn in os.listdir(input_dir):
        if fn.endswith('.png'):
            shutil.copy2(os.path.join(input_dir, fn), os.path.join(tmp_in, fn))

    cmd = [
        PYTHON, os.path.join(ESRGAN_ROOT, 'inference_realesrgan.py'),
        '-n', 'RealESRGAN_x4plus',
        '-i', tmp_in, '-o', output_dir,
        '-s', '4', '--tile', '256', '--suffix', '',
    ]
    print(f'  ESRGAN ...')
    subprocess.run(cmd, cwd=ESRGAN_ROOT)
    shutil.rmtree(tmp_in, ignore_errors=True)


def main():
    t_start = time.time()

    # Prepare input: copy only 04 and 09 from resize dir
    tmp_input = os.path.join(RESULTS_DIR, '_input')
    os.makedirs(tmp_input, exist_ok=True)
    for iid in IMAGE_IDS:
        src = os.path.join(RESIZE_DIR, f'{iid}.png')
        dst = os.path.join(tmp_input, f'{iid}.png')
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    for s in STRENGTHS:
        tag = f's{s:.1f}'.replace('.', '')
        print(f'\n{"=" * 60}')
        print(f'Strength = {s}')
        print(f'{"=" * 60}')

        diffbir_out = os.path.join(RESULTS_DIR, f'step1_diffbir_{tag}')
        esrgan_out = os.path.join(RESULTS_DIR, f'step2_esrgan_{tag}')

        # Check if ESRGAN outputs already exist
        all_exist = all(
            os.path.exists(os.path.join(esrgan_out, f'{iid}.png'))
            for iid in IMAGE_IDS
        )
        if all_exist:
            print(f'  All outputs exist, skip')
            continue

        run_diffbir(s, tmp_input, diffbir_out)
        run_esrgan(diffbir_out, esrgan_out)

    shutil.rmtree(tmp_input, ignore_errors=True)

    dt = time.time() - t_start
    print(f'\n{"=" * 60}')
    print(f'Pipeline v22 done in {dt / 60:.1f} min')
    print(f'Results: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
