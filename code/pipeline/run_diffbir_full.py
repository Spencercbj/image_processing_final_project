"""
DiffBIR full blind-restoration mode (NOT SDEdit) for images 04 and 10.
Difference from v22: start_point_type=noise + strength 1.0 + more steps, so the
stage-1 cleaner (SwinIR) output drives IRControlNet generation from pure noise.
This is DiffBIR's native blind-IR behaviour, distinct from the cond/SDEdit path.

resize-1440 input (reuse v18) -> DiffBIR full (noise, 30 steps) -> ESRGAN x4
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
RESIZE_DIR = os.path.join(BASE_DIR, 'results', 'pipeline_v18', 'step1_resize')
OUT = os.path.join(BASE_DIR, 'results', 'pipeline_diffbir_full')

IMAGE_IDS = ['04', '10']

neg_prompt = ("painting, oil painting, illustration, drawing, art, sketch, cartoon, "
              "CG Style, 3D render, unreal engine, blurring, dirty, messy, "
              "worst quality, low quality, frames, watermark, signature, jpeg artifacts, "
              "deformed, lowres, over-smooth")
pos_prompt = ("sharp photograph, high resolution, highly detailed, "
              "hyper detailed photo-realistic, ultra HD, well-defined edges, hyper sharpness")


def run_diffbir_full(input_dir, output_dir, steps=30):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        PYTHON, os.path.join(DIFFBIR, 'inference.py'),
        '--task', 'sr', '--version', 'v2.1', '--upscale', '1',
        '--start_point_type', 'noise', '--strength', '1.0',
        '--noise_aug', '0', '--steps', str(steps), '--cfg_scale', '4.0',
        '--cleaner_tiled', '--cleaner_tile_size', '512',
        '--vae_encoder_tiled', '--vae_encoder_tile_size', '512',
        '--vae_decoder_tiled', '--vae_decoder_tile_size', '512',
        '--cldm_tiled', '--cldm_tile_size', '512',
        '--captioner', 'none',
        '--neg_prompt', neg_prompt, '--pos_prompt', pos_prompt,
        '--input', input_dir, '--output', output_dir,
    ]
    print(f'  DiffBIR full (noise, steps={steps}) ...')
    t0 = time.time()
    r = subprocess.run(cmd, cwd=DIFFBIR)
    print(f'    exit={r.returncode} ({time.time()-t0:.1f}s)')


def run_esrgan(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tmp = os.path.join(output_dir, '_tmp'); os.makedirs(tmp, exist_ok=True)
    for fn in os.listdir(input_dir):
        if fn.endswith('.png'):
            shutil.copy2(os.path.join(input_dir, fn), os.path.join(tmp, fn))
    cmd = [PYTHON, os.path.join(ESRGAN_ROOT, 'inference_realesrgan.py'),
           '-n', 'RealESRGAN_x4plus', '-i', tmp, '-o', output_dir,
           '-s', '4', '--tile', '256', '--suffix', '']
    subprocess.run(cmd, cwd=ESRGAN_ROOT)
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    tmp_in = os.path.join(OUT, '_input'); os.makedirs(tmp_in, exist_ok=True)
    for iid in IMAGE_IDS:
        shutil.copy2(os.path.join(RESIZE_DIR, f'{iid}.png'), os.path.join(tmp_in, f'{iid}.png'))
    db = os.path.join(OUT, 'step1_diffbir_full')
    es = os.path.join(OUT, 'step2_esrgan')
    run_diffbir_full(tmp_in, db)
    run_esrgan(db, es)
    shutil.rmtree(tmp_in, ignore_errors=True)
    print(f'Done. Results: {es}')


if __name__ == '__main__':
    main()
