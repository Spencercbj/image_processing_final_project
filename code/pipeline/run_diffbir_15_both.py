"""One-off: DiffBIR SDEdit s=0.8 + ESRGAN face enhance on v13 and v19 sharpen outputs of image 15."""
import os, shutil, subprocess, sys, tempfile

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DIFFBIR = os.path.join(BASE, 'DiffBIR')
ESRGAN = os.path.join(BASE, 'Real-ESRGAN')
PYTHON = sys.executable

sources = {
    'v13': os.path.join(BASE, 'results', 'v13_omd_15_face', '15_sharpen.png'),
    'v19': os.path.join(BASE, 'results', 'v19_omd_15_face', '15_sharpen.png'),
}

tmpdir = tempfile.mkdtemp(prefix='diffbir_15_')
tmp_out = tempfile.mkdtemp(prefix='diffbir_15_out_')
shutil.copy2(sources['v13'], os.path.join(tmpdir, '15_v13.png'))
shutil.copy2(sources['v19'], os.path.join(tmpdir, '15_v19.png'))

neg_prompt = (
    "painting, oil painting, illustration, drawing, art, sketch, cartoon, "
    "CG Style, 3D render, unreal engine, blurring, dirty, messy, "
    "worst quality, low quality, frames, watermark, signature, jpeg artifacts, "
    "deformed, lowres, over-smooth, "
    "ghost, ghosting, motion trail, double exposure, transparent overlay, "
    "duplicate face, blurred edges, halo artifact, chromatic aberration"
)
pos_prompt = (
    "sharp photograph, high resolution, highly detailed, taken with Canon EOS R, "
    "hyper detailed photo-realistic, 32k, Color Grading, ultra HD, "
    "clear face, sharp facial features, well-defined edges, "
    "skin pore detailing, hyper sharpness, perfect without deformations, no ghosting"
)

cmd = [
    PYTHON, os.path.join(DIFFBIR, 'inference.py'),
    '--task', 'sr', '--version', 'v2.1', '--upscale', '1',
    '--start_point_type', 'cond', '--strength', '0.8',
    '--noise_aug', '0', '--steps', '5', '--cfg_scale', '4.0',
    '--cleaner_tiled', '--cleaner_tile_size', '512',
    '--vae_encoder_tiled', '--vae_encoder_tile_size', '512',
    '--vae_decoder_tiled', '--vae_decoder_tile_size', '512',
    '--cldm_tiled', '--cldm_tile_size', '512',
    '--captioner', 'none',
    '--neg_prompt', neg_prompt, '--pos_prompt', pos_prompt,
    '--input', tmpdir, '--output', tmp_out,
]
print('Running DiffBIR SDEdit s=0.8 on both images...')
result = subprocess.run(cmd, cwd=DIFFBIR)
print(f'DiffBIR exit code: {result.returncode}')

out_v13_dir = os.path.join(BASE, 'results', 'v13_omd_15_face_diffbir')
out_v19_dir = os.path.join(BASE, 'results', 'v19_omd_15_face_diffbir')
os.makedirs(out_v13_dir, exist_ok=True)
os.makedirs(out_v19_dir, exist_ok=True)

for root, dirs, fnames in os.walk(tmp_out):
    for fn in fnames:
        fp = os.path.join(root, fn)
        if 'v13' in fn:
            shutil.copy2(fp, os.path.join(out_v13_dir, '15_diffbir.png'))
            print('v13 DiffBIR saved')
        elif 'v19' in fn:
            shutil.copy2(fp, os.path.join(out_v19_dir, '15_diffbir.png'))
            print('v19 DiffBIR saved')

shutil.rmtree(tmpdir, ignore_errors=True)
shutil.rmtree(tmp_out, ignore_errors=True)

for label, odir in [('v13', out_v13_dir), ('v19', out_v19_dir)]:
    diffbir_img = os.path.join(odir, '15_diffbir.png')
    if not os.path.exists(diffbir_img):
        print(f'{label} DiffBIR output missing, skip')
        continue
    tmp_in = os.path.join(odir, '_tmp_in')
    os.makedirs(tmp_in, exist_ok=True)
    shutil.copy2(diffbir_img, os.path.join(tmp_in, '15.png'))
    cmd2 = [
        PYTHON, os.path.join(ESRGAN, 'inference_realesrgan.py'),
        '-n', 'RealESRGAN_x4plus',
        '-i', tmp_in, '-o', odir,
        '-s', '4', '--tile', '256', '--suffix', '', '--face_enhance',
    ]
    print(f'Running ESRGAN+face on {label}...')
    subprocess.run(cmd2, cwd=ESRGAN)
    shutil.rmtree(tmp_in, ignore_errors=True)
    print(f'{label} done: {os.path.join(odir, "15.png")}')

print('All done!')
