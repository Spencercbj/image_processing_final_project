"""
Image 04 aggressive low-resolution deblur experiment.
Hypothesis (v5 finding + ideas.md): 04's blur kernel (~165 deg, est >80px @1440)
is outside model training range. Downscaling to 480/720 shrinks the kernel
proportionally into range; deblur there, then ESRGAN x4 restores resolution.

For each scale in {480, 720}:
  PIL resize original -> [Restormer | NAFNet | NAFNet->OMDNet] -> ESRGAN x4
OMDNet receives PIL-resized input (low-res is the intentional exception here).
"""
import os
import sys
import shutil
import subprocess

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import functional as tf

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'OMDNet'))
sys.path.insert(0, os.path.join(CODE_DIR, '..'))
PYTHON = sys.executable
ESRGAN_ROOT = os.path.join(BASE_DIR, 'Real-ESRGAN')

from Restormer.inference import load_model as rest_load, tile_inference as rest_tile
from NAFNet.inference import load_model as naf_load
from models.deblur_model import OMDNet
from utils.general import move_to_cuda
from utils.record import load_checkpoint, plot_img

SRC = os.path.join(BASE_DIR, 'photos', '04_Cyclist_Passing_Warm_Storefront_Lights.jpg')
OUT = os.path.join(BASE_DIR, 'results', 'pipeline_04_lowres')
SCALES = [480, 720]


def pil_resize(path, max_dim):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    sf = max_dim / max(w, h)
    return img.resize((int(w * sf), int(h * sf)), Image.LANCZOS)


def to_tensor_bgr(pil):
    rgb = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def save_tensor(t, h, w, path):
    out = t[:, :, :h, :w].squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    cv2.imwrite(path, (cv2.cvtColor(out, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8))


def run_restormer(net, pil, path):
    t = to_tensor_bgr(pil); h, w = pil.size[1], pil.size[0]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    if ph or pw: t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    out = rest_tile(net, t, tile_size=512, overlap=128, device='cuda')
    save_tensor(out, h, w, path); torch.cuda.empty_cache()


def run_nafnet(net, pil, path):
    t = to_tensor_bgr(pil); h, w = pil.size[1], pil.size[0]
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    if ph or pw: t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    with torch.no_grad():
        out = net(t.cuda()).cpu()
    save_tensor(out, h, w, path); torch.cuda.empty_cache()


def run_omdnet(model, in_path, out_path):
    img = np.uint8(np.array(Image.open(in_path).convert('RGB')))
    t = tf.to_tensor(img).unsqueeze_(0)
    if t.shape[2] % 8: t = F.pad(t, (0, 0, 0, 8 - t.shape[2] % 8), mode='reflect')
    if t.shape[3] % 8: t = F.pad(t, (0, 8 - t.shape[3] % 8, 0, 0), mode='reflect')
    t = move_to_cuda(t)
    _, deb, _, _, _, _, _ = model(t, t, t, train=False)
    plot_img(deb[0][0]).save(out_path); torch.cuda.empty_cache()


def main():
    os.makedirs(OUT, exist_ok=True)
    pre = os.path.join(OUT, 'pre_esrgan'); os.makedirs(pre, exist_ok=True)

    # Restormer pass
    print('Restormer...')
    rnet = rest_load(os.path.join(BASE_DIR, 'Restormer', 'Motion_Deblurring',
                                  'pretrained_models', 'motion_deblurring.pth'), 'cuda')
    for s in SCALES:
        pil = pil_resize(SRC, s)
        pil.save(os.path.join(pre, f'04_s{s}_input.png'))
        run_restormer(rnet, pil, os.path.join(pre, f'04_s{s}_rest.png'))
    del rnet; torch.cuda.empty_cache()

    # NAFNet pass
    print('NAFNet...')
    nnet = naf_load(os.path.join(BASE_DIR, 'NAFNet', 'experiments',
                                 'pretrained_models', 'NAFNet-GoPro-width64.pth'),
                    width=64, device='cuda')
    for s in SCALES:
        pil = pil_resize(SRC, s)
        run_nafnet(nnet, pil, os.path.join(pre, f'04_s{s}_naf.png'))
    del nnet; torch.cuda.empty_cache()

    # OMDNet refine on NAFNet output (low-res)
    print('OMDNet on NAFNet output...')
    model = OMDNet(num_res=20)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    load_checkpoint(model, opt, None, os.path.join(BASE_DIR, 'OMDNet', 'checkpoints', 'test1'))
    torch.set_grad_enabled(False); model.eval()
    for s in SCALES:
        run_omdnet(model, os.path.join(pre, f'04_s{s}_naf.png'),
                   os.path.join(pre, f'04_s{s}_naf_omd.png'))
    del model; torch.cuda.empty_cache(); torch.set_grad_enabled(True)

    # ESRGAN x4 on all pre_esrgan variants (except the raw inputs)
    print('ESRGAN x4...')
    esr_in = os.path.join(OUT, '_esr_in'); os.makedirs(esr_in, exist_ok=True)
    for f in os.listdir(pre):
        if f.endswith('.png') and 'input' not in f:
            shutil.copy2(os.path.join(pre, f), os.path.join(esr_in, f))
    cmd = [PYTHON, os.path.join(ESRGAN_ROOT, 'inference_realesrgan.py'),
           '-n', 'RealESRGAN_x4plus', '-i', esr_in, '-o', OUT,
           '-s', '4', '--tile', '400', '--suffix', '']
    subprocess.run(cmd, cwd=ESRGAN_ROOT)
    shutil.rmtree(esr_in, ignore_errors=True)
    print(f'Done. Results in {OUT}')


if __name__ == '__main__':
    main()
