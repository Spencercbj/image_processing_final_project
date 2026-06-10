"""
Pipeline v23: NAFNet (GoPro) as the deblur backbone instead of Restormer.
PIL resize 1440 -> NAFNet(GoPro) -> OMDNet(PIL 1440 input) -> Sharpen(0.5) -> ESRGAN x4
NAFNet has never been used before; GoPro benchmark > Restormer on motion blur.

Reuses v18's resize/omdnet/sharpen/esrgan helpers for consistency.
OMDNet always receives the PIL-1440 NAFNet output (kept at 1440), per project rule.
"""
import os
import sys
import glob
import time

import torch
import torch.nn.functional as F
import numpy as np
import cv2

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
sys.path.insert(0, os.path.join(CODE_DIR, '..'))   # for NAFNet.inference, OMDNet via v18

import run_pipeline_v18 as v18   # reuse resize / omdnet / sharpen / esrgan
from NAFNet.inference import load_model as naf_load, tile_inference as naf_tile

NAF_WEIGHTS = os.path.join(BASE_DIR, 'NAFNet', 'experiments',
                           'pretrained_models', 'NAFNet-GoPro-width64.pth')


def get_image_id(p):
    return v18.get_image_id(p)


def run_nafnet(image_ids, input_dir, output_dir, tile_size=512, overlap=64):
    print('\n' + '=' * 60)
    print('STEP 2: NAFNet (GoPro)')
    print('=' * 60)
    os.makedirs(output_dir, exist_ok=True)
    net = naf_load(NAF_WEIGHTS, width=64, device='cuda')

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if image_ids:
        ids = set(image_ids)
        files = [f for f in files if get_image_id(f) in ids]

    for fpath in files:
        iid = get_image_id(fpath)
        out_path = os.path.join(output_dir, f'{iid}.png')
        if os.path.exists(out_path):
            print(f'  [{iid}] skip'); continue
        img = cv2.imread(fpath)
        h, w = img.shape[:2]
        print(f'  [{iid}] {w}x{h}')
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode='reflect')
        out = naf_tile(net, t, tile_size=tile_size, overlap=overlap, device='cuda')
        out = out[:, :, :h, :w].squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        out_bgr = (cv2.cvtColor(out, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)
        cv2.imwrite(out_path, out_bgr)
        print(f'    Saved')
        torch.cuda.empty_cache()

    del net
    torch.cuda.empty_cache()
    print('Step 2 done.\n')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=str, default='01,04,05,06,07,08,09')
    args = ap.parse_args()
    image_ids = [x.strip().zfill(2) for x in args.images.split(',')]

    rd = os.path.join(BASE_DIR, 'results', 'pipeline_v23_nafnet')
    s1 = os.path.join(rd, 'step1_resize')
    s2 = os.path.join(rd, 'step2_nafnet')
    s3 = os.path.join(rd, 'step3_omdnet')
    s4 = os.path.join(rd, 'step4_sharpen')
    s5 = os.path.join(rd, 'step5_realesrgan')

    print('Pipeline v23: resize1440 -> NAFNet(GoPro) -> OMDNet -> Sharpen -> ESRGAN x4')
    print(f'Images: {image_ids}')
    t0 = time.time()

    v18.run_step1_resize(image_ids, s1)
    run_nafnet(image_ids, s1, s2)
    v18.run_step3_omdnet(image_ids, s2, s3)       # OMDNet gets PIL-1440 NAFNet output
    v18.run_step5_sharpen(image_ids, s3, s4, amount=0.5)
    v18.run_step6_realesrgan(image_ids, s4, s5)

    print(f'\nv23 done in {(time.time()-t0)/60:.1f} min. Results: {s5}')


if __name__ == '__main__':
    main()
