"""
OMDNet inference — CVPR 2026 local motion deblurring.
Gate mechanism selectively deblurs only motion-blurred regions.
Tile-based inference with cosine blending for high-res images.
"""
import argparse
import sys
import os
import glob
import re
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math

OMDNET_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'OMDNet')
sys.path.insert(0, OMDNET_ROOT)
from models.deblur_model import OMDNet


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def load_model(weights_path, num_res=20, device='cuda'):
    net = OMDNet(num_res=num_res)
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']['model']
    net.load_state_dict(state, strict=True)
    net.eval().to(device)
    return net


def _make_tile_weight(tile_h, tile_w, overlap, device='cpu'):
    def _ramp(length, ramp_len):
        w = torch.ones(length, device=device)
        if ramp_len > 0:
            ramp = torch.linspace(0, 1, ramp_len + 2, device=device)[1:-1]
            w[:ramp_len] = ramp
            w[-ramp_len:] = ramp.flip(0)
        return w
    wy = _ramp(tile_h, overlap)
    wx = _ramp(tile_w, overlap)
    return (wy[:, None] * wx[None, :]).unsqueeze(0).unsqueeze(0)


def tile_inference(net, img_tensor, tile_size=512, overlap=128, device='cuda'):
    _, _, h, w = img_tensor.shape
    stride = tile_size - overlap
    num_h = max(1, math.ceil((h - overlap) / stride))
    num_w = max(1, math.ceil((w - overlap) / stride))

    output = torch.zeros_like(img_tensor, device='cpu')
    weight = torch.zeros((1, 1, h, w), device='cpu')
    tw = _make_tile_weight(tile_size, tile_size, overlap, device='cpu')

    total = num_h * num_w
    count = 0

    with torch.no_grad():
        for i in range(num_h):
            for j in range(num_w):
                top = min(i * stride, h - tile_size)
                left = min(j * stride, w - tile_size)
                top = max(top, 0)
                left = max(left, 0)
                th = min(tile_size, h - top)
                t_w = min(tile_size, w - left)

                tile = img_tensor[:, :, top:top+th, left:left+t_w].to(device)
                # OMDNet forward: (x, gt, mask, train) — for inference pass x 3 times
                _, deblured, _, gates, _, _, _ = net(tile, tile, tile, train=False)
                pred = deblured[0].cpu()
                tw_crop = tw[:, :, :th, :t_w]
                output[:, :, top:top+th, left:left+t_w] += pred * tw_crop
                weight[:, :, top:top+th, left:left+t_w] += tw_crop

                count += 1
                if count % 10 == 0 or count == total:
                    print(f'    tile {count}/{total}')

                del tile, deblured, gates, pred
                torch.cuda.empty_cache()

    return output / weight.clamp(min=1e-8)


def process_image(net, img_path, output_path, tile_size=512, overlap=128, device='cuda',
                  save_gate=False, gate_dir=None):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read {img_path}')
    h, w = img.shape[:2]
    print(f'  Input: {w}x{h}')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    output = tile_inference(net, img_tensor, tile_size, overlap, device)

    output = output[:, :, :h, :w]
    output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    output_bgr = (cv2.cvtColor(output, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)
    cv2.imwrite(output_path, output_bgr)
    print(f'  Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--weights', '-w', type=str, default=None)
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--images', type=str, default=None)
    args = parser.parse_args()

    if args.weights is None:
        args.weights = os.path.join(OMDNET_ROOT, 'checkpoints', 'test1', 'model_ckpt.ckpt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output, exist_ok=True)

    filter_ids = None
    if args.images:
        filter_ids = set(x.strip().zfill(2) for x in args.images.split(','))

    print(f'Loading OMDNet from {args.weights}')
    net = load_model(args.weights, num_res=20, device=device)
    print('Model loaded.')

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, '*.*')))
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    else:
        files = [args.input]

    for fpath in files:
        img_id = get_image_id(fpath)
        if filter_ids and img_id not in filter_ids:
            continue
        out_name = f'{img_id}.png' if img_id else os.path.splitext(os.path.basename(fpath))[0] + '.png'
        out_path = os.path.join(args.output, out_name)
        if os.path.exists(out_path):
            print(f'  [{img_id}] Already exists, skipping')
            continue
        print(f'Processing {os.path.basename(fpath)} ...')
        process_image(net, fpath, out_path, args.tile_size, args.overlap, device)
        torch.cuda.empty_cache()

    print('OMDNet done.')


if __name__ == '__main__':
    main()
