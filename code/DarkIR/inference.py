"""
DarkIR inference — CVPR 2025 all-in-one low-light restoration.
Handles deblur + denoise + illumination enhancement in a single model.
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

DARKIR_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'DarkIR')
sys.path.insert(0, DARKIR_ROOT)
from archs.DarkIR import DarkIR


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def load_model(weights_path, width=32, device='cuda'):
    net = DarkIR(
        img_channel=3, width=width,
        middle_blk_num_enc=2, middle_blk_num_dec=2,
        enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1],
        dilations=[1, 4, 9], extra_depth_wise=True,
    )
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    state = ckpt.get('params', ckpt.get('model_state_dict', ckpt))
    cleaned = {k.replace('module.', ''): v for k, v in state.items()}
    net.load_state_dict(cleaned, strict=True)
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


def tile_inference(net, img_tensor, tile_size=384, overlap=128, device='cuda'):
    _, _, h, w = img_tensor.shape
    stride = tile_size - overlap
    num_h = max(1, math.ceil((h - overlap) / stride))
    num_w = max(1, math.ceil((w - overlap) / stride))

    output = torch.zeros_like(img_tensor, device='cpu')
    weight = torch.zeros((1, 1, h, w), device='cpu')
    tw = _make_tile_weight(tile_size, tile_size, overlap, device='cpu')

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
                pred = net(tile, side_loss=False)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                pred = pred.cpu()
                tw_crop = tw[:, :, :th, :t_w]
                output[:, :, top:top+th, left:left+t_w] += pred * tw_crop
                weight[:, :, top:top+th, left:left+t_w] += tw_crop

    return output / weight.clamp(min=1e-8)


def process_image(net, img_path, output_path, tile_size=384, overlap=128, device='cuda'):
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

    if h * w <= 512 * 512:
        with torch.no_grad():
            output = net(img_tensor.to(device), side_loss=False)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = output.cpu()
    else:
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
    parser.add_argument('--tile_size', type=int, default=384)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--images', type=str, default=None)
    args = parser.parse_args()

    if args.weights is None:
        args.weights = os.path.join(DARKIR_ROOT, 'models', 'DarkIR_384.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output, exist_ok=True)

    filter_ids = None
    if args.images:
        filter_ids = set(x.strip().zfill(2) for x in args.images.split(','))

    print(f'Loading DarkIR from {args.weights}')
    net = load_model(args.weights, width=32, device=device)
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

    print('DarkIR done.')


if __name__ == '__main__':
    main()
