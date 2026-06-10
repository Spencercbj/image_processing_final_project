import argparse
import os
import sys
import glob
import re
import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms

CIDNET_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'HVI-CIDNet')
sys.path.insert(0, CIDNET_ROOT)
from net.CIDNet import CIDNet


def load_model(weights_path, device='cuda'):
    net = CIDNet().to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()
    net.trans.gated2 = True
    net.trans.alpha = 1.0
    return net


def _make_tile_weight(tile_h, tile_w, overlap, device='cpu'):
    """Create a 2D cosine blending weight that ramps smoothly from 0 at edges to 1 in the centre."""
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
    tile_weight = _make_tile_weight(tile_size, tile_size, overlap, device='cpu')

    with torch.no_grad():
        for i in range(num_h):
            for j in range(num_w):
                top = min(i * stride, h - tile_size)
                left = min(j * stride, w - tile_size)
                top = max(top, 0)
                left = max(left, 0)
                th = min(tile_size, h - top)
                tw = min(tile_size, w - left)
                tile = img_tensor[:, :, top:top+th, left:left+tw].to(device)
                pred = net(tile).cpu()
                tw_crop = tile_weight[:, :, :th, :tw]
                output[:, :, top:top+th, left:left+tw] += pred * tw_crop
                weight[:, :, top:top+th, left:left+tw] += tw_crop
    return output / weight.clamp(min=1e-8)


def process_image(net, img_path, output_path, tile_size=512, overlap=64, device='cuda'):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read {img_path}')
    h_orig, w_orig = img.shape[:2]
    print(f'  Input: {w_orig}x{h_orig}')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

    factor = 8
    pad_h = (factor - h_orig % factor) % factor
    pad_w = (factor - w_orig % factor) % factor
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    if h_orig * w_orig <= 1024 * 1024:
        with torch.no_grad():
            output = net(img_tensor.to(device)).cpu()
    else:
        output = tile_inference(net, img_tensor, tile_size, overlap, device)

    output = output[:, :, :h_orig, :w_orig]
    output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    output_bgr = (cv2.cvtColor(output, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)
    cv2.imwrite(output_path, output_bgr)
    print(f'  Saved: {output_path}')


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--weights', '-w', type=str, default=None)
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--images', type=str, default=None,
                        help='Comma-separated image IDs to process, e.g. "01,08,12"')
    args = parser.parse_args()

    if args.weights is None:
        args.weights = os.path.join(CIDNET_ROOT, 'weights', 'LOLv2_syn', 'w_perc.pth')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output, exist_ok=True)

    filter_ids = None
    if args.images:
        filter_ids = set(args.images.split(','))

    print(f'Loading HVI-CIDNet from {args.weights}')
    net = load_model(args.weights, device)

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
        print(f'Processing {os.path.basename(fpath)} ...')
        process_image(net, fpath, out_path, args.tile_size, args.overlap, device)
        torch.cuda.empty_cache()

    print('HVI-CIDNet done.')


if __name__ == '__main__':
    main()
