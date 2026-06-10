import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math

RESTORMER_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'Restormer')
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'restormer_arch',
    os.path.join(RESTORMER_ROOT, 'basicsr', 'models', 'archs', 'restormer_arch.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Restormer = _mod.Restormer


def load_model(weights_path, device='cuda'):
    net = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias', dual_pixel_task=False,
    )
    ckpt = torch.load(weights_path, map_location='cpu')
    state = ckpt.get('params', ckpt)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--weights', '-w', type=str, default=None)
    parser.add_argument('--tile_size', type=int, default=896)
    parser.add_argument('--overlap', type=int, default=64)
    args = parser.parse_args()

    if args.weights is None:
        args.weights = os.path.join(
            RESTORMER_ROOT, 'Motion_Deblurring', 'pretrained_models',
            'motion_deblurring.pth')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Loading model from {args.weights} ...')
    net = load_model(args.weights, device=device)

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f'Cannot read {args.input}')
    print(f'Input: {args.input} ({img.shape[1]}x{img.shape[0]})')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

    _, _, h, w = img_tensor.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    if h * w <= 1024 * 1024:
        print('Image small enough for direct inference.')
        with torch.no_grad():
            output = net(img_tensor.to(device)).cpu()
    else:
        print(f'Tiled inference: tile={args.tile_size}, overlap={args.overlap}')
        output = tile_inference(net, img_tensor, args.tile_size, args.overlap, device)

    output = output[:, :, :h, :w]
    output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    output_bgr = (cv2.cvtColor(output, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cv2.imwrite(args.output, output_bgr)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
