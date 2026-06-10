"""
Pipeline: DarkIR (low-light enhance) → Restormer (motion deblur)
"""
import argparse
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math

BASE = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(BASE, 'DarkIR'))
sys.path.insert(0, os.path.join(BASE, 'Restormer'))

from archs.DarkIR import DarkIR
from basicsr.models.archs.restormer_arch import Restormer


def load_darkir(weights_path, width=32, device='cuda'):
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


def load_restormer(weights_path, device='cuda'):
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


def tile_inference(net, img_tensor, tile_size, overlap, device, forward_fn):
    _, _, h, w = img_tensor.shape
    stride = tile_size - overlap

    num_h = max(1, math.ceil((h - overlap) / stride))
    num_w = max(1, math.ceil((w - overlap) / stride))

    output = torch.zeros_like(img_tensor, device='cpu')
    weight = torch.zeros((1, 1, h, w), device='cpu')

    with torch.no_grad():
        for i in range(num_h):
            for j in range(num_w):
                top = min(i * stride, h - tile_size)
                left = min(j * stride, w - tile_size)
                top = max(top, 0)
                left = max(left, 0)

                tile = img_tensor[:, :, top:top+tile_size, left:left+tile_size].to(device)
                pred = forward_fn(tile).cpu()

                output[:, :, top:top+tile_size, left:left+tile_size] += pred
                weight[:, :, top:top+tile_size, left:left+tile_size] += 1.0

    return output / weight


def run_stage(net, img_tensor, tile_size, overlap, pad_multiple, device, forward_fn):
    _, _, h, w = img_tensor.shape
    pad_h = (pad_multiple - h % pad_multiple) % pad_multiple
    pad_w = (pad_multiple - w % pad_multiple) % pad_multiple
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    if h * w <= 768 * 768:
        with torch.no_grad():
            output = forward_fn(img_tensor.to(device)).cpu()
    else:
        output = tile_inference(net, img_tensor, tile_size, overlap, device, forward_fn)

    return output[:, :, :h, :w].clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--darkir_weights', type=str, default=None)
    parser.add_argument('--restormer_weights', type=str, default=None)
    args = parser.parse_args()

    if args.darkir_weights is None:
        args.darkir_weights = os.path.join(BASE, 'DarkIR', 'models', 'DarkIR_384.pt')
    if args.restormer_weights is None:
        args.restormer_weights = os.path.join(
            BASE, 'Restormer', 'Motion_Deblurring', 'pretrained_models',
            'motion_deblurring.pth')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f'Cannot read {args.input}')
    print(f'Input: {args.input} ({img.shape[1]}x{img.shape[0]})')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

    # Stage 1: DarkIR (low-light enhancement)
    print('Stage 1: DarkIR (low-light enhance) ...')
    darkir = load_darkir(args.darkir_weights, device=device)
    enhanced = run_stage(
        darkir, img_tensor, tile_size=1280, overlap=64, pad_multiple=8,
        device=device, forward_fn=lambda x: darkir(x, side_loss=False))
    del darkir
    torch.cuda.empty_cache()

    # Stage 2: Restormer (motion deblurring)
    print('Stage 2: Restormer (motion deblur) ...')
    restormer = load_restormer(args.restormer_weights, device=device)
    deblurred = run_stage(
        restormer, enhanced, tile_size=896, overlap=64, pad_multiple=8,
        device=device, forward_fn=restormer)
    del restormer
    torch.cuda.empty_cache()

    output = deblurred.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    output_bgr = (cv2.cvtColor(output, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)

    cv2.imwrite(args.output, output_bgr)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
