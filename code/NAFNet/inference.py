import argparse
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math

NAFNET_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'NAFNet')
sys.path.insert(0, NAFNET_ROOT)
from basicsr.models.archs.NAFNet_arch import NAFNet


def load_model(weights_path, width=64, device='cuda'):
    net = NAFNet(
        img_channel=3, width=width,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1],
    )
    state = torch.load(weights_path, map_location='cpu')
    if 'params' in state:
        state = state['params']
    net.load_state_dict(state, strict=True)
    net.eval().to(device)
    return net


def tile_inference(net, img_tensor, tile_size=512, overlap=64, device='cuda'):
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
                pred = net(tile).cpu()

                output[:, :, top:top+tile_size, left:left+tile_size] += pred
                weight[:, :, top:top+tile_size, left:left+tile_size] += 1.0

    return output / weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--weights', '-w', type=str, default=None)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=64)
    args = parser.parse_args()

    if args.weights is None:
        args.weights = os.path.join(
            NAFNET_ROOT, 'experiments', 'pretrained_models',
            'NAFNet-REDS-width64.pth')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Loading model from {args.weights} ...')
    net = load_model(args.weights, width=args.width, device=device)

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f'Cannot read {args.input}')
    print(f'Input: {args.input} ({img.shape[1]}x{img.shape[0]})')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

    _, _, h, w = img_tensor.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
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

    cv2.imwrite(args.output, output_bgr)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
