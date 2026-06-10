import argparse
import os
import sys
import glob
import re
import math
import torch
import torch.nn as nn
import numpy as np
import cv2

MISC_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'MISCFilter')
sys.path.insert(0, MISC_ROOT)
from models.MISCFilterNet import MISCKernelNet as MyNet
from models.layers import window_partitionx, window_reversex
import utils


def load_model(weights_path, device='cuda'):
    net = MyNet(inference=False)
    utils.load_checkpoint(net, weights_path)
    net = net.to(device)
    net.eval()
    return net


def windowed_inference(net, img_tensor, win_size=256, batch_size=16, device='cuda'):
    """Process image using MISCFilter's window partition, batching windows to avoid OOM."""
    _, _, H, W = img_tensor.shape
    input_re, batch_list = window_partitionx(img_tensor, win_size)
    total_windows = input_re.shape[0]

    outputs = []
    with torch.no_grad():
        for start in range(0, total_windows, batch_size):
            end = min(start + batch_size, total_windows)
            batch = input_re[start:end].to(device)
            out, _ = net(batch)
            out = out[0] if isinstance(out, (list, tuple)) else out
            outputs.append(out.cpu())
            torch.cuda.empty_cache()

    restored = torch.cat(outputs, dim=0)
    restored = window_reversex(restored, win_size, H, W, batch_list)
    return restored


def process_image(net, img_path, output_path, win_size=256, batch_size=16, device='cuda'):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read {img_path}')
    h_orig, w_orig = img.shape[:2]
    print(f'  Input: {w_orig}x{h_orig}')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

    restored = windowed_inference(net, img_tensor, win_size, batch_size, device)
    restored = restored.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    output_bgr = (cv2.cvtColor(restored, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)

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
    parser.add_argument('--win_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--images', type=str, default=None,
                        help='Comma-separated image IDs to process, e.g. "01,08,12"')
    args = parser.parse_args()

    if args.weights is None:
        args.weights = os.path.join(MISC_ROOT, 'pretrained_models', 'checkpoints', 'RealBlur_J.pth')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output, exist_ok=True)

    filter_ids = None
    if args.images:
        filter_ids = set(args.images.split(','))

    print(f'Loading MISCFilter from {args.weights}')
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
        process_image(net, fpath, out_path, args.win_size, args.batch_size, device)
        torch.cuda.empty_cache()

    print('MISCFilter done.')


if __name__ == '__main__':
    main()
