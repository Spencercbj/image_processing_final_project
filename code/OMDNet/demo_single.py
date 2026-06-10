"""
Run OMDNet on a single image using the exact same logic as demo.py
(no tiling, whole image at once, with optional resize).
"""
import sys
import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import functional as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'OMDNet'))
from models.deblur_model import OMDNet
from utils.general import move_to_cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--weights', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', '..', 'OMDNet', 'checkpoints', 'test1', 'model_ckpt.ckpt'))
    parser.add_argument('--scale', type=int, default=1,
                        help='Downscale factor before processing (1=no resize, 2=half, 3=third)')
    parser.add_argument('--gate', action='store_true',
                        help='Also save the gate map')
    args = parser.parse_args()

    # Load model
    print(f'Loading OMDNet...')
    net = OMDNet(num_res=20)
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt['state_dict']['model'], strict=True)
    net.cuda()
    net.eval()
    del ckpt
    torch.cuda.empty_cache()
    print('Model loaded.')

    # Load image (exactly like demo.py)
    img_blur = Image.open(args.input)
    orig_w, orig_h = img_blur.size
    print(f'Original: {orig_w}x{orig_h}')

    if args.scale > 1:
        new_w = int(orig_w // args.scale)
        new_h = int(orig_h // args.scale)
        img_blur = img_blur.resize((new_w, new_h), Image.LANCZOS)
        print(f'Resized to: {new_w}x{new_h}')

    img_blur = np.uint8(np.array(img_blur))
    img_tensor = tf.to_tensor(img_blur)
    img_tensor.unsqueeze_(0)

    # Pad to multiple of 8 (exactly like demo.py)
    multiple_width = 8
    if img_tensor.shape[2] % multiple_width != 0:
        l_pad = multiple_width - img_tensor.shape[2] % multiple_width
        img_tensor = F.pad(img_tensor, (0, 0, 0, l_pad), mode='reflect')
    if img_tensor.shape[3] % multiple_width != 0:
        l_pad = multiple_width - img_tensor.shape[3] % multiple_width
        img_tensor = F.pad(img_tensor, (0, l_pad, 0, 0), mode='reflect')

    print(f'Tensor shape: {img_tensor.shape}')

    # Run model (exactly like demo.py)
    img_tensor = move_to_cuda(img_tensor)
    with torch.no_grad():
        _, deblured, _, gates, _, _, _ = net(img_tensor, img_tensor, img_tensor, train=False)

    # Save result (exactly like demo.py's plot_img)
    out_img = deblured[0][0].data.cpu().numpy()
    out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(np.transpose(out_img, (1, 2, 0)))
    out_img = out_img.convert("RGB")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out_img.save(args.output)
    print(f'Saved: {args.output}')

    if args.gate:
        gate_img = gates[0][0].data.cpu().numpy()
        gate_img = np.clip(gate_img * 255, 0, 255).astype(np.uint8)
        gate_img = Image.fromarray(np.transpose(gate_img, (1, 2, 0)))
        gate_path = args.output.replace('.png', '_gate.png')
        gate_img.save(gate_path)
        print(f'Gate saved: {gate_path}')

    print('Done!')


if __name__ == '__main__':
    main()
