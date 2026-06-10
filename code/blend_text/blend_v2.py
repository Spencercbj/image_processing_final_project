"""
Text-aware blending v2:
  - Text regions (detected by EasyOCR) come from Restormer (pre-diffusion, faithful text)
  - Non-text regions come from the diffusion/ESRGAN result (better details)
"""
import argparse
import os
import glob
import re
import cv2
import numpy as np
import easyocr

PADDING = 30
BLUR_KERNEL = 51


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def build_text_mask(image, reader):
    results = reader.readtext(image)
    mask = np.zeros(image.shape[:2], dtype=np.float32)
    for (bbox, text, conf) in results:
        if conf < 0.1:
            continue
        pts = np.array(bbox, dtype=np.int32)
        x_min = max(0, pts[:, 0].min() - PADDING)
        y_min = max(0, pts[:, 1].min() - PADDING)
        x_max = min(image.shape[1], pts[:, 0].max() + PADDING)
        y_max = min(image.shape[0], pts[:, 1].max() + PADDING)
        mask[y_min:y_max, x_min:x_max] = 1.0
    mask = cv2.GaussianBlur(mask, (BLUR_KERNEL, BLUR_KERNEL), 0)
    return mask


def blend(text_img, detail_img, mask):
    mask_3ch = mask[:, :, np.newaxis]
    return (text_img * mask_3ch + detail_img * (1 - mask_3ch)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restormer_dir', type=str, required=True,
                        help='Dir with Restormer results (text source)')
    parser.add_argument('--diffusion_dir', type=str, required=True,
                        help='Dir with diffusion/ESRGAN results (detail source)')
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--images', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    filter_ids = None
    if args.images:
        filter_ids = set(args.images.split(','))

    print('Loading EasyOCR ...')
    reader = easyocr.Reader(['ch_tra', 'en'], gpu=True)

    diff_files = sorted(glob.glob(os.path.join(args.diffusion_dir, '*.png')))
    if not diff_files:
        print(f'No results in {args.diffusion_dir}')
        return

    for diff_path in diff_files:
        img_id = get_image_id(diff_path)
        if img_id is None:
            continue
        if filter_ids and img_id not in filter_ids:
            continue

        rest_candidates = glob.glob(os.path.join(args.restormer_dir, f'{img_id}*.*'))
        if not rest_candidates:
            print(f'[skip] {img_id}: no Restormer result')
            continue

        rest_path = rest_candidates[0]
        print(f'\n=== {img_id} ===')

        diff_img = cv2.imread(diff_path)
        rest_img = cv2.imread(rest_path)

        if rest_img.shape[:2] != diff_img.shape[:2]:
            rest_img = cv2.resize(rest_img, (diff_img.shape[1], diff_img.shape[0]),
                                  interpolation=cv2.INTER_LANCZOS4)

        mask = build_text_mask(diff_img, reader)
        text_count = int((mask > 0.5).sum())
        total = mask.shape[0] * mask.shape[1]
        print(f'  Text pixels: {text_count}/{total} ({100*text_count/total:.1f}%)')

        if text_count == 0:
            result = diff_img
            print('  No text detected, using diffusion result directly')
        else:
            result = blend(rest_img.astype(np.float32), diff_img.astype(np.float32), mask)

        out_path = os.path.join(args.output, f'{img_id}.png')
        cv2.imwrite(out_path, result)
        print(f'  Saved: {out_path}')

    print('\nBlending done.')


if __name__ == '__main__':
    main()
