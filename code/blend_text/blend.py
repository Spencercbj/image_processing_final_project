"""
Text-aware blending: keep text regions from Restormer, use DiffBIR SDEdit for the rest.
EasyOCR detects text bounding boxes, builds a soft mask, then blends.
"""
import os
import sys
import glob
import re
import cv2
import numpy as np
import easyocr

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESTORMER_DIR = os.path.join(BASE, 'results', 'Restormer')
SDEDIT_DIR = os.path.join(BASE, 'results', 'pipeline_Restormer_DiffBIR_sdedit')
OUTPUT_DIR = os.path.join(BASE, 'results', 'pipeline_Restormer_DiffBIR_sdedit_blended')

PADDING = 30
BLUR_KERNEL = 51


def get_short_name(filename):
    m = re.match(r'^(\d+)', filename)
    return m.group(1) if m else None


def build_text_mask(image, reader):
    """Detect text regions and create a soft mask (1 = text area = use Restormer)."""
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


def blend(restormer_img, sdedit_img, mask):
    """mask=1 -> restormer, mask=0 -> sdedit"""
    mask_3ch = mask[:, :, np.newaxis]
    return (restormer_img * mask_3ch + sdedit_img * (1 - mask_3ch)).astype(np.uint8)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading EasyOCR (first run downloads models)...')
    reader = easyocr.Reader(['ch_tra', 'ja', 'en'], gpu=True)

    sdedit_files = sorted(glob.glob(os.path.join(SDEDIT_DIR, '*.png')))
    if not sdedit_files:
        print(f'No SDEdit results found in {SDEDIT_DIR}')
        return

    for sdedit_path in sdedit_files:
        fname = os.path.basename(sdedit_path)
        short = get_short_name(fname)
        if short is None:
            continue

        restormer_candidates = glob.glob(os.path.join(RESTORMER_DIR, f'{short}_*.png'))
        if not restormer_candidates:
            print(f'[skip] {short}: no matching Restormer result')
            continue

        restormer_path = restormer_candidates[0]
        out_path = os.path.join(OUTPUT_DIR, f'{short}.png')

        print(f'\n=== {short} ===')
        restormer_img = cv2.imread(restormer_path)
        sdedit_img = cv2.imread(sdedit_path)

        if restormer_img.shape != sdedit_img.shape:
            print(f'  Resizing Restormer {restormer_img.shape[:2]} -> {sdedit_img.shape[:2]}')
            restormer_img = cv2.resize(restormer_img,
                                       (sdedit_img.shape[1], sdedit_img.shape[0]),
                                       interpolation=cv2.INTER_LANCZOS4)

        print('  Detecting text regions...')
        mask = build_text_mask(restormer_img, reader)
        n_text_pixels = (mask > 0.5).sum()
        total_pixels = mask.shape[0] * mask.shape[1]
        pct = n_text_pixels / total_pixels * 100
        print(f'  Text area: {pct:.1f}% of image')

        result = blend(restormer_img, sdedit_img, mask)
        cv2.imwrite(out_path, result)
        print(f'  Saved {out_path}')

    print('\nDone! Blended results:')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f'  {f}')


if __name__ == '__main__':
    main()
