"""
Image 03 post-brightening experiment.
The v18 deblur is already fine; the only problem is it's too dark.
Brighten the FINAL result (noise already cleaned), preserving warm bulbs.

Variants:
  a) CLAHE on L channel (clipLimit 1.5)
  b) Gamma 0.8
  c) Gamma 0.7
  d) Luminance-mask shadow lift (protect highlights/bulbs) + gentle gamma
Also copies v16 DarkIR 03 final for comparison.
"""
import os
import cv2
import numpy as np
import shutil

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
SRC = os.path.join(BASE_DIR, 'results', 'pipeline_v18', 'step6_realesrgan', '03.png')
V16 = os.path.join(BASE_DIR, 'results', 'pipeline_v16_darkir', 'step5_realesrgan', '03.png')
OUT = os.path.join(BASE_DIR, 'results', 'pipeline_03_brighten')
os.makedirs(OUT, exist_ok=True)


def clahe_L(img, clip=1.5, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l2 = cl.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def gamma(img, g=0.8):
    inv = 1.0 / g
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, lut)


def shadow_lift(img, g=0.75, highlight_protect=200):
    """Lift shadows via gamma but blend back highlights so bulbs don't blow out."""
    bright = gamma(img, g)
    # luminance of original
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # weight: 1 in shadows (use brightened), 0 in highlights (keep original)
    # smooth transition around highlight_protect
    w = np.clip((highlight_protect - lum) / 80.0, 0, 1)  # 0..1
    w = cv2.GaussianBlur(w, (0, 0), 5)[..., None]
    out = bright.astype(np.float32) * w + img.astype(np.float32) * (1 - w)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    img = cv2.imread(SRC)
    print(f'Source: {img.shape[1]}x{img.shape[0]}')

    cv2.imwrite(os.path.join(OUT, '03_a_clahe15.png'), clahe_L(img, 1.5))
    cv2.imwrite(os.path.join(OUT, '03_b_gamma08.png'), gamma(img, 0.8))
    cv2.imwrite(os.path.join(OUT, '03_c_gamma07.png'), gamma(img, 0.7))
    cv2.imwrite(os.path.join(OUT, '03_d_shadowlift.png'), shadow_lift(img, 0.72, 205))
    # combo: shadow lift then mild CLAHE for local contrast
    cv2.imwrite(os.path.join(OUT, '03_e_lift_clahe.png'),
                clahe_L(shadow_lift(img, 0.75, 205), 1.2))

    if os.path.exists(V16):
        shutil.copy2(V16, os.path.join(OUT, '03_z_v16_darkir.png'))
        print('copied v16 darkir for comparison')

    print(f'Done. Variants in {OUT}')


if __name__ == '__main__':
    main()
