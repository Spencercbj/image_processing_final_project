"""
09 v2: diagnostic + refined-center log-polar deconv + center-crop fallback.
- Save the log-polar image itself: if zoom blur, it should look like HORIZONTAL
  streaks (uniform) -> deconv-able. If streaks are slanted, blur isn't pure zoom.
- Center-crop 60% fallback (the zoom centre region is naturally sharpest).
"""
import os
import numpy as np
import cv2
from skimage.restoration import richardson_lucy

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(BASE, 'results', 'pipeline_09_logpolar')
INP = os.path.join(OUT, '00_input1440.png')
V18 = os.path.join(BASE, 'results', 'pipeline_v18', 'step6_realesrgan', '09.png')
CENTER = (711, 595)


def main():
    img = cv2.imread(INP)
    h, w = img.shape[:2]
    cx, cy = CENTER
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    maxR = max(np.hypot(cx - x, cy - y) for x, y in corners)
    pw = int(maxR); ph = min(int(2 * np.pi * maxR), 2880)
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG

    lp = cv2.warpPolar(img, (pw, ph), (cx, cy), maxR, flags)
    cv2.imwrite(os.path.join(OUT, 'diag_logpolar.png'), lp)
    print(f'log-polar {lp.shape[1]}x{lp.shape[0]} saved (diagnostic)')

    for k in (11, 15):
        psf = np.ones((1, k), np.float32) / k
        lpf = lp.astype(np.float32) / 255.0
        dec = np.zeros_like(lpf)
        for c in range(3):
            dec[:, :, c] = richardson_lucy(lpf[:, :, c], psf, num_iter=25, clip=True)
        dec = np.clip(dec * 255, 0, 255).astype(np.uint8)
        inv = cv2.warpPolar(dec, (w, h), (cx, cy), maxR, flags + cv2.WARP_INVERSE_MAP)
        cv2.imwrite(os.path.join(OUT, f'deconv_refined_k{k}.png'), inv)
        print(f'refined k{k} done')

    # center-crop 60% fallback from the v18 final (already deblurred + ESRGAN)
    big = cv2.imread(V18)
    H, W = big.shape[:2]
    cw, ch = int(W * 0.60), int(H * 0.60)
    x0 = (W - cw) // 2; y0 = int(H * 0.30)  # bias down toward truck/road
    y0 = min(y0, H - ch)
    crop = big[y0:y0 + ch, x0:x0 + cw]
    cv2.imwrite(os.path.join(OUT, '09_centercrop.png'), crop)
    print(f'center crop {crop.shape[1]}x{crop.shape[0]} saved')


if __name__ == '__main__':
    main()
