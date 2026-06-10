"""
Image 09 LOG-polar deconvolution (the key fix vs the earlier linear-polar attempt).

Zoom blur: a point at radius r is smeared radially by Δr ∝ r.
In log-polar (ρ = log r), that becomes a UNIFORM horizontal shift Δρ = const,
so a single fixed 1D horizontal PSF deconvolves the whole frame.

Pipeline (CPU): estimate zoom center -> warpPolar(LOG) -> per-channel 1D
Richardson-Lucy horizontal deconv (several kernel lengths) -> inverse warpPolar.
Outputs at 1440 for visual inspection; GPU cleanup + ESRGAN done separately on the best.
"""
import os
import numpy as np
import cv2
from PIL import Image
from skimage.restoration import richardson_lucy

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
SRC = os.path.join(BASE_DIR, 'photos', '09_White_Truck_Zoom_Blur_Rain.jpg')
OUT = os.path.join(BASE_DIR, 'results', 'pipeline_09_logpolar')
os.makedirs(OUT, exist_ok=True)

KERNELS = [9, 15, 23]   # horizontal PSF lengths in log-polar space
RL_ITERS = 25


def resize_1440(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    sf = 1440 / max(w, h)
    img = img.resize((int(w * sf), int(h * sf)), Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def estimate_center(img_bgr):
    """Zoom center = sharpest region (least blurred). Heavy smoothing for robustness."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    sharp = cv2.boxFilter(lap, -1, (81, 81))
    sharp = cv2.GaussianBlur(sharp, (0, 0), 31)
    _, _, _, maxloc = cv2.minMaxLoc(sharp)
    return maxloc  # (x, y)


def logpolar_deconv(img_bgr, center, klen, iters=RL_ITERS):
    h, w = img_bgr.shape[:2]
    cx, cy = center
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    maxR = max(np.hypot(cx - x, cy - y) for x, y in corners)

    pw = int(maxR)
    ph = min(int(2 * np.pi * maxR), 2880)
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG
    lp = cv2.warpPolar(img_bgr, (pw, ph), (cx, cy), maxR, flags)

    # horizontal 1D PSF (radius axis is horizontal)
    psf = np.ones((1, klen), np.float32) / klen
    lp_f = lp.astype(np.float32) / 255.0
    dec = np.zeros_like(lp_f)
    for c in range(3):
        dec[:, :, c] = richardson_lucy(lp_f[:, :, c], psf, num_iter=iters, clip=True)
    dec = np.clip(dec * 255, 0, 255).astype(np.uint8)

    inv = cv2.warpPolar(dec, (w, h), (cx, cy), maxR,
                        flags + cv2.WARP_INVERSE_MAP)
    return inv


def main():
    img = resize_1440(SRC)
    h, w = img.shape[:2]
    cv2.imwrite(os.path.join(OUT, '00_input1440.png'), img)

    auto = estimate_center(img)
    print(f'Image {w}x{h}; auto center = {auto}')
    centers = {
        'auto': auto,
        'geo': (w // 2, h // 2),
        'cab': (int(w * 0.52), int(h * 0.45)),
    }

    for cname, ctr in centers.items():
        # mark center on a copy for reference
        vis = img.copy(); cv2.circle(vis, ctr, 10, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT, f'01_center_{cname}.png'), vis)
        for k in KERNELS:
            res = logpolar_deconv(img, ctr, k)
            cv2.imwrite(os.path.join(OUT, f'deconv_{cname}_k{k}.png'), res)
            print(f'  {cname} k{k} done')

    print(f'Done. Variants in {OUT}')


if __name__ == '__main__':
    main()
