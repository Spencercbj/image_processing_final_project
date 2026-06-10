"""
Wiener / Richardson-Lucy deconvolution with per-image motion blur PSF.
Reads blur direction from Freq_analyze/results_v2.json,
estimates kernel length from autocorrelation FWHM.
"""
import argparse
import os
import sys
import glob
import re
import json
import math

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ANALYSIS_JSON = os.path.join(BASE_DIR, 'Freq_analyze', 'results_v2.json')


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


def load_analysis():
    with open(ANALYSIS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {r['id']: r for r in data}


# ── PSF creation ──────────────────────────────────────────────────────────

def create_motion_blur_psf(length, angle_deg):
    """Create a 1-pixel-wide motion blur PSF of given length and angle."""
    size = int(length * 2) | 1
    size = max(size, 3)
    psf = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    rad = np.radians(angle_deg)
    half = length / 2.0
    x1 = int(round(center + half * np.cos(rad)))
    y1 = int(round(center + half * np.sin(rad)))
    x2 = int(round(center - half * np.cos(rad)))
    y2 = int(round(center - half * np.sin(rad)))
    cv2.line(psf, (x1, y1), (x2, y2), 1.0, 1)
    total = psf.sum()
    if total > 0:
        psf /= total
    else:
        psf[center, center] = 1.0
    return psf


# ── Blur length estimation ────────────────────────────────────────────────

def _measure_fwhm_along(acorr, center, patch_size, angle_rad, max_d):
    """Measure FWHM of autocorrelation along a given direction."""
    profile = []
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    for d in range(-max_d, max_d + 1):
        x = int(round(center + d * cos_a))
        y = int(round(center + d * sin_a))
        if 0 <= y < patch_size and 0 <= x < patch_size:
            profile.append(acorr[y, x])
        else:
            profile.append(0)
    profile = np.array(profile)
    above = profile >= 0.5
    indices = np.where(above)[0]
    if len(indices) >= 2:
        return indices[-1] - indices[0]
    return 0


def estimate_blur_length(gray, angle_deg, orig_h, orig_w, max_analysis_dim=1024):
    """Estimate blur kernel length using differential autocorrelation FWHM.

    Measures FWHM along blur direction and perpendicular; the difference
    isolates the blur contribution from the image content autocorrelation.
    """
    h, w = gray.shape
    scale = min(max_analysis_dim / max(h, w), 1.0)
    if scale < 1.0:
        gray_s = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        gray_s = gray.copy()

    sh, sw = gray_s.shape
    patch_size = min(512, sh, sw)
    rad_along = np.radians(angle_deg)
    rad_perp = rad_along + np.pi / 2
    diff_values = []

    positions = [
        (sh // 4, sw // 4), (sh // 4, sw // 2), (sh // 4, 3 * sw // 4),
        (sh // 2, sw // 4), (sh // 2, sw // 2), (sh // 2, 3 * sw // 4),
        (3 * sh // 4, sw // 4), (3 * sh // 4, sw // 2), (3 * sh // 4, 3 * sw // 4),
    ]

    for (py, px) in positions:
        y0 = max(0, min(py - patch_size // 2, sh - patch_size))
        x0 = max(0, min(px - patch_size // 2, sw - patch_size))
        patch = gray_s[y0:y0 + patch_size, x0:x0 + patch_size].astype(np.float64)

        if patch.mean() < 10 or np.sum(patch > 240) / patch.size > 0.1:
            continue
        if patch.std() < 5:
            continue

        patch -= patch.mean()
        window = np.outer(np.hanning(patch_size), np.hanning(patch_size))
        patch *= window

        f = np.fft.fft2(patch)
        acorr = np.fft.ifft2(f * np.conj(f)).real
        acorr = np.fft.fftshift(acorr)
        peak = acorr.max()
        if peak > 0:
            acorr /= peak

        center = patch_size // 2
        max_d = min(100, patch_size // 4)

        fwhm_along = _measure_fwhm_along(acorr, center, patch_size, rad_along, max_d)
        fwhm_perp = _measure_fwhm_along(acorr, center, patch_size, rad_perp, max_d)

        excess = fwhm_along - fwhm_perp
        if excess > 1:
            diff_values.append(excess)

    orig_to_analysis = max(orig_h, orig_w) / max(sh, sw)

    if diff_values:
        median_diff = np.median(diff_values)
        blur_length_full = (median_diff / 2) * orig_to_analysis
    else:
        blur_length_full = max(orig_h, orig_w) / 400

    return int(np.clip(blur_length_full, 5, 25))


# ── Wiener deconvolution ──────────────────────────────────────────────────

def wiener_deconv(img, psf, K=0.01):
    """Frequency-domain Wiener deconvolution."""
    h, w = img.shape[:2]
    img_f = img.astype(np.float64) / 255.0

    ph, pw = psf.shape
    psf_pad = np.zeros((h, w), dtype=np.float64)
    psf_pad[:ph, :pw] = psf
    cy, cx = ph // 2, pw // 2
    psf_pad = np.roll(np.roll(psf_pad, -cy, axis=0), -cx, axis=1)

    PSF = np.fft.fft2(psf_pad)
    PSF_conj = np.conj(PSF)
    PSF_abs2 = np.abs(PSF) ** 2

    result = np.zeros_like(img_f)
    for c in range(img_f.shape[2] if img_f.ndim == 3 else 1):
        channel = img_f[:, :, c] if img_f.ndim == 3 else img_f
        IMG = np.fft.fft2(channel)
        RES = IMG * PSF_conj / (PSF_abs2 + K)
        result_c = np.fft.ifft2(RES).real
        if img_f.ndim == 3:
            result[:, :, c] = result_c
        else:
            result = result_c

    return np.clip(result * 255, 0, 255).astype(np.uint8)


# ── Richardson-Lucy deconvolution ─────────────────────────────────────────

def richardson_lucy(img, psf, iterations=15):
    """Iterative Richardson-Lucy deconvolution."""
    img_f = img.astype(np.float64) / 255.0
    img_f = np.clip(img_f, 1e-6, 1.0)
    psf_flip = psf[::-1, ::-1].copy()

    estimate = img_f.copy()
    for it in range(iterations):
        for c in range(img_f.shape[2]):
            est_blur = cv2.filter2D(estimate[:, :, c], -1, psf,
                                    borderType=cv2.BORDER_REFLECT)
            ratio = img_f[:, :, c] / (est_blur + 1e-10)
            correction = cv2.filter2D(ratio, -1, psf_flip,
                                      borderType=cv2.BORDER_REFLECT)
            estimate[:, :, c] *= correction
            estimate[:, :, c] = np.clip(estimate[:, :, c], 1e-6, 10.0)

    return np.clip(estimate * 255, 0, 255).astype(np.uint8)


# ── Debug visualization ──────────────────────────────────────────────────

def save_debug(img_before, img_after, psf, info, out_path):
    """Save a side-by-side comparison + PSF visualization."""
    h, w = img_before.shape[:2]
    crop_size = min(800, h // 2, w // 2)
    cy, cx = h // 2, w // 2
    y0 = cy - crop_size // 2
    x0 = cx - crop_size // 2

    crop_before = img_before[y0:y0 + crop_size, x0:x0 + crop_size]
    crop_after = img_after[y0:y0 + crop_size, x0:x0 + crop_size]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    axes[0].imshow(cv2.cvtColor(crop_before, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Before (center crop)', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(crop_after, cv2.COLOR_BGR2RGB))
    axes[1].set_title('After deconv (center crop)', fontsize=11)
    axes[1].axis('off')

    psf_display = psf / psf.max()
    axes[2].imshow(psf_display, cmap='hot', interpolation='nearest')
    label = (f"PSF: angle={info['angle']:.0f}°  length={info['length']}px\n"
             f"method={info['method']}  K={info.get('K', 'N/A')}")
    axes[2].set_title(label, fontsize=10)
    axes[2].axis('off')

    fig.suptitle(f"Image {info['id']} — {info['severity']} blur, {info['blur_type']}, "
                 f"confidence={info['confidence']}", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def process_single(img_path, output_path, analysis_entry, method='wiener',
                   K=0.01, rl_iters=15, debug_dir=None):
    """Process a single image with deconvolution."""
    img_id = analysis_entry['id']
    angle = analysis_entry.get('consensus_angle')
    confidence = analysis_entry.get('confidence', 'none')
    severity = analysis_entry.get('severity', 'unknown')

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read {img_path}')
    h, w = img.shape[:2]

    if angle is None or confidence == 'none':
        print(f'  [{img_id}] No clear blur direction — copying as-is')
        cv2.imwrite(output_path, img)
        return

    if severity == 'sharp':
        print(f'  [{img_id}] Image is sharp (lap={analysis_entry["laplacian_var"]}) — copying as-is')
        cv2.imwrite(output_path, img)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_length = estimate_blur_length(gray, angle, h, w)

    if severity == 'severe':
        K_actual = K * 0.5
        rl_iters_actual = rl_iters + 10
    elif severity == 'moderate':
        K_actual = K * 0.8
        rl_iters_actual = rl_iters + 5
    else:
        K_actual = K
        rl_iters_actual = rl_iters

    print(f'  [{img_id}] {severity} {analysis_entry["consensus_type"]} '
          f'angle={angle:.0f}° length={blur_length}px method={method} K={K_actual:.4f}')

    psf = create_motion_blur_psf(blur_length, angle)

    if method == 'wiener':
        result = wiener_deconv(img, psf, K=K_actual)
    elif method == 'rl':
        result = richardson_lucy(img, psf, iterations=rl_iters_actual)
    else:
        result = wiener_deconv(img, psf, K=K_actual)

    cv2.imwrite(output_path, result)
    print(f'  [{img_id}] Saved: {output_path}')

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        info = {
            'id': img_id, 'angle': angle, 'length': blur_length,
            'method': method, 'K': K_actual,
            'severity': severity,
            'blur_type': analysis_entry['consensus_type'],
            'confidence': confidence,
        }
        save_debug(img, result, psf, info,
                   os.path.join(debug_dir, f'{img_id}_debug.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--images', type=str, default=None)
    parser.add_argument('--method', type=str, default='wiener',
                        choices=['wiener', 'rl'])
    parser.add_argument('--K', type=float, default=0.01)
    parser.add_argument('--rl_iters', type=int, default=15)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    analysis = load_analysis()

    filter_ids = None
    if args.images:
        filter_ids = set(args.images.split(','))

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, '*.*')))
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    else:
        files = [args.input]

    debug_dir = os.path.join(args.output, 'debug') if args.debug else None

    for fpath in files:
        img_id = get_image_id(fpath)
        if img_id is None:
            continue
        if filter_ids and img_id not in filter_ids:
            continue
        if img_id not in analysis:
            print(f'  [{img_id}] No analysis data — skipping')
            continue

        out_name = f'{img_id}.png'
        out_path = os.path.join(args.output, out_name)
        process_single(fpath, out_path, analysis[img_id],
                       method=args.method, K=args.K, rl_iters=args.rl_iters,
                       debug_dir=debug_dir)

    print('Deconvolution done.')


if __name__ == '__main__':
    main()
