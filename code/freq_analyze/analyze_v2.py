"""
Blur analysis v2 — multiple methods to handle nighttime images.

Methods:
  1. Patch-based FFT: avoid bright sources by selecting darker mid-tone patches
  2. Gradient histogram: Sobel Gx/Gy ratio reveals blur direction
  3. Cepstrum: detect periodic zeros from motion blur kernel
  4. Laplacian variance: robust sharpness metric
  5. Autocorrelation elongation: shape reveals blur direction + length
"""
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
from matplotlib.gridspec import GridSpec

PHOTOS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'photos'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Freq_analyze'))


def get_image_id(filename):
    m = re.match(r'^(\d+)', os.path.basename(filename))
    return m.group(1) if m else None


# ── Method 1: Patch-based FFT ──────────────────────────────────────────────

def select_patches(gray, patch_size=256, n_patches=12):
    """Select patches that avoid saturated bright sources."""
    h, w = gray.shape
    candidates = []
    stride = patch_size // 2
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = gray[y:y+patch_size, x:x+patch_size]
            mean_val = patch.mean()
            bright_ratio = np.sum(patch > 240) / patch.size
            std_val = patch.std()
            if bright_ratio < 0.05 and mean_val > 20 and std_val > 10:
                candidates.append((y, x, std_val))

    candidates.sort(key=lambda c: -c[2])
    return [(c[0], c[1]) for c in candidates[:n_patches]]


def patch_angular_power(gray, patches, patch_size=256, num_bins=180):
    """Aggregate angular power from multiple patches."""
    agg = np.zeros(num_bins)
    count = 0
    for (py, px) in patches:
        patch = gray[py:py+patch_size, px:px+patch_size].astype(np.float64)
        patch = patch - patch.mean()
        window = np.outer(np.hanning(patch_size), np.hanning(patch_size))
        patch = patch * window

        f = np.fft.fft2(patch)
        fshift = np.fft.fftshift(f)
        ps = np.log10(np.abs(fshift) + 1e-10)

        cy, cx = patch_size // 2, patch_size // 2
        y, x = np.mgrid[:patch_size, :patch_size]
        r = np.sqrt((y - cy)**2 + (x - cx)**2)
        theta = np.degrees(np.arctan2(y - cy, x - cx)) % 180

        mask = (r > 3) & (r < patch_size // 2 * 0.8)
        bin_edges = np.linspace(0, 180, num_bins + 1)
        for i in range(num_bins):
            amask = mask & (theta >= bin_edges[i]) & (theta < bin_edges[i+1])
            if amask.any():
                agg[i] += np.mean(ps[amask])
                count += 1

    if count > 0:
        agg /= len(patches)
    angles = (bin_edges[:-1] + bin_edges[1:]) / 2
    return angles, agg


# ── Method 2: Gradient histogram ──────────────────────────────────────────

def gradient_analysis(gray):
    """
    Analyze gradient magnitude by direction.
    Motion blur suppresses gradients along the blur direction.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)
    angle = np.degrees(np.arctan2(gy, gx)) % 180

    strong = mag > np.percentile(mag, 70)

    num_bins = 180
    bin_edges = np.linspace(0, 180, num_bins + 1)
    grad_by_angle = np.zeros(num_bins)
    for i in range(num_bins):
        amask = strong & (angle >= bin_edges[i]) & (angle < bin_edges[i+1])
        if amask.any():
            grad_by_angle[i] = np.sum(mag[amask])

    angles = (bin_edges[:-1] + bin_edges[1:]) / 2

    kernel = np.ones(9) / 9
    grad_smooth = np.convolve(np.tile(grad_by_angle, 3), kernel, 'same')[num_bins:2*num_bins]

    min_idx = np.argmin(grad_smooth)
    blur_dir_grad = angles[min_idx]

    grad_range = grad_smooth.max() - grad_smooth.min()
    grad_aniso = grad_range / (grad_smooth.mean() + 1e-10)

    return angles, grad_smooth, blur_dir_grad, grad_aniso


# ── Method 3: Cepstrum ────────────────────────────────────────────────────

def cepstrum_analysis(gray, patch_size=512):
    """
    Power cepstrum to detect blur kernel size.
    Motion blur creates periodic peaks in cepstrum along blur direction.
    """
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - patch_size // 2)
    x0 = max(0, cx - patch_size // 2)
    patch = gray[y0:y0+patch_size, x0:x0+patch_size].astype(np.float64)
    patch = patch - patch.mean()
    window = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
    patch = patch * window

    f = np.fft.fft2(patch)
    log_ps = np.log(np.abs(f)**2 + 1e-10)
    cepstrum = np.abs(np.fft.ifft2(log_ps))**2
    cepstrum = np.fft.fftshift(cepstrum)

    return cepstrum


# ── Method 4: Laplacian variance ──────────────────────────────────────────

def laplacian_sharpness(gray):
    """Variance of Laplacian — lower = more blurry."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


# ── Method 5: Autocorrelation ─────────────────────────────────────────────

def autocorrelation_analysis(gray, patch_size=512):
    """
    Autocorrelation of a central patch.
    Motion blur elongates the autocorrelation peak in the blur direction.
    """
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - patch_size // 2)
    x0 = max(0, cx - patch_size // 2)
    patch = gray[y0:y0+patch_size, x0:x0+patch_size].astype(np.float64)
    patch = patch - patch.mean()

    f = np.fft.fft2(patch)
    acorr = np.fft.ifft2(f * np.conj(f)).real
    acorr = np.fft.fftshift(acorr)
    acorr = acorr / acorr.max()

    threshold = 0.5
    mask = acorr > threshold

    ys, xs = np.where(mask)
    if len(ys) < 5:
        return acorr, 0, 0, 1.0

    cy_a, cx_a = patch_size // 2, patch_size // 2
    ys_c = ys - cy_a
    xs_c = xs - cx_a

    cov = np.cov(xs_c, ys_c)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)

    major_idx = np.argmax(eigvals)
    major_vec = eigvecs[:, major_idx]
    elongation_angle = np.degrees(np.arctan2(major_vec[1], major_vec[0])) % 180
    aspect_ratio = np.sqrt(eigvals.max() / (eigvals.min() + 1e-10))

    return acorr, elongation_angle, aspect_ratio, eigvals


# ── Main analysis per image ───────────────────────────────────────────────

def analyze_image(img_path, output_dir):
    img_id = get_image_id(img_path)
    basename = os.path.splitext(os.path.basename(img_path))[0]

    img = cv2.imread(img_path)
    h_orig, w_orig = img.shape[:2]

    max_dim = 1024
    scale = min(max_dim / max(h_orig, w_orig), 1.0)
    if scale < 1.0:
        img_small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()

    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    gh, gw = gray.shape

    # Method 1: Patch-based FFT
    patches = select_patches(gray, patch_size=256, n_patches=12)
    if len(patches) >= 3:
        patch_angles, patch_power = patch_angular_power(gray, patches, patch_size=256)
        kernel = np.ones(9) / 9
        patch_smooth = np.convolve(np.tile(patch_power, 3), kernel, 'same')[180:360]
        patch_min_idx = np.argmin(patch_smooth)
        patch_blur_dir = patch_angles[patch_min_idx]
        patch_range = patch_smooth.max() - patch_smooth.min()
        patch_aniso = patch_range / (patch_smooth.mean() + 1e-10)
    else:
        patch_angles = np.linspace(0, 180, 180)
        patch_smooth = np.zeros(180)
        patch_blur_dir = 0
        patch_aniso = 0

    # Method 2: Gradient
    grad_angles, grad_smooth, grad_blur_dir, grad_aniso = gradient_analysis(gray)

    # Method 3: Cepstrum
    cep_size = min(512, gh, gw)
    cepstrum = cepstrum_analysis(gray, cep_size)

    # Method 4: Laplacian sharpness
    lap_var = laplacian_sharpness(gray)

    # Method 5: Autocorrelation
    ac_size = min(512, gh, gw)
    acorr, ac_angle, ac_aspect, ac_eigvals = autocorrelation_analysis(gray, ac_size)

    # ── Consensus ──
    votes = []
    if patch_aniso > 0.05 and len(patches) >= 3:
        votes.append(('patch_fft', patch_blur_dir, patch_aniso))
    if grad_aniso > 0.3:
        votes.append(('gradient', grad_blur_dir, grad_aniso))
    if ac_aspect > 1.5:
        votes.append(('autocorr', ac_angle, ac_aspect))

    if len(votes) >= 2:
        dirs = [v[1] for v in votes]
        weights = [v[2] for v in votes]
        sin_sum = sum(w * math.sin(2 * math.radians(d)) for d, w in zip(dirs, weights))
        cos_sum = sum(w * math.cos(2 * math.radians(d)) for d, w in zip(dirs, weights))
        consensus_angle = (math.degrees(math.atan2(sin_sum, cos_sum)) / 2) % 180
        confidence = 'high' if len(votes) == 3 else 'medium'
    elif len(votes) == 1:
        consensus_angle = votes[0][1]
        confidence = 'low'
    else:
        consensus_angle = 0
        confidence = 'none'

    # Classify
    a = consensus_angle % 180
    if confidence == 'none':
        blur_type = 'unknown/complex'
    elif a < 20 or a > 160:
        blur_type = 'horizontal'
    elif 70 < a < 110:
        blur_type = 'vertical'
    else:
        blur_type = f'diagonal_{a:.0f}deg'

    # Blur severity from Laplacian
    if lap_var < 50:
        severity = 'severe'
    elif lap_var < 200:
        severity = 'moderate'
    elif lap_var < 500:
        severity = 'mild'
    else:
        severity = 'sharp'

    # ── Plot ──
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Row 0: Original + patch locations | Gradient histogram
    ax_img = fig.add_subplot(gs[0, 0:2])
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    ax_img.imshow(img_rgb)
    for (py, px) in patches[:12]:
        rect = plt.Rectangle((px, py), 256, 256, linewidth=1, edgecolor='lime', facecolor='none', alpha=0.6)
        ax_img.add_patch(rect)
    ax_img.set_title(f'{img_id}: {basename}\n{w_orig}x{h_orig} | Laplacian var={lap_var:.1f} ({severity})',
                     fontsize=10)
    ax_img.axis('off')

    ax_grad = fig.add_subplot(gs[0, 2:4])
    ax_grad.fill_between(grad_angles, grad_smooth, alpha=0.3, color='blue')
    ax_grad.plot(grad_angles, grad_smooth, 'b-', linewidth=1.5)
    if grad_aniso > 0.3:
        ax_grad.axvline(grad_blur_dir, color='red', linewidth=2, linestyle='--',
                        label=f'min gradient dir: {grad_blur_dir:.0f}° (blur dir)')
    ax_grad.set_xlabel('Gradient Direction (°)')
    ax_grad.set_ylabel('Total Gradient Magnitude')
    ax_grad.set_title(f'Method 2: Gradient Histogram\nanisotropy={grad_aniso:.3f}', fontsize=10)
    ax_grad.legend(fontsize=8)
    ax_grad.grid(True, alpha=0.3)

    # Row 1: Patch FFT angular | Autocorrelation
    ax_patch = fig.add_subplot(gs[1, 0:2])
    if len(patches) >= 3:
        ax_patch.fill_between(patch_angles, patch_smooth, alpha=0.3, color='green')
        ax_patch.plot(patch_angles, patch_smooth, 'g-', linewidth=1.5)
        if patch_aniso > 0.05:
            ax_patch.axvline(patch_blur_dir, color='red', linewidth=2, linestyle='--',
                             label=f'dark band: {patch_blur_dir:.0f}° → blur: {(patch_blur_dir+90)%180:.0f}°')
        ax_patch.legend(fontsize=8)
    ax_patch.set_xlabel('Angle in Spectrum (°)')
    ax_patch.set_ylabel('Mean Log Power')
    ax_patch.set_title(f'Method 1: Patch-based FFT ({len(patches)} patches)\nanisotropy={patch_aniso:.3f}',
                       fontsize=10)
    ax_patch.grid(True, alpha=0.3)

    ax_ac = fig.add_subplot(gs[1, 2:4])
    ac_center = ac_size // 2
    crop = 60
    ac_crop = acorr[ac_center-crop:ac_center+crop, ac_center-crop:ac_center+crop]
    ax_ac.imshow(ac_crop, cmap='hot', vmin=0, vmax=0.3)
    if ac_aspect > 1.5:
        rad = math.radians(ac_angle)
        length = crop * 0.7
        dx, dy = length * math.cos(rad), length * math.sin(rad)
        ax_ac.annotate('', xy=(crop+dx, crop+dy), xytext=(crop-dx, crop-dy),
                       arrowprops=dict(arrowstyle='<->', color='cyan', lw=2))
    ax_ac.set_title(f'Method 5: Autocorrelation\nangle={ac_angle:.0f}° aspect_ratio={ac_aspect:.1f}',
                    fontsize=10)
    ax_ac.axis('off')

    # Row 2: Cepstrum | Summary
    ax_cep = fig.add_subplot(gs[2, 0:2])
    cep_center = cep_size // 2
    cep_crop_sz = 80
    cep_crop = cepstrum[cep_center-cep_crop_sz:cep_center+cep_crop_sz,
                        cep_center-cep_crop_sz:cep_center+cep_crop_sz]
    cep_display = np.log10(cep_crop + 1e-20)
    vmin, vmax = np.percentile(cep_display, [5, 95])
    ax_cep.imshow(cep_display, cmap='magma', vmin=vmin, vmax=vmax)
    ax_cep.set_title('Method 3: Power Cepstrum (central region)', fontsize=10)
    ax_cep.axis('off')

    ax_sum = fig.add_subplot(gs[2, 2:4])
    ax_sum.axis('off')

    method_results = []
    if len(patches) >= 3 and patch_aniso > 0.05:
        method_results.append(f'Patch FFT:     dark band at {patch_blur_dir:.0f}° → blur ~{(patch_blur_dir+90)%180:.0f}°  (aniso={patch_aniso:.3f})')
    else:
        method_results.append(f'Patch FFT:     weak/no directional signal (aniso={patch_aniso:.3f})')

    if grad_aniso > 0.3:
        method_results.append(f'Gradient:      min at {grad_blur_dir:.0f}° → blur ~{grad_blur_dir:.0f}°  (aniso={grad_aniso:.3f})')
    else:
        method_results.append(f'Gradient:      weak directional signal (aniso={grad_aniso:.3f})')

    method_results.append(f'Cepstrum:      (see plot — look for peaks away from center)')

    if ac_aspect > 1.5:
        method_results.append(f'Autocorr:      elongated at {ac_angle:.0f}°, ratio={ac_aspect:.1f}')
    else:
        method_results.append(f'Autocorr:      near-circular, ratio={ac_aspect:.1f} (no clear direction)')

    method_results.append(f'Laplacian var: {lap_var:.1f} → {severity}')
    method_results.append('')
    method_results.append(f'─── CONSENSUS ───')
    method_results.append(f'Blur direction:  {consensus_angle:.0f}°  ({blur_type})')
    method_results.append(f'Confidence:      {confidence} ({len(votes)}/3 methods agree)')
    method_results.append(f'Severity:        {severity} (lap_var={lap_var:.1f})')

    if blur_type == 'unknown/complex':
        method_results.append(f'Recommendation:  Restormer (neural, no kernel needed)')
    elif confidence in ('high', 'medium'):
        method_results.append(f'Recommendation:  Wiener deconv @ {consensus_angle:.0f}° + Restormer')
    else:
        method_results.append(f'Recommendation:  Restormer primary, optional Wiener @ {consensus_angle:.0f}°')

    summary_text = '\n'.join(method_results)
    ax_sum.text(0.05, 0.95, summary_text, transform=ax_sum.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax_sum.set_title('Analysis Summary', fontsize=10)

    out_path = os.path.join(output_dir, f'{img_id}_analysis_v2.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')

    return {
        'id': img_id,
        'filename': os.path.basename(img_path),
        'resolution': f'{w_orig}x{h_orig}',
        'laplacian_var': round(lap_var, 1),
        'severity': severity,
        'patch_fft_blur_dir': round((patch_blur_dir + 90) % 180, 1) if patch_aniso > 0.05 else None,
        'patch_fft_aniso': round(patch_aniso, 4),
        'gradient_blur_dir': round(grad_blur_dir, 1) if grad_aniso > 0.3 else None,
        'gradient_aniso': round(grad_aniso, 4),
        'autocorr_angle': round(ac_angle, 1) if ac_aspect > 1.5 else None,
        'autocorr_aspect': round(ac_aspect, 2),
        'consensus_angle': round(consensus_angle, 1) if confidence != 'none' else None,
        'consensus_type': blur_type,
        'confidence': confidence,
        'n_patches': len(patches),
    }


def plot_summary_v2(results, output_dir):
    """Summary comparison: severity + direction overview."""
    fig, axes = plt.subplots(3, 5, figsize=(28, 18))
    fig.suptitle('Blur Analysis v2 — All 15 Images', fontsize=16, fontweight='bold')

    for idx, r in enumerate(results):
        row, col = idx // 5, idx % 5
        ax = axes[row][col]

        img_path = os.path.join(PHOTOS_DIR, r['filename'])
        img = cv2.imread(img_path)
        img_small = cv2.resize(img, (320, 213), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        sev_color = {
            'severe': '#FF2222',
            'moderate': '#FF8800',
            'mild': '#88AA00',
            'sharp': '#00AA00',
        }.get(r['severity'], '#FFFFFF')

        conf_marker = {'high': '***', 'medium': '**', 'low': '*', 'none': ''}[r['confidence']]

        angle_str = f"{r['consensus_angle']:.0f}°" if r['consensus_angle'] is not None else 'N/A'
        label = (f"{r['id']}: {r['severity'].upper()} (lap={r['laplacian_var']:.0f})\n"
                 f"{r['consensus_type']} @ {angle_str} {conf_marker}")
        ax.set_title(label, fontsize=8, color=sev_color, fontweight='bold')

        if r['consensus_angle'] is not None and r['confidence'] != 'none':
            rad = math.radians(r['consensus_angle'])
            cx, cy = 160, 106
            length = 50
            dx, dy = length * math.cos(rad), length * math.sin(rad)
            ax.annotate('', xy=(cx+dx, cy+dy), xytext=(cx-dx, cy-dy),
                        arrowprops=dict(arrowstyle='<->', color='cyan', lw=2.5))

        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'summary_v2.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Summary saved: {out_path}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(PHOTOS_DIR, '*.*')))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    print(f'Analyzing {len(files)} images (v2 multi-method)\n')

    results = []
    for fpath in files:
        img_id = get_image_id(fpath)
        if img_id is None:
            continue
        print(f'[{img_id}] {os.path.basename(fpath)}')
        r = analyze_image(fpath, OUTPUT_DIR)
        results.append(r)

    results.sort(key=lambda x: x['id'])
    plot_summary_v2(results, OUTPUT_DIR)

    # ── Report ──
    lines = []
    lines.append('# Frequency Analysis Report v2 (Multi-Method)')
    lines.append('')
    lines.append('## Methods Used')
    lines.append('1. **Patch-based FFT**: Selects mid-tone patches (avoids bright sources), analyzes angular power')
    lines.append('2. **Gradient histogram**: Sobel gradient magnitude by direction — blur suppresses gradients along blur axis')
    lines.append('3. **Cepstrum**: Detects periodic zeros from motion blur kernel')
    lines.append('4. **Laplacian variance**: Robust sharpness metric (lower = more blurry)')
    lines.append('5. **Autocorrelation**: Shape elongation reveals blur direction')
    lines.append('')
    lines.append('## Results')
    lines.append('')
    lines.append('| ID | Resolution | Severity | Lap.Var | Blur Type | Angle | Confidence | Recommendation |')
    lines.append('|----|-----------|----------|---------|-----------|-------|------------|----------------|')
    for r in results:
        angle_str = f"{r['consensus_angle']:.0f}°" if r['consensus_angle'] is not None else 'N/A'
        if r['consensus_type'] == 'unknown/complex':
            rec = 'Restormer only'
        elif r['confidence'] in ('high', 'medium'):
            rec = f"Wiener @ {angle_str} + Restormer"
        else:
            rec = f"Restormer (optional Wiener @ {angle_str})"
        lines.append(
            f"| {r['id']} | {r['resolution']} | {r['severity']} | {r['laplacian_var']:.0f} | "
            f"{r['consensus_type']} | {angle_str} | {r['confidence']} | {rec} |"
        )

    lines.append('')
    lines.append('## Severity Groups')
    lines.append('')
    groups = {}
    for r in results:
        groups.setdefault(r['severity'], []).append(r['id'])
    for sev in ['severe', 'moderate', 'mild', 'sharp']:
        if sev in groups:
            lines.append(f'- **{sev}**: {", ".join(groups[sev])}')

    lines.append('')
    lines.append('## Per-Method Detail')
    lines.append('')
    for r in results:
        lines.append(f"### Image {r['id']}")
        lines.append(f"- Patch FFT: aniso={r['patch_fft_aniso']:.4f}" +
                     (f", blur~{r['patch_fft_blur_dir']:.0f}°" if r['patch_fft_blur_dir'] is not None else ', no direction'))
        lines.append(f"- Gradient: aniso={r['gradient_aniso']:.4f}" +
                     (f", blur~{r['gradient_blur_dir']:.0f}°" if r['gradient_blur_dir'] is not None else ', no direction'))
        lines.append(f"- Autocorr: aspect={r['autocorr_aspect']:.2f}" +
                     (f", angle~{r['autocorr_angle']:.0f}°" if r['autocorr_angle'] is not None else ', near-circular'))
        lines.append(f"- Laplacian: {r['laplacian_var']:.1f} ({r['severity']})")
        lines.append('')

    report_path = os.path.join(OUTPUT_DIR, 'report_v2.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nReport: {report_path}')

    json_path = os.path.join(OUTPUT_DIR, 'results_v2.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'JSON: {json_path}')


if __name__ == '__main__':
    main()
