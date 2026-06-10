"""
Frequency-domain blur analysis for all 15 nighttime photos.

For each image:
  1. Convert to grayscale, compute FFT power spectrum
  2. Analyze radial & angular power distribution to estimate:
     - Dominant blur direction (angle)
     - Blur kernel length (from dark band width in spectrum)
     - Blur severity (sharpness metric via high-freq energy ratio)
  3. Generate per-image diagnostic figure:
     - Original image (downscaled for display)
     - Log power spectrum with annotated blur direction
     - Angular power distribution (polar plot)
     - Radial power falloff curve
  4. Generate a summary report (text + comparison figure)
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


def compute_power_spectrum(gray):
    """Compute centred log power spectrum."""
    h, w = gray.shape
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude[magnitude == 0] = 1e-10
    log_ps = np.log10(magnitude)
    return log_ps, fshift


def angular_power(log_ps, num_bins=360):
    """Compute power as a function of angle (0-360 degrees)."""
    h, w = log_ps.shape
    cy, cx = h // 2, w // 2
    r_max = min(cy, cx) * 0.8

    y, x = np.mgrid[:h, :w]
    y = y - cy
    x = x - cx
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(y, x)) % 360

    mask = (r > 5) & (r < r_max)

    bin_edges = np.linspace(0, 360, num_bins + 1)
    power_by_angle = np.zeros(num_bins)
    for i in range(num_bins):
        angle_mask = mask & (theta >= bin_edges[i]) & (theta < bin_edges[i + 1])
        if angle_mask.any():
            power_by_angle[i] = np.mean(log_ps[angle_mask])

    return bin_edges[:-1] + 0.5, power_by_angle


def radial_power(log_ps, num_bins=200):
    """Compute power as a function of radial frequency."""
    h, w = log_ps.shape
    cy, cx = h // 2, w // 2
    r_max = min(cy, cx)

    y, x = np.mgrid[:h, :w]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)

    bin_edges = np.linspace(0, r_max, num_bins + 1)
    power_by_r = np.zeros(num_bins)
    for i in range(num_bins):
        rmask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if rmask.any():
            power_by_r[i] = np.mean(log_ps[rmask])

    freqs = (bin_edges[:-1] + bin_edges[1:]) / 2 / r_max
    return freqs, power_by_r


def estimate_blur(angles, ang_power):
    """
    Estimate blur direction and type from angular power distribution.

    Motion blur creates a DARK BAND perpendicular to blur direction in spectrum,
    meaning there's a power MINIMUM at the angle perpendicular to motion.
    Equivalently, power is concentrated ALONG the blur direction.

    Returns:
        blur_angle: estimated direction of motion blur (degrees)
        blur_type: 'horizontal', 'vertical', 'diagonal', 'radial/zoom', 'omnidirectional'
        anisotropy: ratio of max/min angular power (higher = more directional)
    """
    smooth_k = 15
    kernel = np.ones(smooth_k) / smooth_k
    ang_smooth = np.convolve(np.tile(ang_power, 3), kernel, mode='same')[len(ang_power):2*len(ang_power)]

    min_idx = np.argmin(ang_smooth)
    max_idx = np.argmax(ang_smooth)

    dark_band_angle = angles[min_idx]
    blur_angle = (dark_band_angle + 90) % 180

    power_range = ang_smooth.max() - ang_smooth.min()
    power_mean = ang_smooth.mean()
    anisotropy = power_range / (power_mean + 1e-10)

    if anisotropy < 0.03:
        blur_type = 'omnidirectional'
    elif anisotropy < 0.06:
        blur_type = 'mild_directional'
    else:
        a = blur_angle % 180
        if a < 15 or a > 165:
            blur_type = 'horizontal'
        elif 75 < a < 105:
            blur_type = 'vertical'
        elif 15 <= a <= 75:
            blur_type = 'diagonal_NE'
        else:
            blur_type = 'diagonal_NW'

    return blur_angle, blur_type, anisotropy, ang_smooth


def sharpness_metric(log_ps):
    """High-frequency energy ratio as sharpness indicator (0-1, higher=sharper)."""
    h, w = log_ps.shape
    cy, cx = h // 2, w // 2
    r_max = min(cy, cx)

    y, x = np.mgrid[:h, :w]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)

    total_energy = np.sum(10**log_ps)
    hf_mask = r > (r_max * 0.5)
    hf_energy = np.sum(10**(log_ps[hf_mask]))

    return hf_energy / (total_energy + 1e-10)


def estimate_kernel_length(ang_smooth, angles, anisotropy):
    """Rough estimate of blur kernel length from dark band width."""
    if anisotropy < 0.03:
        return 0

    threshold = ang_smooth.min() + 0.3 * (ang_smooth.max() - ang_smooth.min())
    below = ang_smooth < threshold
    width_deg = np.sum(below) * (360.0 / len(angles))
    return width_deg


def plot_single_image(img_path, output_dir):
    """Analyze one image and save diagnostic figure."""
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
    log_ps, _ = compute_power_spectrum(gray)
    angles, ang_power = angular_power(log_ps, num_bins=360)
    freqs, rad_power = radial_power(log_ps, num_bins=200)
    blur_angle, blur_type, anisotropy, ang_smooth = estimate_blur(angles, ang_power)
    sharpness = sharpness_metric(log_ps)
    dark_band_width = estimate_kernel_length(ang_smooth, angles, anisotropy)

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0:2])
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    ax1.set_title(f'{img_id}: {basename}\n{w_orig}x{h_orig}', fontsize=11)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2:4])
    ps_display = log_ps.copy()
    vmin, vmax = np.percentile(ps_display, [2, 98])
    ax2.imshow(ps_display, cmap='inferno', vmin=vmin, vmax=vmax)
    cy, cx = log_ps.shape[0] // 2, log_ps.shape[1] // 2
    r_line = min(cy, cx) * 0.4
    if anisotropy >= 0.03:
        rad = math.radians(blur_angle)
        dx, dy = r_line * math.cos(rad), r_line * math.sin(rad)
        ax2.annotate('', xy=(cx + dx, cy + dy), xytext=(cx - dx, cy - dy),
                     arrowprops=dict(arrowstyle='<->', color='cyan', lw=2))
        ax2.text(cx + dx * 1.1, cy + dy * 1.1, f'blur dir: {blur_angle:.0f}°',
                 color='cyan', fontsize=10, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    ax2.set_title(f'Log Power Spectrum\nType: {blur_type} | Anisotropy: {anisotropy:.3f}', fontsize=11)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 0:2], projection='polar')
    theta_rad = np.radians(angles)
    ax3.plot(theta_rad, ang_smooth, 'b-', linewidth=1.5, label='smoothed')
    ax3.plot(theta_rad, ang_power, 'b-', alpha=0.2, linewidth=0.5, label='raw')
    if anisotropy >= 0.03:
        min_angle_rad = np.radians(angles[np.argmin(ang_smooth)])
        ax3.axvline(min_angle_rad, color='red', linewidth=2, linestyle='--', label='dark band')
        ax3.axvline(min_angle_rad + np.pi, color='red', linewidth=2, linestyle='--')
    ax3.set_title('Angular Power Distribution', fontsize=11, pad=15)
    ax3.legend(loc='upper right', fontsize=8)

    ax4 = fig.add_subplot(gs[1, 2:4])
    ax4.plot(freqs, rad_power, 'b-', linewidth=1.5)
    ax4.set_xlabel('Normalized Frequency (0=DC, 1=Nyquist)')
    ax4.set_ylabel('Log Power')
    ax4.set_title('Radial Power Falloff', fontsize=11)
    ax4.grid(True, alpha=0.3)

    info_text = (
        f'Blur direction: {blur_angle:.0f}°\n'
        f'Blur type: {blur_type}\n'
        f'Anisotropy: {anisotropy:.4f}\n'
        f'Dark band width: {dark_band_width:.1f}°\n'
        f'Sharpness (HF ratio): {sharpness:.4f}\n'
        f'Resolution: {w_orig}x{h_orig}'
    )
    ax4.text(0.98, 0.98, info_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontfamily='monospace')

    out_path = os.path.join(output_dir, f'{img_id}_analysis.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')

    return {
        'id': img_id,
        'filename': os.path.basename(img_path),
        'resolution': f'{w_orig}x{h_orig}',
        'blur_angle': round(blur_angle, 1),
        'blur_type': blur_type,
        'anisotropy': round(anisotropy, 4),
        'dark_band_width': round(dark_band_width, 1),
        'sharpness': round(sharpness, 6),
    }


def plot_summary(results, output_dir):
    """Generate summary comparison figure."""
    n = len(results)
    fig, axes = plt.subplots(3, 5, figsize=(25, 16))
    fig.suptitle('Blur Analysis Summary — All 15 Images', fontsize=16, fontweight='bold')

    for idx, r in enumerate(results):
        row, col = idx // 5, idx % 5
        ax = axes[row][col]

        img_path = os.path.join(PHOTOS_DIR, r['filename'])
        img = cv2.imread(img_path)
        img_small = cv2.resize(img, (320, 213), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        color = {
            'horizontal': '#FF4444',
            'vertical': '#4444FF',
            'diagonal_NE': '#FF8800',
            'diagonal_NW': '#FF8800',
            'omnidirectional': '#888888',
            'mild_directional': '#AAAAAA',
        }.get(r['blur_type'], '#FFFFFF')

        label = (f"{r['id']}: {r['blur_type']}\n"
                 f"angle={r['blur_angle']:.0f}° aniso={r['anisotropy']:.3f}\n"
                 f"sharp={r['sharpness']:.4f}")
        ax.set_title(label, fontsize=8, color=color, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'summary.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Summary saved: {out_path}')


def suggest_method(r):
    """Suggest deblurring strategy based on analysis."""
    btype = r['blur_type']
    aniso = r['anisotropy']
    sharp = r['sharpness']

    suggestions = []

    if sharp > 0.01:
        suggestions.append('Mild blur — light processing only')
    elif sharp > 0.005:
        suggestions.append('Moderate blur')
    else:
        suggestions.append('Severe blur')

    if btype in ('horizontal', 'vertical', 'diagonal_NE', 'diagonal_NW'):
        angle = r['blur_angle']
        suggestions.append(f'Directional blur at {angle:.0f}° → Wiener/RL deconvolution with estimated kernel')
        suggestions.append('+ Restormer for residual cleanup')
    elif btype == 'mild_directional':
        suggestions.append('Weak directional component → Restormer should handle this')
    else:
        suggestions.append('No clear direction → Restormer (neural deblur) is best choice')

    if sharp < 0.003:
        suggestions.append('Very low sharpness — consider aggressive deblurring settings')

    return suggestions


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(PHOTOS_DIR, '*.*')))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    print(f'Analyzing {len(files)} images from {PHOTOS_DIR}\n')

    results = []
    for fpath in files:
        img_id = get_image_id(fpath)
        if img_id is None:
            continue
        print(f'[{img_id}] {os.path.basename(fpath)}')
        r = plot_single_image(fpath, OUTPUT_DIR)
        results.append(r)

    results.sort(key=lambda x: x['id'])

    plot_summary(results, OUTPUT_DIR)

    report_lines = []
    report_lines.append('# Frequency Analysis Report')
    report_lines.append('')
    report_lines.append('| ID | Resolution | Blur Type | Angle | Anisotropy | Sharpness | Dark Band Width |')
    report_lines.append('|----|-----------|-----------|-------|------------|-----------|-----------------|')
    for r in results:
        report_lines.append(
            f"| {r['id']} | {r['resolution']} | {r['blur_type']} | {r['blur_angle']:.0f}° | "
            f"{r['anisotropy']:.4f} | {r['sharpness']:.6f} | {r['dark_band_width']:.1f}° |"
        )

    report_lines.append('')
    report_lines.append('## Per-Image Recommendations')
    report_lines.append('')
    for r in results:
        suggestions = suggest_method(r)
        report_lines.append(f"### Image {r['id']} ({r['blur_type']}, sharpness={r['sharpness']:.6f})")
        for s in suggestions:
            report_lines.append(f'- {s}')
        report_lines.append('')

    report_lines.append('## Blur Type Groups')
    report_lines.append('')
    groups = {}
    for r in results:
        groups.setdefault(r['blur_type'], []).append(r['id'])
    for btype, ids in sorted(groups.items()):
        report_lines.append(f'- **{btype}**: {", ".join(ids)}')

    report_path = os.path.join(OUTPUT_DIR, 'report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f'\nReport saved: {report_path}')

    json_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'JSON saved: {json_path}')


if __name__ == '__main__':
    main()
