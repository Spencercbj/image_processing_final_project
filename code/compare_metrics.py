"""Compare results across methods using no-reference metrics."""
import os
import glob
import cv2
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..')
PHOTOS = os.path.join(BASE, 'photos')
RESULTS = os.path.join(BASE, 'results')

METHODS = ['NAFNet', 'DarkIR', 'Restormer', 'pipeline_DarkIR_Restormer']


def laplacian_variance(img_gray):
    """Sharpness metric: higher = sharper."""
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def edge_energy(img_gray):
    """Sum of Sobel edge magnitudes, normalized by pixel count."""
    sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sx**2 + sy**2).mean()


def brightness(img_bgr):
    return img_bgr.astype(float).mean()


def pixel_diff(img1, img2):
    return np.abs(img1.astype(float) - img2.astype(float)).mean()


def main():
    images = sorted(glob.glob(os.path.join(PHOTOS, '*.*')))
    names = [os.path.splitext(os.path.basename(p))[0] for p in images]

    # Collect per-image metrics
    all_metrics = {}
    for method in ['original'] + METHODS:
        all_metrics[method] = {'sharp': [], 'edge': [], 'bright': [], 'diff': []}

    for img_path, name in zip(images, names):
        orig = cv2.imread(img_path)
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        orig_sharp = laplacian_variance(orig_gray)
        orig_edge = edge_energy(orig_gray)
        orig_bright = brightness(orig)

        all_metrics['original']['sharp'].append(orig_sharp)
        all_metrics['original']['edge'].append(orig_edge)
        all_metrics['original']['bright'].append(orig_bright)
        all_metrics['original']['diff'].append(0.0)

        for method in METHODS:
            result_path = os.path.join(RESULTS, method, f'{name}.png')
            if not os.path.exists(result_path):
                for k in all_metrics[method]:
                    all_metrics[method][k].append(float('nan'))
                continue
            res = cv2.imread(result_path)
            res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            all_metrics[method]['sharp'].append(laplacian_variance(res_gray))
            all_metrics[method]['edge'].append(edge_energy(res_gray))
            all_metrics[method]['bright'].append(brightness(res))
            all_metrics[method]['diff'].append(pixel_diff(orig, res))

    # Print summary table
    print('=' * 90)
    print(f'{"Method":<28} {"Sharpness":>12} {"Edge Energy":>12} {"Brightness":>12} {"Diff from orig":>14}')
    print(f'{"(Laplacian var)":>40} {"(Sobel mean)":>12} {"(mean pixel)":>12} {"(mean |Δ|)":>14}')
    print('=' * 90)

    for method in ['original'] + METHODS:
        m = all_metrics[method]
        s = np.nanmean(m['sharp'])
        e = np.nanmean(m['edge'])
        b = np.nanmean(m['bright'])
        d = np.nanmean(m['diff'])
        label = method if method != 'pipeline_DarkIR_Restormer' else 'Pipeline (D+R)'
        print(f'{label:<28} {s:>12.1f} {e:>12.2f} {b:>12.1f} {d:>14.2f}')

    # Per-image sharpness improvement
    print('\n' + '=' * 100)
    print('Per-image sharpness (Laplacian variance) — higher is sharper')
    print('=' * 100)
    header = f'{"Image":<45}'
    for method in ['original'] + METHODS:
        label = method[:8] if method != 'pipeline_DarkIR_Restormer' else 'Pipe D+R'
        header += f' {label:>10}'
    print(header)
    print('-' * 100)

    for i, name in enumerate(names):
        short = name[:43]
        row = f'{short:<45}'
        orig_val = all_metrics['original']['sharp'][i]
        for method in ['original'] + METHODS:
            val = all_metrics[method]['sharp'][i]
            if method == 'original':
                row += f' {val:>10.1f}'
            else:
                pct = (val - orig_val) / orig_val * 100 if orig_val > 0 else 0
                row += f' {val:>7.1f}{pct:>+4.0f}%'

        print(row)


if __name__ == '__main__':
    main()
