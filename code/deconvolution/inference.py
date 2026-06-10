"""
Blind motion deconvolution using Wiener filter.
Estimates dominant blur direction via Fourier spectrum analysis,
then applies Wiener deconvolution.
"""
import argparse
import os
import cv2
import numpy as np
from scipy.signal import fftconvolve


def estimate_motion_kernel(gray, ksize=31):
    """Estimate motion blur angle from Fourier spectrum."""
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    r = min(cy, cx) // 2

    best_angle = 0
    best_var = 0
    for angle_deg in range(0, 180, 5):
        angle_rad = np.deg2rad(angle_deg)
        vals = []
        for t in np.linspace(-r, r, 100):
            y = int(cy + t * np.sin(angle_rad))
            x = int(cx + t * np.cos(angle_rad))
            if 0 <= y < h and 0 <= x < w:
                vals.append(magnitude[y, x])
        v = np.var(vals) if vals else 0
        if v > best_var:
            best_var = v
            best_angle = angle_deg

    kernel = np.zeros((ksize, ksize), dtype=np.float64)
    center = ksize // 2
    angle_rad = np.deg2rad(best_angle)
    for i in range(ksize):
        offset = i - center
        y = int(center + offset * np.sin(angle_rad))
        x = int(center + offset * np.cos(angle_rad))
        if 0 <= y < ksize and 0 <= x < ksize:
            kernel[y, x] = 1.0
    kernel /= kernel.sum()
    return kernel, best_angle


def wiener_deconvolve(img, kernel, snr=0.01):
    """Per-channel Wiener deconvolution."""
    kernel_padded = np.zeros_like(img[:, :, 0], dtype=np.float64)
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel
    # center the kernel
    kernel_padded = np.roll(kernel_padded, -(kh // 2), axis=0)
    kernel_padded = np.roll(kernel_padded, -(kw // 2), axis=1)

    K = np.fft.fft2(kernel_padded)
    K_conj = np.conj(K)
    K_abs2 = np.abs(K) ** 2

    result = np.zeros_like(img, dtype=np.float64)
    for c in range(img.shape[2]):
        F = np.fft.fft2(img[:, :, c].astype(np.float64))
        W = K_conj / (K_abs2 + snr)
        result[:, :, c] = np.real(np.fft.ifft2(F * W))

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--kernel_size', type=int, default=21)
    parser.add_argument('--snr', type=float, default=0.005)
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f'Cannot read {args.input}')

    print(f'Input: {args.input} ({img.shape[1]}x{img.shape[0]})')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel, angle = estimate_motion_kernel(gray, ksize=args.kernel_size)
    print(f'Estimated motion blur angle: {angle} degrees, kernel size: {args.kernel_size}')

    result = wiener_deconvolve(img, kernel, snr=args.snr)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, result)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
