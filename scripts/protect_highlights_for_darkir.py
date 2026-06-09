#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Protect bright regions before and after DarkIR.")
    parser.add_argument("--mode", required=True, choices=["prepare", "composite"], help="prepare input for DarkIR or composite DarkIR output.")
    parser.add_argument("--original", default=str(ROOT / "img"), help="Original image folder.")
    parser.add_argument("--darkir", default=None, help="DarkIR result folder. Required in composite mode.")
    parser.add_argument("--output", required=True, help="Output image folder.")
    parser.add_argument("--threshold", type=float, default=0.82, help="Luminance threshold where highlight protection begins.")
    parser.add_argument("--softness", type=float, default=0.12, help="Soft transition width above threshold.")
    parser.add_argument("--highlight-scale", type=float, default=0.75, help="Brightness scale for highlights in prepare mode.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based index in sorted original input order to start from.")
    parser.add_argument("--limit", type=int, default=0, help="Only process N images. 0 means all selected images.")
    parser.add_argument("--dry-run", action="store_true", help="List planned operations without writing images.")
    return parser.parse_args()


def list_images(input_dir):
    input_dir = Path(input_dir)
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def select_images(image_paths, start_index, limit):
    if start_index < 1:
        raise ValueError("--start-index is 1-based and must be >= 1")
    indexed_paths = list(enumerate(image_paths, start=1))[start_index - 1 :]
    if limit > 0:
        indexed_paths = indexed_paths[:limit]
    return indexed_paths


def find_matching_result(result_dir, original_path):
    direct = result_dir / original_path.name
    if direct.is_file():
        return direct

    matches = sorted(result_dir.glob(f"{original_path.stem}.*"))
    matches = [path for path in matches if path.suffix.lower() in IMAGE_EXTENSIONS]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No matching result found for {original_path.name} in {result_dir}")


def image_to_float(image_path):
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0


def float_to_image(array):
    array = np.clip(array, 0.0, 1.0)
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def highlight_mask(original, threshold, softness):
    luminance = (
        0.2126 * original[:, :, 0]
        + 0.7152 * original[:, :, 1]
        + 0.0722 * original[:, :, 2]
    )
    softness = max(softness, 1e-6)
    mask = np.clip((luminance - threshold) / softness, 0.0, 1.0)
    return smoothstep(mask)[:, :, None]


def smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


def prepare_image(original, mask, highlight_scale):
    scale = 1.0 - mask * (1.0 - highlight_scale)
    return original * scale


def composite_image(original, darkir, mask):
    if original.shape != darkir.shape:
        darkir_image = float_to_image(darkir).resize(
            (original.shape[1], original.shape[0]), Image.Resampling.LANCZOS
        )
        darkir = np.asarray(darkir_image, dtype=np.float32) / 255.0
    return mask * original + (1.0 - mask) * darkir


def main():
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    if args.softness <= 0:
        raise ValueError("--softness must be > 0")
    if not 0 <= args.highlight_scale <= 1:
        raise ValueError("--highlight-scale must be between 0 and 1")
    if args.mode == "composite" and not args.darkir:
        raise ValueError("--darkir is required in composite mode")

    original_dir = Path(args.original)
    darkir_dir = Path(args.darkir) if args.darkir else None
    output_dir = Path(args.output)

    original_images = list_images(original_dir)
    indexed_images = select_images(original_images, args.start_index, args.limit)

    print(f"mode: {args.mode}")
    print(f"original: {original_dir}")
    if darkir_dir:
        print(f"darkir: {darkir_dir}")
    print(f"output: {output_dir}")
    print(f"threshold: {args.threshold}")
    print(f"softness: {args.softness}")
    print(f"highlight_scale: {args.highlight_scale}")
    print(f"images: {len(indexed_images)} / {len(original_images)}")
    print(f"start_index: {args.start_index}")

    if not indexed_images:
        raise SystemExit("No images found.")

    for image_index, original_path in tqdm(indexed_images, desc=f"Highlight {args.mode}"):
        output_path = output_dir / original_path.name
        darkir_path = find_matching_result(darkir_dir, original_path) if darkir_dir else None

        if args.dry_run:
            original_size = Image.open(original_path).size
            if darkir_path:
                darkir_size = Image.open(darkir_path).size
                print(f"{image_index:02d}. {original_path.name}: original {original_size}, darkir {darkir_size} -> {output_path}")
            else:
                print(f"{image_index:02d}. {original_path.name}: original {original_size} -> {output_path}")
            continue

        original = image_to_float(original_path)
        mask = highlight_mask(original, args.threshold, args.softness)

        if args.mode == "prepare":
            result = prepare_image(original, mask, args.highlight_scale)
        else:
            darkir = image_to_float(darkir_path)
            result = composite_image(original, darkir, mask)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        float_to_image(result).save(output_path)

    if not args.dry_run:
        print(f"Saved {args.mode} results to {output_dir}")


if __name__ == "__main__":
    main()
