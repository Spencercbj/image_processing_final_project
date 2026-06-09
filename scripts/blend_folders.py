#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Blend matching images from two folders.")
    parser.add_argument("--base", required=True, help="Base image folder. Alpha 0 keeps this.")
    parser.add_argument("--enhanced", required=True, help="Enhanced image folder. Alpha 1 keeps this.")
    parser.add_argument("--output", required=True, help="Output folder.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Enhanced image weight.")
    return parser.parse_args()


def list_images(folder):
    folder = Path(folder)
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def find_match(folder, image_path):
    direct = folder / image_path.name
    if direct.is_file():
        return direct

    matches = sorted(folder.glob(f"{image_path.stem}.*"))
    matches = [path for path in matches if path.suffix.lower() in IMAGE_EXTENSIONS]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No match found for {image_path.name} in {folder}")


def main():
    args = parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    base_dir = Path(args.base)
    enhanced_dir = Path(args.enhanced)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_images = list_images(base_dir)
    if not base_images:
        raise SystemExit(f"No images found in {base_dir}")

    print(f"base: {base_dir}")
    print(f"enhanced: {enhanced_dir}")
    print(f"output: {output_dir}")
    print(f"alpha: {args.alpha}")

    for base_path in tqdm(base_images, desc="blend folders"):
        enhanced_path = find_match(enhanced_dir, base_path)
        with Image.open(base_path) as base, Image.open(enhanced_path) as enhanced:
            base = base.convert("RGB")
            enhanced = enhanced.convert("RGB")
            if enhanced.size != base.size:
                enhanced = enhanced.resize(base.size, Image.Resampling.LANCZOS)
            blended = Image.blend(base, enhanced, args.alpha)
            blended.save(output_dir / base_path.name)

    print(f"Saved blended images to {output_dir}")


if __name__ == "__main__":
    main()
