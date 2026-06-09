#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Resize images before/after DiffBIR.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare = subparsers.add_parser("prepare", help="Downscale large inputs for DiffBIR.")
    prepare.add_argument("--input", required=True, help="Input image folder.")
    prepare.add_argument("--output", required=True, help="Output folder for resized images.")
    prepare.add_argument("--max-side", type=int, default=2048, help="Resize longest side to this size.")

    restore = subparsers.add_parser("restore", help="Resize DiffBIR outputs back to reference size.")
    restore.add_argument("--diffbir", required=True, help="DiffBIR output folder.")
    restore.add_argument("--reference", required=True, help="Reference folder with target image sizes.")
    restore.add_argument("--output", required=True, help="Output folder for restored-size images.")

    return parser.parse_args()


def list_images(folder):
    folder = Path(folder)
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def resize_to_max_side(image, max_side):
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_side:
        return image.copy()
    scale = max_side / long_side
    size = (round(width * scale), round(height * scale))
    return image.resize(size, Image.Resampling.LANCZOS)


def prepare(input_dir, output_dir, max_side):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")

    print(f"input: {input_dir}")
    print(f"output: {output_dir}")
    print(f"max_side: {max_side}")
    for path in tqdm(image_paths, desc="prepare for DiffBIR"):
        with Image.open(path) as image:
            image = image.convert("RGB")
            resized = resize_to_max_side(image, max_side)
            resized.save(output_dir / path.name)
            print(f"{path.name}: {image.size[0]}x{image.size[1]} -> {resized.size[0]}x{resized.size[1]}")


def restore(diffbir_dir, reference_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(diffbir_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {diffbir_dir}")

    print(f"diffbir: {diffbir_dir}")
    print(f"reference: {reference_dir}")
    print(f"output: {output_dir}")
    for path in tqdm(image_paths, desc="restore DiffBIR size"):
        reference_path = reference_dir / path.name
        if not reference_path.is_file():
            print(f"skip {path.name}: missing reference image")
            continue
        with Image.open(path) as image, Image.open(reference_path) as reference:
            image = image.convert("RGB")
            target_size = reference.size
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            image.save(output_dir / path.name)
            print(f"{path.name}: -> {target_size[0]}x{target_size[1]}")


def main():
    args = parse_args()
    if args.mode == "prepare":
        prepare(Path(args.input), Path(args.output), args.max_side)
    elif args.mode == "restore":
        restore(Path(args.diffbir), Path(args.reference), Path(args.output))


if __name__ == "__main__":
    main()
