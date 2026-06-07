#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Blend DarkIR results with original inputs before feeding them to EVSSM."
    )
    parser.add_argument("--original", default=str(ROOT / "img"), help="Original input image folder.")
    parser.add_argument("--darkir", required=True, help="DarkIR result folder.")
    parser.add_argument("--output", required=True, help="Output folder for blended EVSSM inputs.")
    parser.add_argument("--alpha", type=float, default=0.7, help="DarkIR weight. 0 uses original, 1 uses DarkIR.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based index in sorted original input order to start from.")
    parser.add_argument("--limit", type=int, default=0, help="Only process N images. 0 means all selected images.")
    parser.add_argument("--dry-run", action="store_true", help="List planned blends without writing images.")
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


def find_matching_darkir_result(darkir_dir, original_path):
    direct = darkir_dir / original_path.name
    if direct.is_file():
        return direct

    matches = sorted(darkir_dir.glob(f"{original_path.stem}.*"))
    matches = [path for path in matches if path.suffix.lower() in IMAGE_EXTENSIONS]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No DarkIR result found for {original_path.name} in {darkir_dir}")


def blend_images(original_path, darkir_path, output_path, alpha):
    original = Image.open(original_path).convert("RGB")
    enhanced = Image.open(darkir_path).convert("RGB")

    if original.size != enhanced.size:
        original = original.resize(enhanced.size, Image.Resampling.LANCZOS)

    blended = Image.blend(original, enhanced, alpha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(output_path)


def main():
    args = parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    original_dir = Path(args.original)
    darkir_dir = Path(args.darkir)
    output_dir = Path(args.output)

    original_images = list_images(original_dir)
    indexed_images = select_images(original_images, args.start_index, args.limit)

    print(f"original: {original_dir}")
    print(f"darkir: {darkir_dir}")
    print(f"output: {output_dir}")
    print(f"alpha: {args.alpha}")
    print(f"images: {len(indexed_images)} / {len(original_images)}")
    print(f"start_index: {args.start_index}")

    if not indexed_images:
        raise SystemExit("No images found.")

    for image_index, original_path in tqdm(indexed_images, desc="Blend DarkIR with original"):
        darkir_path = find_matching_darkir_result(darkir_dir, original_path)
        output_path = output_dir / original_path.name

        if args.dry_run:
            original_size = Image.open(original_path).size
            darkir_size = Image.open(darkir_path).size
            print(
                f"{image_index:02d}. {original_path.name}: "
                f"original {original_size}, darkir {darkir_size} -> {output_path}"
            )
            continue

        blend_images(original_path, darkir_path, output_path, args.alpha)

    if not args.dry_run:
        print(f"Saved blended inputs to {output_dir}")


if __name__ == "__main__":
    main()
