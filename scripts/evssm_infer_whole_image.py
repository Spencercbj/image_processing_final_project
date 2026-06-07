import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import resize, to_pil_image, to_tensor
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
EVSSM_ROOT = ROOT / "methods" / "EVSSM"
sys.path.insert(0, str(EVSSM_ROOT))

from models.EVSSM import EVSSM  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        if "params" in checkpoint:
            return checkpoint["params"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def pad_to_multiple(image: torch.Tensor, multiple: int = 4) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = image.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, height, width
    return F.pad(image, (0, pad_w, 0, pad_h), mode="reflect"), height, width


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def select_images(image_paths: list[Path], start_index: int, limit: int) -> list[tuple[int, Path]]:
    if start_index < 1:
        raise ValueError("--start-index is 1-based and must be >= 1")

    indexed_paths = list(enumerate(image_paths, start=1))
    indexed_paths = indexed_paths[start_index - 1 :]
    if limit > 0:
        indexed_paths = indexed_paths[:limit]
    return indexed_paths


def resize_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_side:
        return image
    scale = max_side / long_side
    new_size = (round(height * scale), round(width * scale))
    return resize(image, new_size, antialias=True)


def autocast_dtype(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run whole-image EVSSM inference on a folder of small images.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to EVSSM checkpoint, e.g. net_g_GoPro.pth.")
    parser.add_argument("--input-dir", default=ROOT / "img", type=Path, help="Folder containing blurred input images.")
    parser.add_argument("--output-dir", default=ROOT / "results" / "EVSSM", type=Path, help="Folder for deblurred outputs.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based index in sorted input order to start from.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images. 0 means all images.")
    parser.add_argument("--max-side", type=int, default=0, help="Resize the longest side before inference. 0 keeps original size.")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"], help="Inference precision.")
    parser.add_argument("--dry-run", action="store_true", help="List planned inputs/outputs without running the model.")
    args = parser.parse_args()

    all_image_paths = iter_images(args.input_dir)
    indexed_image_paths = select_images(all_image_paths, args.start_index, args.limit)
    if not indexed_image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    print(f"input: {args.input_dir}")
    print(f"output: {args.output_dir}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"images: {len(indexed_image_paths)} / {len(all_image_paths)}")
    print(f"start_index: {args.start_index}")
    print("mode: whole image")
    print(f"max_side: {args.max_side if args.max_side > 0 else 'original'}")
    print(f"precision: {args.precision}")

    if args.dry_run:
        for image_index, image_path in indexed_image_paths:
            with Image.open(image_path) as image:
                resized = resize_to_max_side(image, args.max_side)
                print(f"{image_index:02d}. {image_path.name}: {image.size} -> {resized.size} -> {args.output_dir / image_path.name}")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run the GPU test in setup.md first, or pass --device cpu for debugging.")
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = EVSSM().to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device), strict=True)
    model.eval()

    amp_dtype = autocast_dtype(args.precision)
    use_amp = device.type == "cuda" and amp_dtype is not None

    with torch.inference_mode():
        for image_index, image_path in tqdm(indexed_image_paths, desc="EVSSM inference"):
            image = Image.open(image_path).convert("RGB")
            image = resize_to_max_side(image, args.max_side)
            input_tensor = to_tensor(image).unsqueeze(0).to(device)
            input_tensor, original_h, original_w = pad_to_multiple(input_tensor)

            try:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    output = model(input_tensor)
            except RuntimeError as error:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"EVSSM whole-image inference failed on image {image_index} ({image_path.name}) at {image.size}. "
                    "This is usually GPU memory pressure or a CUDA kernel failure from an image that is too large. "
                    "Try --precision fp16, --max-side 2048, then 1536 or 1024 if it still fails."
                ) from error
            output = output[:, :, :original_h, :original_w]
            output = torch.clamp(output, 0, 1).squeeze(0).cpu()

            output_path = args.output_dir / image_path.name
            to_pil_image(output).save(output_path)


if __name__ == "__main__":
    main()
