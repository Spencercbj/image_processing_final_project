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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run whole-image EVSSM inference on a folder of small images.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to EVSSM checkpoint, e.g. net_g_GoPro.pth.")
    parser.add_argument("--input-dir", default=ROOT / "img", type=Path, help="Folder containing blurred input images.")
    parser.add_argument("--output-dir", default=ROOT / "results" / "EVSSM", type=Path, help="Folder for deblurred outputs.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images. 0 means all images.")
    parser.add_argument("--max-side", type=int, default=0, help="Resize the longest side before inference. 0 keeps original size.")
    parser.add_argument("--dry-run", action="store_true", help="List planned inputs/outputs without running the model.")
    args = parser.parse_args()

    image_paths = iter_images(args.input_dir)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    print(f"input: {args.input_dir}")
    print(f"output: {args.output_dir}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"images: {len(image_paths)}")
    print("mode: whole image")
    print(f"max_side: {args.max_side if args.max_side > 0 else 'original'}")

    if args.dry_run:
        for image_path in image_paths:
            with Image.open(image_path) as image:
                resized = resize_to_max_side(image, args.max_side)
                print(f"{image_path.name}: {image.size} -> {resized.size} -> {args.output_dir / image_path.name}")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run the GPU test in setup.md first, or pass --device cpu for debugging.")
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = EVSSM().to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device), strict=True)
    model.eval()

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="EVSSM inference"):
            image = Image.open(image_path).convert("RGB")
            image = resize_to_max_side(image, args.max_side)
            input_tensor = to_tensor(image).unsqueeze(0).to(device)
            input_tensor, original_h, original_w = pad_to_multiple(input_tensor)

            try:
                output = model(input_tensor)
            except RuntimeError as error:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"EVSSM whole-image inference failed on {image_path.name} at {image.size}. "
                    "This is usually GPU memory pressure or a CUDA kernel failure from an image that is too large. "
                    "Try --max-side 2048, then 1536 or 1024 if it still fails."
                ) from error
            output = output[:, :, :original_h, :original_w]
            output = torch.clamp(output, 0, 1).squeeze(0).cpu()

            output_path = args.output_dir / image_path.name
            to_pil_image(output).save(output_path)


if __name__ == "__main__":
    main()
