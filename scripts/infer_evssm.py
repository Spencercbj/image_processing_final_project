import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EVSSM inference on a folder of images.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to EVSSM checkpoint, e.g. net_g_GoPro.pth.")
    parser.add_argument("--input-dir", default=ROOT / "img", type=Path, help="Folder containing blurred input images.")
    parser.add_argument("--output-dir", default=ROOT / "results" / "EVSSM", type=Path, help="Folder for deblurred outputs.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = EVSSM().to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device), strict=True)
    model.eval()

    image_paths = iter_images(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="EVSSM inference"):
            image = Image.open(image_path).convert("RGB")
            input_tensor = to_tensor(image).unsqueeze(0).to(device)
            input_tensor, original_h, original_w = pad_to_multiple(input_tensor)

            output = model(input_tensor)
            output = output[:, :, :original_h, :original_w]
            output = torch.clamp(output, 0, 1).squeeze(0).cpu()

            output_path = args.output_dir / image_path.name
            to_pil_image(output).save(output_path)


if __name__ == "__main__":
    main()
