#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
EVSSM_ROOT = ROOT / "methods" / "EVSSM"
sys.path.insert(0, str(EVSSM_ROOT))

from models.EVSSM import EVSSM  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run tiled EVSSM inference on a folder of high-resolution images.")
    parser.add_argument("--input", default=str(ROOT / "img"), help="Input image folder.")
    parser.add_argument("--output", default=str(ROOT / "results" / "EVSSM" / "GoPro"), help="Output folder.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "net_g_GoPro.pth"), help="EVSSM checkpoint path.")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size for inference. Use 0 for whole image.")
    parser.add_argument("--overlap", type=int, default=96, help="Overlap between neighboring tiles.")
    parser.add_argument("--blend", default="cosine", choices=["cosine", "crop", "uniform"], help="Tile blending mode.")
    parser.add_argument("--pad-multiple", type=int, default=4, help="Pad each tile so H/W are divisible by this value.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based index in sorted input order to start from.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images. 0 means all images.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--dry-run", action="store_true", help="List planned inputs/outputs without running the model.")
    return parser.parse_args()


def list_images(input_dir):
    input_dir = Path(input_dir)
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def select_images(image_paths, start_index, limit):
    if start_index < 1:
        raise ValueError("--start-index is 1-based and must be >= 1")

    indexed_paths = list(enumerate(image_paths, start=1))
    indexed_paths = indexed_paths[start_index - 1 :]
    if limit > 0:
        indexed_paths = indexed_paths[:limit]
    return indexed_paths


def load_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        return checkpoint.get("params", checkpoint.get("state_dict", checkpoint))
    return checkpoint


def pad_to_multiple(x, multiple):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), h, w


def run_model(model, tile, device, pad_multiple):
    x = tile.unsqueeze(0).to(device, non_blocking=True)
    x, h, w = pad_to_multiple(x, pad_multiple)
    pred = model(x)
    pred = pred[:, :, :h, :w].clamp(0, 1)
    if device == "cuda":
        torch.cuda.synchronize()
    return pred.squeeze(0).cpu()


def tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def cosine_ramp(length):
    if length <= 0:
        return torch.empty(0)
    if length == 1:
        return torch.ones(1)
    t = torch.linspace(0, 1, steps=length)
    return 0.5 - 0.5 * torch.cos(torch.pi * t)


def tile_weight(tile_h, tile_w, y, x, image_h, image_w, overlap, blend):
    weight = torch.ones((1, tile_h, tile_w))
    if blend == "uniform" or overlap <= 0:
        return weight

    if blend == "crop":
        border_h = min(overlap // 2, tile_h // 2)
        border_w = min(overlap // 2, tile_w // 2)

        if y > 0 and border_h > 0:
            weight[:, :border_h, :] = 1e-3
        if y + tile_h < image_h and border_h > 0:
            weight[:, -border_h:, :] = 1e-3
        if x > 0 and border_w > 0:
            weight[:, :, :border_w] = 1e-3
        if x + tile_w < image_w and border_w > 0:
            weight[:, :, -border_w:] = 1e-3

        return weight

    ramp_h = min(overlap, tile_h)
    ramp_w = min(overlap, tile_w)

    if y > 0 and ramp_h > 0:
        weight[:, :ramp_h, :] *= cosine_ramp(ramp_h).view(1, ramp_h, 1).clamp_min(1e-3)
    if y + tile_h < image_h and ramp_h > 0:
        weight[:, -ramp_h:, :] *= cosine_ramp(ramp_h).flip(0).view(1, ramp_h, 1).clamp_min(1e-3)
    if x > 0 and ramp_w > 0:
        weight[:, :, :ramp_w] *= cosine_ramp(ramp_w).view(1, 1, ramp_w).clamp_min(1e-3)
    if x + tile_w < image_w and ramp_w > 0:
        weight[:, :, -ramp_w:] *= cosine_ramp(ramp_w).flip(0).view(1, 1, ramp_w).clamp_min(1e-3)

    return weight


def infer_image(model, image_tensor, device, tile_size, overlap, pad_multiple, blend):
    _, h, w = image_tensor.shape
    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        return run_model(model, image_tensor, device, pad_multiple)

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("--overlap must be smaller than --tile-size")

    output = torch.zeros_like(image_tensor)
    weight = torch.zeros((1, h, w), dtype=image_tensor.dtype)

    ys = tile_starts(h, tile_size, stride)
    xs = tile_starts(w, tile_size, stride)

    for y in ys:
        for x in xs:
            tile = image_tensor[:, y : y + tile_size, x : x + tile_size]
            pred = run_model(model, tile, device, pad_multiple)
            tile_h, tile_w = pred.shape[-2:]
            tile_wgt = tile_weight(tile_h, tile_w, y, x, h, w, overlap, blend)
            output[:, y : y + tile_h, x : x + tile_w] += pred * tile_wgt
            weight[:, y : y + tile_h, x : x + tile_w] += tile_wgt

    return output / weight.clamp_min(1)


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    checkpoint = Path(args.checkpoint)

    images = list_images(input_dir)
    indexed_images = select_images(images, args.start_index, args.limit)

    print(f"input: {input_dir}")
    print(f"output: {output_dir}")
    print(f"checkpoint: {checkpoint}")
    print(f"images: {len(indexed_images)} / {len(images)}")
    print(f"start_index: {args.start_index}")
    print(f"tile_size: {args.tile_size}, overlap: {args.overlap}, blend: {args.blend}")

    if not indexed_images:
        raise SystemExit("No images found.")

    if args.dry_run:
        for image_index, image_path in indexed_images:
            with Image.open(image_path) as image:
                print(f"{image_index:02d}. {image_path.name}: {image.size} -> {output_dir / image_path.name}")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run the GPU test in setup.md first, or pass --device cpu for debugging.")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = EVSSM()
    model.load_state_dict(load_state_dict(checkpoint), strict=True)
    model.to(args.device)
    model.eval()

    for image_index, image_path in tqdm(indexed_images, desc="EVSSM inference"):
        image = Image.open(image_path).convert("RGB")
        image_tensor = TF.to_tensor(image)

        with torch.no_grad():
            pred = infer_image(
                model=model,
                image_tensor=image_tensor,
                device=args.device,
                tile_size=args.tile_size,
                overlap=args.overlap,
                pad_multiple=args.pad_multiple,
                blend=args.blend,
            )

        pred = (pred + 0.5 / 255).clamp(0, 1)
        TF.to_pil_image(pred).save(output_dir / image_path.name)

    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
