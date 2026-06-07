#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DARKIR_ROOT = ROOT / "methods" / "DarkIR"
DARKIR_ARCHS = DARKIR_ROOT / "archs"
sys.path.insert(0, str(DARKIR_ARCHS))

from DarkIR import DarkIR  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run DarkIR inference on a folder of images.")
    parser.add_argument("--input", default=str(ROOT / "img"), help="Input image folder.")
    parser.add_argument("--output", default=str(ROOT / "results" / "DarkIR"), help="Output folder.")
    parser.add_argument("--config", default=str(DARKIR_ROOT / "options" / "inference" / "LOLBlur.yml"), help="DarkIR config yaml.")
    parser.add_argument("--checkpoint", default=None, help="DarkIR checkpoint path. If omitted, uses save.path from config.")
    parser.add_argument("--tile-size", type=int, default=0, help="Tile size for inference. 0 means whole image.")
    parser.add_argument("--overlap", type=int, default=256, help="Overlap between neighboring tiles.")
    parser.add_argument("--blend", default="crop", choices=["cosine", "crop", "uniform"], help="Tile blending mode.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based index in sorted input order to start from.")
    parser.add_argument("--limit", type=int, default=0, help="Only process N images. 0 means all selected images.")
    parser.add_argument("--max-side", type=int, default=0, help="Resize longest side before inference. 0 keeps original size.")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"], help="Inference precision.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--dry-run", action="store_true", help="List planned inputs/outputs without running the model.")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(path, base_dir):
    path = Path(path)
    if path.is_absolute():
        return path
    return base_dir / path


def list_images(input_dir):
    input_dir = Path(input_dir)
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def select_images(image_paths, start_index, limit):
    if start_index < 1:
        raise ValueError("--start-index is 1-based and must be >= 1")
    indexed_paths = list(enumerate(image_paths, start=1))[start_index - 1 :]
    if limit > 0:
        indexed_paths = indexed_paths[:limit]
    return indexed_paths


def resize_to_max_side(image, max_side):
    if max_side <= 0:
        return image
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_side:
        return image
    scale = max_side / long_side
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)


def checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("params", "state_dict", "model_state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = normalize_state_dict_keys(checkpoint_state_dict(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"warning: unexpected checkpoint keys: {len(unexpected)}")
    if missing:
        print(f"warning: missing checkpoint keys: {len(missing)}")
    if missing or unexpected:
        model_keys = set(model.state_dict())
        loaded_keys = set(state_dict)
        matched = len(model_keys & loaded_keys)
        if matched == 0:
            raise RuntimeError("Checkpoint did not match the DarkIR model architecture.")
        print(f"matched checkpoint keys: {matched} / {len(model_keys)}")


def build_model(network_opt):
    return DarkIR(
        img_channel=network_opt.get("img_channels", 3),
        width=network_opt.get("width", 32),
        middle_blk_num_enc=network_opt.get("middle_blk_num_enc", 2),
        middle_blk_num_dec=network_opt.get("middle_blk_num_dec", 2),
        enc_blk_nums=network_opt.get("enc_blk_nums", [1, 2, 3]),
        dec_blk_nums=network_opt.get("dec_blk_nums", [3, 1, 1]),
        dilations=network_opt.get("dilations", [1, 4, 9]),
        extra_depth_wise=network_opt.get("extra_depth_wise", True),
    )


def pad_to_multiple(x, multiple):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    return F.pad(x, (0, pad_w, 0, pad_h), value=0), h, w


def autocast_dtype(precision):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def run_model(model, tile, device, precision):
    x = tile.unsqueeze(0).to(device, non_blocking=True)
    x, h, w = pad_to_multiple(x, 8)
    amp_dtype = autocast_dtype(precision)
    use_amp = device == "cuda" and amp_dtype is not None
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
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


def infer_image(model, image_tensor, device, tile_size, overlap, blend, precision):
    _, h, w = image_tensor.shape
    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        return run_model(model, image_tensor, device, precision)

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
            pred = run_model(model, tile, device, precision)
            tile_h, tile_w = pred.shape[-2:]
            tile_wgt = tile_weight(tile_h, tile_w, y, x, h, w, overlap, blend)
            output[:, y : y + tile_h, x : x + tile_w] += pred * tile_wgt
            weight[:, y : y + tile_h, x : x + tile_w] += tile_wgt

    return output / weight.clamp_min(1)


def main():
    args = parse_args()
    config_path = Path(args.config)
    opt = load_config(config_path)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else resolve_path(opt["save"]["path"], DARKIR_ROOT)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    images = list_images(input_dir)
    indexed_images = select_images(images, args.start_index, args.limit)

    print(f"input: {input_dir}")
    print(f"output: {output_dir}")
    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"images: {len(indexed_images)} / {len(images)}")
    print(f"start_index: {args.start_index}")
    print(f"tile_size: {args.tile_size}, overlap: {args.overlap}, blend: {args.blend}")
    print(f"max_side: {args.max_side if args.max_side > 0 else 'original'}")
    print(f"precision: {args.precision}")

    if not indexed_images:
        raise SystemExit("No images found.")

    if args.dry_run:
        for image_index, image_path in indexed_images:
            with Image.open(image_path) as image:
                resized = resize_to_max_side(image, args.max_side)
                print(f"{image_index:02d}. {image_path.name}: {image.size} -> {resized.size} -> {output_dir / image_path.name}")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Use a GPU runtime or pass --device cpu for debugging.")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(opt["network"]).to(args.device)
    load_checkpoint(model, checkpoint_path, args.device)
    model.eval()

    with torch.inference_mode():
        for image_index, image_path in tqdm(indexed_images, desc="DarkIR inference"):
            image = Image.open(image_path).convert("RGB")
            image = resize_to_max_side(image, args.max_side)
            image_tensor = TF.to_tensor(image)
            try:
                pred = infer_image(
                    model=model,
                    image_tensor=image_tensor,
                    device=args.device,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    blend=args.blend,
                    precision=args.precision,
                )
            except RuntimeError as error:
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"DarkIR inference failed on image {image_index} ({image_path.name}) at {image.size}. "
                    "Try a smaller --tile-size, --precision fp16, or --max-side."
                ) from error
            TF.to_pil_image(pred.clamp(0, 1)).save(output_dir / image_path.name)

    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
