#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python -m pip install -U pip
python -m pip install \
  einops \
  natsort \
  opencv-python \
  scikit-image \
  tqdm

python - <<'PY'
import sys
import torch
from pathlib import Path
from runpy import run_path

root = Path.cwd()
arch_path = root / "methods" / "Restormer" / "basicsr" / "models" / "archs" / "restormer_arch.py"
if not arch_path.is_file():
    raise SystemExit(
        f"Restormer arch not found at {arch_path}. "
        "Run: git submodule update --init --recursive methods/Restormer"
    )

load_arch = run_path(str(arch_path))
Restormer = load_arch["Restormer"]
model = Restormer(
    inp_channels=3,
    out_channels=3,
    dim=48,
    num_blocks=[4, 6, 6, 8],
    num_refinement_blocks=4,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type="WithBias",
    dual_pixel_task=False,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
x = torch.randn(1, 3, 64, 64, device=device)
with torch.inference_mode():
    y = model(x)
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
print("Restormer smoke test:", y.shape, y.device)
PY
