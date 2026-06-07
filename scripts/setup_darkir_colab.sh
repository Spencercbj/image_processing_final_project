#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
python -m pip install \
  einops==0.8.0 \
  kornia==0.7.2 \
  numpy \
  opencv-python==4.10.0.84 \
  pandas==2.2.2 \
  pillow==10.3.0 \
  ptflops==0.7.3 \
  PyYAML==6.0.1 \
  scikit-image==0.24.0 \
  scipy \
  tqdm==4.66.4

python - <<'PY'
import sys
import torch
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "methods" / "DarkIR" / "archs"))

from DarkIR import DarkIR

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")

model = DarkIR(width=32).cuda().eval() if torch.cuda.is_available() else DarkIR(width=32).eval()
x = torch.randn(1, 3, 64, 64)
if torch.cuda.is_available():
    x = x.cuda()
with torch.inference_mode():
    y = model(x)
print("DarkIR smoke test:", y.shape, y.device)
PY
