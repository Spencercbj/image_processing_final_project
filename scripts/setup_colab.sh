#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
python -m pip install "setuptools<70" wheel packaging ninja

python -m pip install \
  einops==0.8.0 \
  imageio==2.35.1 \
  lmdb==1.5.1 \
  matplotlib==3.9.2 \
  numpy==1.26.3 \
  opencv-python==4.10.0.84 \
  pillow==10.2.0 \
  pyyaml==6.0.2 \
  scikit-image==0.24.0 \
  scipy==1.14.1 \
  tensorboard==2.17.1 \
  termcolor==2.5.0 \
  thop==0.1.1.post2209072238 \
  tqdm==4.66.5 \
  yacs==0.1.8 \
  transformers==4.44.2

python - <<'PY'
import shutil
import torch

print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("nvcc:", shutil.which("nvcc"))

if not torch.cuda.is_available():
    raise SystemExit("Colab GPU is not enabled. Use Runtime > Change runtime type > GPU.")
if shutil.which("nvcc") is None:
    raise SystemExit("nvcc was not found. mamba-ssm needs a Colab GPU runtime with CUDA compiler support.")
PY

MAX_JOBS="${MAX_JOBS:-2}" python -m pip install causal-conv1d==1.4.0 --no-build-isolation
MAX_JOBS="${MAX_JOBS:-2}" python -m pip install mamba-ssm==2.2.2 --no-build-isolation

python - <<'PY'
import torch
import mamba_ssm
import causal_conv1d
from mamba_ssm import Mamba

print("mamba_ssm:", mamba_ssm.__version__)
print("causal_conv1d import ok")
print("Mamba import ok:", Mamba)

x = torch.randn(1, 64, 64, device="cuda")
model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda().eval()
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()
print("Mamba smoke test:", y.shape, y.device)
PY
