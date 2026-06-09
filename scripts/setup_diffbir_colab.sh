#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIFFBIR_DIR="${PROJECT_ROOT}/methods/DiffBIR"

cd "${PROJECT_ROOT}"

if [ ! -d "${DIFFBIR_DIR}/.git" ]; then
  git clone --depth 1 https://github.com/XPixelGroup/DiffBIR.git "${DIFFBIR_DIR}"
else
  git -C "${DIFFBIR_DIR}" pull --ff-only
fi

cd "${DIFFBIR_DIR}"

python -m pip install -U pip

# Colab already provides a CUDA-enabled PyTorch build. DiffBIR's official
# requirements pin torch/torchvision/torchaudio/xformers to CUDA 11.8 builds,
# which can fail on newer Colab Python runtimes or replace the working torch.
python -m pip install \
  omegaconf==2.3.0 \
  accelerate==0.28.0 \
  einops==0.7.0 \
  opencv_python==4.9.0.80 \
  scipy==1.12.0 \
  ftfy==6.2.0 \
  regex==2023.12.25 \
  python-dateutil==2.9.0.post0 \
  timm==0.9.16 \
  pytorch-lightning==2.2.1 \
  tensorboard==2.16.2 \
  protobuf==4.25.3 \
  lpips==0.1.4 \
  facexlib==0.3.0 \
  gradio==4.43.0 \
  polars==1.12.0 \
  torchsde==0.2.6 \
  bitsandbytes==0.44.1 \
  transformers==4.37.2 \
  tokenizers==0.15.1 \
  sentencepiece==0.1.99 \
  fairscale==0.4.4

python - <<'PY'
import sys
import torch

from diffbir.inference import BIDInferenceLoop

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
print("DiffBIR import ok:", BIDInferenceLoop)
PY
