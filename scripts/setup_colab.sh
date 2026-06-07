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
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("nvcc:", shutil.which("nvcc"))

if not torch.cuda.is_available():
    raise SystemExit("Colab GPU is not enabled. Use Runtime > Change runtime type > GPU.")
if shutil.which("nvcc") is None:
    raise SystemExit("nvcc was not found. mamba-ssm needs a Colab GPU runtime with CUDA compiler support.")
PY

MAMBA_SSM_VERSION="${MAMBA_SSM_VERSION:-2.2.2}"
MAMBA_SSM_FALLBACK_VERSION="${MAMBA_SSM_FALLBACK_VERSION:-2.3.2.post1}"

install_mamba_ssm() {
  local version="$1"
  echo "Installing mamba-ssm==${version}"
  MAX_JOBS="${MAX_JOBS:-2}" python -m pip install "mamba-ssm==${version}" --no-build-isolation --no-cache-dir -v
}

if ! install_mamba_ssm "${MAMBA_SSM_VERSION}"; then
  echo "mamba-ssm==${MAMBA_SSM_VERSION} failed to build or install."
  if [[ "${MAMBA_SSM_FALLBACK_VERSION}" != "${MAMBA_SSM_VERSION}" ]]; then
    echo "Trying fallback mamba-ssm==${MAMBA_SSM_FALLBACK_VERSION}."
    install_mamba_ssm "${MAMBA_SSM_FALLBACK_VERSION}"
  else
    exit 1
  fi
fi

if [[ "${INSTALL_CAUSAL_CONV1D:-0}" == "1" ]]; then
  MAX_JOBS="${MAX_JOBS:-2}" python -m pip install causal-conv1d==1.4.0 --no-build-isolation -v
else
  echo "Skipping causal-conv1d. EVSSM inference uses mamba_ssm selective_scan and does not require causal-conv1d."
  echo "Set INSTALL_CAUSAL_CONV1D=1 before running this script if you explicitly need causal-conv1d."
fi

python - <<'PY'
import torch
import mamba_ssm
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

print("mamba_ssm:", mamba_ssm.__version__)
print("Mamba import ok:", Mamba)
print("selective_scan_fn import ok:", selective_scan_fn)

x = torch.randn(1, 64, 64, device="cuda")
model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda().eval()
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()
print("Mamba smoke test:", y.shape, y.device)
PY
