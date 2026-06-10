#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WEIGHT_DIR="${WEIGHT_DIR:-/content/drive/MyDrive/deblur_weights}"
IMAGE_DIR="${IMAGE_DIR:-/content/drive/MyDrive/deblur_project/img}"
OMDNET_DIR="${OMDNET_DIR:-/content/drive/MyDrive/deblur_project/OMDNet}"
COPY_IMAGES="${COPY_IMAGES:-1}"
COPY_WEIGHTS="${COPY_WEIGHTS:-1}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
INSTALL_DIFFBIR_REQUIREMENTS="${INSTALL_DIFFBIR_REQUIREMENTS:-1}"

cd "$PROJECT_ROOT"

log() {
  printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

install_python_deps() {
  log "Installing Python dependencies"
  python -m pip install --upgrade pip setuptools wheel

  if [[ "$INSTALL_TORCH" == "1" ]]; then
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  fi

  python -m pip install \
    opencv-python-headless pillow numpy scipy scikit-image \
    einops timm tqdm pyyaml lmdb addict yapf future requests gdown \
    matplotlib

  python -m pip install basicsr facexlib gfpgan realesrgan

  python -m pip install \
    omegaconf pytorch-lightning transformers accelerate safetensors \
    kornia open-clip-torch sentencepiece protobuf
}

clone_if_missing() {
  local repo="$1"
  local dir="$2"
  if [[ -d "$dir" ]]; then
    echo "[skip] $dir already exists"
    return
  fi
  git clone --depth 1 "$repo" "$dir"
}

prepare_external_repos() {
  log "Preparing external model repositories"
  clone_if_missing https://github.com/Fundacion-Cidaut/DarkIR.git DarkIR
  clone_if_missing https://github.com/swz30/Restormer.git Restormer
  clone_if_missing https://github.com/yudingchuan/OMDNet.git OMDNet
  clone_if_missing https://github.com/XPixelGroup/DiffBIR.git DiffBIR
  clone_if_missing https://github.com/xinntao/Real-ESRGAN.git Real-ESRGAN

  if [[ ! -d OMDNet && -d "$OMDNET_DIR" ]]; then
    cp -r "$OMDNET_DIR" ./OMDNet
  fi

  if [[ "$INSTALL_DIFFBIR_REQUIREMENTS" == "1" && -f DiffBIR/requirements.txt ]]; then
    grep -vi '^xformers' DiffBIR/requirements.txt > /tmp/diffbir_requirements_no_xformers.txt || true
    python -m pip install -r /tmp/diffbir_requirements_no_xformers.txt
  fi
}

copy_inputs() {
  if [[ "$COPY_IMAGES" != "1" ]]; then
    return
  fi

  log "Copying input images"
  mkdir -p img
  if [[ -d "$IMAGE_DIR" ]]; then
    shopt -s nullglob
    local images=("$IMAGE_DIR"/*)
    if (( ${#images[@]} > 0 )); then
      cp -n "${images[@]}" img/
    else
      echo "WARNING: no input files found in IMAGE_DIR: $IMAGE_DIR"
    fi
    shopt -u nullglob
  else
    echo "WARNING: IMAGE_DIR not found: $IMAGE_DIR"
  fi
}

copy_file_if_present() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
  else
    echo "WARNING: missing weight: $src"
  fi
}

copy_weights() {
  if [[ "$COPY_WEIGHTS" != "1" ]]; then
    return
  fi

  log "Copying weights from $WEIGHT_DIR"
  mkdir -p DarkIR/models
  mkdir -p Restormer/Motion_Deblurring/pretrained_models
  mkdir -p OMDNet/checkpoints/test1
  mkdir -p Real-ESRGAN/weights
  mkdir -p DiffBIR/weights

  copy_file_if_present "$WEIGHT_DIR/DarkIR_384.pt" DarkIR/models/DarkIR_384.pt
  copy_file_if_present "$WEIGHT_DIR/motion_deblurring.pth" Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth
  copy_file_if_present "$WEIGHT_DIR/model_ckpt_epoch_219.ckpt" OMDNet/checkpoints/test1/model_ckpt_epoch_219.ckpt
  copy_file_if_present "$WEIGHT_DIR/RealESRGAN_x4plus.pth" Real-ESRGAN/weights/RealESRGAN_x4plus.pth

  if [[ -d "$WEIGHT_DIR/DiffBIR" ]]; then
    cp -r "$WEIGHT_DIR/DiffBIR/." DiffBIR/weights/
  else
    echo "WARNING: DiffBIR weight folder not found: $WEIGHT_DIR/DiffBIR"
  fi
}

verify_setup() {
  log "Verifying setup"
  python - <<'PY'
from pathlib import Path
import torch

required = [
    'code/pipeline/run_darkir_restormer_diffbir_v2.py',
    'code/DarkIR/inference.py',
    'code/Restormer/inference.py',
    'code/OMDNet/inference.py',
    'code/RealESRGAN/inference.py',
    'DarkIR/archs/DarkIR.py',
    'DarkIR/models/DarkIR_384.pt',
    'Restormer/basicsr/models/archs/restormer_arch.py',
    'Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth',
    'OMDNet/models/deblur_model.py',
    'OMDNet/checkpoints/test1/model_ckpt_epoch_219.ckpt',
    'Real-ESRGAN/inference_realesrgan.py',
    'Real-ESRGAN/weights/RealESRGAN_x4plus.pth',
    'DiffBIR/inference.py',
]

missing = [p for p in required if not Path(p).exists()]
if missing:
    print('Missing required files:')
    for p in missing:
        print('  -', p)
    raise SystemExit(1)

img_count = len(list(Path('img').glob('*')))
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
print('input files in img/:', img_count)
print('setup verification passed')
PY

  python -m py_compile code/pipeline/run_darkir_restormer_diffbir_v2.py
}

install_python_deps
prepare_external_repos
copy_inputs
copy_weights
verify_setup

log "Done"
