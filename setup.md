# Environment Setup

This project uses `mamba-ssm==2.2.2`, `causal-conv1d==1.4.0`, and CUDA-enabled PyTorch. These packages include CUDA extensions, so the install environment must have both PyTorch and `nvcc`.

For Google Colab, use [COLAB.md](COLAB.md). Colab does not use this Conda setup directly.

## 1. Activate The Conda Environment

```bash
conda activate image-processing-evssm
```

If `conda` is not available in the current shell, load Conda's shell hook first:

```bash
source /home/spencer/miniconda3/etc/profile.d/conda.sh
conda activate image-processing-evssm
```

For one-off commands without activating the shell, use:

```bash
/home/spencer/miniconda3/bin/conda run -n image-processing-evssm <command>
```

## 2. Install Build Tools

`mamba-ssm==2.2.2` and `causal-conv1d==1.4.0` are built against the PyTorch installed in the active environment. Keep `setuptools` below 70 because `torch==2.1.0` still imports `pkg_resources`.

```bash
python -m pip install "setuptools<70" wheel packaging ninja
```

## 3. Install CUDA Compiler

The PyTorch `+cu121` wheel includes CUDA runtime libraries, but building these extensions also needs `nvcc`.

```bash
conda install -c nvidia/label/cuda-12.1.0 cuda-nvcc cuda-cudart-dev -y
```

Check that `nvcc` is visible:

```bash
nvcc --version
```

Expected CUDA compiler version:

```text
Cuda compilation tools, release 12.1
```

## 4. Install Mamba Packages

Install with `--no-build-isolation` so the build uses the current environment's CUDA-enabled PyTorch.

```bash
python -m pip install mamba-ssm==2.2.2 --no-build-isolation
python -m pip install transformers==4.44.2 causal-conv1d==1.4.0 --no-build-isolation
```

`transformers==4.44.2` is pinned because newer `transformers` releases are not compatible with `mamba-ssm==2.2.2` and `torch==2.1.0`.

## 5. Verify Package Imports

```bash
python -c "import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__); from mamba_ssm import Mamba; print('Mamba import ok'); import causal_conv1d; print('causal_conv1d import ok')"
```

Expected output:

```text
mamba_ssm 2.2.2
Mamba import ok
causal_conv1d import ok
```

## GPU Environment Test

Run this first to confirm PyTorch can see the GPU:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda build', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Expected result:

```text
cuda available True
device count 1
device <your NVIDIA GPU name>
```

Then run a small CUDA tensor test:

```bash
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; torch.cuda.synchronize(); print(y.shape, y.device, y.float().mean().item())"
```

Expected result:

```text
torch.Size([1024, 1024]) cuda:0 <some number>
```

Finally, run a small Mamba forward pass on GPU:

```bash
python - <<'PY'
import torch
from mamba_ssm import Mamba

assert torch.cuda.is_available(), "CUDA is not available to PyTorch"

model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 128, 64, device="cuda")

with torch.no_grad():
    y = model(x)

torch.cuda.synchronize()
print("input:", x.shape, x.device)
print("output:", y.shape, y.device)
print("ok")
PY
```

Expected result:

```text
input: torch.Size([2, 128, 64]) cuda:0
output: torch.Size([2, 128, 64]) cuda:0
ok
```

## Run EVSSM Inference On `img/`

The original EVSSM `test.py` expects benchmark-style folders with both blurred inputs and sharp targets:

```text
<data_dir>/test/input
<data_dir>/test/target
```

For this project's unlabeled images in `img/`, use the folder inference helper:

```bash
conda activate image-processing-evssm
python scripts/evssm_infer_folder.py --dry-run
```

If the dry run lists the images correctly, run inference with the GoPro checkpoint:

```bash
python scripts/evssm_infer_folder.py \
  --input img \
  --checkpoint checkpoints/net_g_GoPro.pth \
  --output results/EVSSM/GoPro \
  --tile-size 640 \
  --overlap 96 \
  --blend cosine
```

To test only one image first:

```bash
python scripts/evssm_infer_folder.py \
  --input img \
  --checkpoint checkpoints/net_g_GoPro.pth \
  --output results/EVSSM/GoPro_smoke \
  --tile-size 640 \
  --overlap 96 \
  --blend cosine \
  --limit 1
```

Available checkpoints:

```text
checkpoints/net_g_GoPro.pth
checkpoints/net_g_realblur_j.pth
checkpoints/net_g_realblur_r.pth
```

The images in `img/` are high resolution, so tiled inference is recommended. If CUDA runs out of memory, reduce the tile size:

```bash
python scripts/evssm_infer_folder.py --tile-size 384 --overlap 32
```

If there is still GPU memory available, try increasing tile size first, then overlap:

```bash
python scripts/evssm_infer_folder.py --tile-size 768 --overlap 128 --blend cosine --limit 1
```

There is also a whole-image inference helper:

```bash
python scripts/evssm_infer_whole_image.py \
  --checkpoint checkpoints/net_g_GoPro.pth \
  --input-dir img \
  --output-dir results/EVSSM/GoPro_whole
```

Use the whole-image helper only for small images or when the GPU has enough VRAM. For the current high-resolution `img/` folder, `scripts/evssm_infer_folder.py` is the recommended pipeline.

## Troubleshooting

If `torch.cuda.is_available()` is `False`, check:

```bash
nvidia-smi
```

If `nvidia-smi` fails, the NVIDIA driver or GPU access is not working in the current shell/container/session.

If importing `mamba_ssm` fails with a `transformers` error, reinstall the compatible version:

```bash
python -m pip install transformers==4.44.2
```

If installation fails with `No module named 'pkg_resources'`, reinstall compatible setuptools:

```bash
python -m pip install "setuptools<70"
```

If installation fails with `nvcc was not found`, install CUDA compiler support:

```bash
conda install -c nvidia/label/cuda-12.1.0 cuda-nvcc cuda-cudart-dev -y
```
