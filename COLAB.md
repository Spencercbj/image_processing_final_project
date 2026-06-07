# Google Colab Setup

This guide runs the EVSSM inference scripts in Google Colab without Conda.

Colab runtimes change over time, so the notebook cells install the Python packages inside the active runtime instead of relying on a prebuilt local environment.

All command blocks below are intended to be pasted into Colab notebook cells. Blocks starting with `%%bash` must have `%%bash` as the first line of the cell.

For DarkIR, use [DARKIR_COLAB.md](DARKIR_COLAB.md).
For the DarkIR-to-EVSSM pipeline, use [DARKIR_EVSSM_PIPELINE_COLAB.md](DARKIR_EVSSM_PIPELINE_COLAB.md).

## 1. Enable GPU

In Colab:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

Then verify:

```python
!nvidia-smi
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
```

## 2. Get The Project Into Colab

### Option A: Clone From Git

Use this if the repository is on GitHub:

```python
%cd /content
REPO_URL = "PASTE_YOUR_REPO_URL_HERE"
assert REPO_URL != "PASTE_YOUR_REPO_URL_HERE", "Set REPO_URL before running this cell."
!git clone --recursive {REPO_URL} image_processing_final_project
%cd /content/image_processing_final_project
```

If the repository was cloned without submodules:

```python
!git submodule update --init --recursive methods/EVSSM
```

### Option B: Use Google Drive

Use this if the project folder is already in Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then change into the project folder. Adjust the path if yours is different:

```python
%cd /content/drive/MyDrive/image_processing_final_project
!git submodule update --init --recursive methods/EVSSM
```

## 3. Check Required Files

The project should contain:

```text
img/                               # input images only
checkpoints/                       # model weights
checkpoints/net_g_GoPro.pth
checkpoints/net_g_realblur_j.pth
checkpoints/net_g_realblur_r.pth
methods/EVSSM/
scripts/evssm_infer_folder.py
scripts/evssm_infer_whole_image.py
```

Check them in Colab:

```python
!find img -maxdepth 1 -type f | head
!ls -lh checkpoints
!test -f methods/EVSSM/models/EVSSM.py && echo "EVSSM submodule ok"
```

If the `checkpoints/` folder is empty in Colab and your weights are stored in Google Drive, copy them into the current Colab project folder.

First mount Drive if it is not already mounted:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then set the Drive folder that contains the checkpoint files and copy them:

```python
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/EVSSM_checkpoints"

!mkdir -p checkpoints
!cp "{DRIVE_CHECKPOINT_DIR}/net_g_GoPro.pth" checkpoints/
!cp "{DRIVE_CHECKPOINT_DIR}/net_g_realblur_j.pth" checkpoints/
!cp "{DRIVE_CHECKPOINT_DIR}/net_g_realblur_r.pth" checkpoints/
!ls -lh checkpoints
```

If your checkpoint folder has a different name, change `DRIVE_CHECKPOINT_DIR` before running the cell.

## 4. Install EVSSM Dependencies

Run:

```python
!bash scripts/setup_colab.sh
```

This installs build tools, project dependencies, and `mamba-ssm==2.3.2.post1` against Colab's active CUDA-enabled PyTorch.
Colab currently tends to use newer Python/PyTorch builds, so the setup script uses the newer Mamba package first. If it cannot build on the current Colab runtime, the setup script automatically tries `mamba-ssm==2.2.2` as a fallback.

`causal-conv1d` is skipped by default in Colab because it often fails while building a CUDA wheel. The EVSSM inference scripts in this project use `mamba_ssm.ops.selective_scan_interface`, so `causal-conv1d` is not required for the current pipeline.

To explicitly use the local-project Mamba version first:

```python
!MAMBA_SSM_VERSION=2.2.2 MAMBA_SSM_FALLBACK_VERSION=2.3.2.post1 bash scripts/setup_colab.sh
```

If you explicitly need `causal-conv1d`, run:

```python
!INSTALL_CAUSAL_CONV1D=1 bash scripts/setup_colab.sh
```

If Colab crashes or asks for a runtime restart after dependency installation, restart the runtime and run the cells again from step 1. Pip-built CUDA extensions may need a clean runtime after installation.

## 5. Run A Dry Run

Tiled inference dry run:

```python
!python scripts/evssm_infer_folder.py --dry-run --limit 1
```

Whole-image dry run:

```python
%%bash
python scripts/evssm_infer_whole_image.py \
  --checkpoint checkpoints/net_g_GoPro.pth \
  --output-dir results/EVSSM/GoPro_whole_1536 \
  --limit 1 \
  --max-side 1536 \
  --dry-run
```

## 6. Recommended Inference Commands

### Tiled Inference For High-Resolution Images

Use this when full-resolution images do not fit in GPU memory:

```python
%%bash
python scripts/evssm_infer_folder.py \
  --input img \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output results/EVSSM/RealBlurJ_tile768 \
  --tile-size 768 \
  --overlap 128 \
  --blend cosine \
  --limit 1
```

If Colab runs out of memory:

```python
%%bash
python scripts/evssm_infer_folder.py \
  --input img \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output results/EVSSM/RealBlurJ_tile512 \
  --tile-size 512 \
  --overlap 96 \
  --blend cosine \
  --limit 1
```

### Whole-Image Inference

Use this when you want to avoid tile stitching. Since the current `img/` files are large, use `--max-side` first:

```python
%%bash
python scripts/evssm_infer_whole_image.py \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output-dir results/EVSSM/RealBlurJ_whole_1536 \
  --limit 1 \
  --max-side 1536
```

To run the original image size without resizing, omit `--max-side`:

```python
%%bash
python scripts/evssm_infer_whole_image.py \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output-dir results/EVSSM/RealBlurJ_whole_original \
  --limit 1
```

This sends the full-resolution image directly into EVSSM. It needs much more GPU memory than the `--max-side` version. If Colab reports CUDA OOM or an internal CUDA error, use `--max-side 1536`, `--max-side 1024`, or the tiled inference command above.

For an L4 GPU, try full-resolution whole-image inference with half precision:

```python
%%bash
python scripts/evssm_infer_whole_image.py \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output-dir results/EVSSM/RealBlurJ_whole_original_fp16 \
  --limit 1 \
  --precision fp16
```

If you just hit CUDA OOM, restart the runtime before retrying, or clear unused GPU memory:

```python
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```

GoPro checkpoint:

```python
%%bash
python scripts/evssm_infer_whole_image.py \
  --checkpoint checkpoints/net_g_GoPro.pth \
  --output-dir results/EVSSM/GoPro_whole_1536 \
  --limit 1 \
  --max-side 1536
```

To start from any image in sorted filename order, use `--start-index`. This example tests the 8th image only:

```python
%%bash
python scripts/evssm_infer_whole_image.py \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output-dir results/EVSSM/RealBlurJ_whole_original_from08 \
  --start-index 8 \
  --limit 1
```

You can also use `--start-index` with tiled inference:

```python
%%bash
python scripts/evssm_infer_folder.py \
  --input img \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output results/EVSSM/RealBlurJ_tile768_from08 \
  --tile-size 768 \
  --overlap 128 \
  --blend cosine \
  --start-index 8 \
  --limit 1
```

## 7. Download Results

Zip results:

```python
!zip -r evssm_results.zip results/EVSSM
```

Download:

```python
from google.colab import files
files.download("evssm_results.zip")
```

## Troubleshooting

If `causal-conv1d` fails while building wheels, use the default Colab setup:

```python
!bash scripts/setup_colab.sh
```

The default setup skips `causal-conv1d` because it is not required by this project's EVSSM inference path.

If `mamba-ssm` fails while building wheels, pull the latest setup script and rerun:

```python
!git pull
!bash scripts/setup_colab.sh
```

The setup script first tries `mamba-ssm==2.3.2.post1`, then falls back to `mamba-ssm==2.2.2`. To force the local-project version first:

```python
!MAMBA_SSM_VERSION=2.2.2 MAMBA_SSM_FALLBACK_VERSION=2.3.2.post1 bash scripts/setup_colab.sh
```

If it still fails, print the Colab runtime versions:

```python
import sys, torch, shutil
print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("nvcc:", shutil.which("nvcc"))
```

If `mamba-ssm` fails with `No module named 'torch'`, make sure Colab GPU runtime has PyTorch:

```python
import torch
print(torch.__version__)
```

If it fails with `No module named 'pkg_resources'`, run:

```python
!python -m pip install "setuptools<70"
```

If it fails with `nvcc was not found`, switch to a GPU runtime and rerun:

```python
!which nvcc
!nvidia-smi
```

If whole-image inference fails with CUDA internal errors or OOM, lower `--max-side`:

```python
%%bash
python scripts/evssm_infer_whole_image.py \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output-dir results/EVSSM/RealBlurJ_whole_1024 \
  --limit 1 \
  --max-side 1024
```

If the failure says it tried to allocate several GiB and only a few GiB were free, retry whole-image inference with half precision:

```python
%%bash
python scripts/evssm_infer_whole_image.py \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output-dir results/EVSSM/RealBlurJ_whole_original_fp16 \
  --limit 1 \
  --precision fp16
```
