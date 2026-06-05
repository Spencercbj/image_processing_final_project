# Google Colab Setup

This guide runs the EVSSM inference scripts in Google Colab without Conda.

Colab runtimes change over time, so the notebook cells install the Python packages inside the active runtime instead of relying on a prebuilt local environment.

## 1. Enable GPU

In Colab:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

Then verify:

```python
!nvidia-smi
!python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY
```

## 2. Get The Project Into Colab

### Option A: Clone From Git

Use this if the repository is on GitHub:

```python
%cd /content
!git clone --recursive <YOUR_REPO_URL> image_processing_final_project
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
img/
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

## 4. Install EVSSM Dependencies

Run:

```python
!bash scripts/setup_colab.sh
```

This installs build tools, project dependencies, `causal-conv1d==1.4.0`, and `mamba-ssm==2.2.2` against Colab's active CUDA-enabled PyTorch.

If Colab crashes or asks for a runtime restart after dependency installation, restart the runtime and run the cells again from step 1. Pip-built CUDA extensions may need a clean runtime after installation.

## 5. Run A Dry Run

Tiled inference dry run:

```python
!python scripts/evssm_infer_folder.py --dry-run --limit 1
```

Whole-image dry run:

```python
!python scripts/evssm_infer_whole_image.py \
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
!python scripts/evssm_infer_folder.py \
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
!python scripts/evssm_infer_folder.py \
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
!python scripts/evssm_infer_whole_image.py \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output-dir results/EVSSM/RealBlurJ_whole_1536 \
  --limit 1 \
  --max-side 1536
```

GoPro checkpoint:

```python
!python scripts/evssm_infer_whole_image.py \
  --checkpoint checkpoints/net_g_GoPro.pth \
  --output-dir results/EVSSM/GoPro_whole_1536 \
  --limit 1 \
  --max-side 1536
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
!python scripts/evssm_infer_whole_image.py \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output-dir results/EVSSM/RealBlurJ_whole_1024 \
  --limit 1 \
  --max-side 1024
```
