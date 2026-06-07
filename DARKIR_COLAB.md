# DarkIR On Google Colab

This guide runs `methods/DarkIR` on this project's `img/` folder.

All command blocks are intended for Colab notebook cells. Blocks starting with `%%bash` must have `%%bash` as the first line of the cell.

## 1. Enable GPU

In Colab:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

Verify:

```python
!nvidia-smi
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
```

## 2. Pull The Project And Submodules

If the project is already cloned:

```python
%cd /content/image_processing_final_project
!git pull
!git submodule update --init --recursive methods/DarkIR
```

If cloning fresh:

```python
%cd /content
REPO_URL = "PASTE_YOUR_REPO_URL_HERE"
assert REPO_URL != "PASTE_YOUR_REPO_URL_HERE", "Set REPO_URL before running this cell."
!git clone --recursive {REPO_URL} image_processing_final_project
%cd /content/image_processing_final_project
!git submodule update --init --recursive methods/DarkIR
```

Check:

```python
!test -f methods/DarkIR/archs/DarkIR.py && echo "DarkIR submodule ok"
!find img -maxdepth 1 -type f | head
```

## 3. Install DarkIR Dependencies

Run:

```python
!bash scripts/setup_darkir_colab.sh
```

This keeps Colab's active PyTorch installation and installs the extra packages needed by DarkIR. It does not force the exact `torch==2.5.1` from the official DarkIR requirements, because changing PyTorch inside Colab can break the GPU runtime.

## 4. Add DarkIR Checkpoints From Google Drive

DarkIR checkpoints are not stored inside `img/`. Put them in a Drive folder, then copy them into the Colab project.

Mount Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Set your Drive checkpoint folder and copy files:

```python
DARKIR_DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/DarkIR_checkpoints"

!mkdir -p methods/DarkIR/models
!cp "{DARKIR_DRIVE_CHECKPOINT_DIR}"/*.pt methods/DarkIR/models/
!cp "{DARKIR_DRIVE_CHECKPOINT_DIR}"/*.pth methods/DarkIR/models/ 2>/dev/null || true
!ls -lh methods/DarkIR/models
```

Common checkpoint names used by DarkIR configs:

```text
methods/DarkIR/models/DarkIR_64width.pt
methods/DarkIR/models/DarkIR_384.pt
methods/DarkIR/models/DarkIR_1k_cr_mt.pt
methods/DarkIR/models/darkir_1k_cr_mt.pt
methods/DarkIR/models/RetinexFormer_LOL_v2_real.pth
```

The default config in this guide is:

```text
methods/DarkIR/options/inference/LOLBlur.yml
```

It expects:

```text
methods/DarkIR/models/DarkIR_64width.pt
```

## 5. Dry Run

Whole-image dry run:

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/LOLBlur_whole_dryrun \
  --config methods/DarkIR/options/inference/LOLBlur.yml \
  --start-index 1 \
  --limit 1 \
  --dry-run
```

Test the 8th image:

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/LOLBlur_whole_from08_dryrun \
  --config methods/DarkIR/options/inference/LOLBlur.yml \
  --start-index 8 \
  --limit 1 \
  --dry-run
```

## 6. Whole-Image Inference

Run one image at original resolution:

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/LOLBlur_whole_original \
  --config methods/DarkIR/options/inference/LOLBlur.yml \
  --start-index 1 \
  --limit 1
```

Run the 8th image at original resolution:

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/LOLBlur_whole_original_from08 \
  --config methods/DarkIR/options/inference/LOLBlur.yml \
  --start-index 8 \
  --limit 1
```

If whole-image inference runs out of memory, try half precision:

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/LOLBlur_whole_original_fp16_from08 \
  --config methods/DarkIR/options/inference/LOLBlur.yml \
  --start-index 8 \
  --limit 1 \
  --precision fp16
```

Or resize the longest side:

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/LOLBlur_whole_2048_from08 \
  --config methods/DarkIR/options/inference/LOLBlur.yml \
  --start-index 8 \
  --limit 1 \
  --max-side 2048
```

## 7. Tiled Inference

For large images, use tiled inference. `crop` blending is recommended because it gives low weight to tile borders and prefers the center of each tile.

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/LOLBlur_tile2048_crop_from08 \
  --config methods/DarkIR/options/inference/LOLBlur.yml \
  --tile-size 2048 \
  --overlap 320 \
  --blend crop \
  --start-index 8 \
  --limit 1
```

For A100 80GB, try larger tiles:

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/LOLBlur_tile3072_crop_from08 \
  --config methods/DarkIR/options/inference/LOLBlur.yml \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop \
  --start-index 8 \
  --limit 1
```

Run all images with tiled inference:

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/LOLBlur_tile2048_crop_all \
  --config methods/DarkIR/options/inference/LOLBlur.yml \
  --tile-size 2048 \
  --overlap 320 \
  --blend crop
```

## 8. Use A Different Checkpoint Or Config

You can override the checkpoint path directly:

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/custom_checkpoint_from08 \
  --config methods/DarkIR/options/inference/LOLBlur.yml \
  --checkpoint methods/DarkIR/models/DarkIR_64width.pt \
  --start-index 8 \
  --limit 1
```

Other official inference configs:

```text
methods/DarkIR/options/inference/LOLBlur.yml
methods/DarkIR/options/inference/real_lsrw.yml
methods/DarkIR/options/inference/ExDark.yml
```

Use a checkpoint whose architecture matches the config. For example, `LOLBlur.yml` uses `width: 64`, while `real_lsrw.yml` uses `width: 32`.

## 9. Download Results

```python
!zip -r darkir_results.zip results/DarkIR
```

```python
from google.colab import files
files.download("darkir_results.zip")
```

## Troubleshooting

If the checkpoint file is missing:

```python
!ls -lh methods/DarkIR/models
```

If the checkpoint does not match the config, use the matching config/checkpoint pair. A `width: 64` checkpoint will not load cleanly into a `width: 32` model.

If CUDA runs out of memory, try one of:

```text
--precision fp16
--max-side 2048
--tile-size 2048 --overlap 320 --blend crop
```
