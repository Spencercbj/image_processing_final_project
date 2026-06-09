# DarkIR 接 EVSSM 的 Colab 完整流程

這份文件說明如何在 Google Colab 上執行：

```text
原圖 img/
  -> DarkIR
  -> alpha 混回原圖，降低過亮感
  -> EVSSM
  -> 最終結果下載
```

文件中有兩種接法：

```text
baseline: 原圖 -> DarkIR -> alpha 混回原圖 -> EVSSM
高亮保護版: 原圖產生 soft mask -> DarkIR 前壓高亮 -> DarkIR -> 高亮區混回原圖 -> EVSSM
```

建議先測第 8 張，確認效果後再跑全部圖片。

所有程式區塊都可以直接貼到 Colab notebook cell 執行。若 cell 第一行是 `%%bash`，它必須放在該 cell 的第一行。

## 1. 開啟 GPU

在 Colab 選：

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

確認 GPU：

```python
!nvidia-smi
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
```

## 2. 進入專案並更新 submodules

如果專案已經在 Colab：

```python
%cd /content/image_processing_final_project
!git pull
!git submodule update --init --recursive
```

如果要重新 clone：

```python
%cd /content
REPO_URL = "PASTE_YOUR_REPO_URL_HERE"
assert REPO_URL != "PASTE_YOUR_REPO_URL_HERE", "Set REPO_URL before running this cell."
!git clone --recursive {REPO_URL} image_processing_final_project
%cd /content/image_processing_final_project
!git submodule update --init --recursive
```

確認檔案存在：

```python
!test -f methods/DarkIR/archs/DarkIR.py && echo "DarkIR ok"
!test -f methods/EVSSM/models/EVSSM.py && echo "EVSSM ok"
!find img -maxdepth 1 -type f | head
```

## 3. 安裝環境

先安裝 EVSSM 環境：

```python
!bash scripts/setup_colab.sh
```

再安裝 DarkIR 額外依賴：

```python
!bash scripts/setup_darkir_colab.sh
```

檢查兩邊 import：

```python
import sys, torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())

import mamba_ssm
print("mamba_ssm:", mamba_ssm.__version__)

sys.path.insert(0, "/content/image_processing_final_project/methods/DarkIR/archs")
from DarkIR import DarkIR
print("DarkIR import ok")
```

如果環境真的衝突，最穩做法是分兩個 Colab runtime：

```text
Runtime A: 跑 DarkIR，結果存到 Google Drive
Runtime B: 從 Drive 讀 DarkIR 結果，接著跑 EVSSM
```

但目前腳本設計上 DarkIR 不會重裝 PyTorch，通常可以和 EVSSM 共用同一個 runtime。

## 4. 從 Google Drive 複製權重

掛載 Drive：

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 4.1 DarkIR 權重

本流程假設你使用：

```text
DarkIR_384.pt
```

並搭配：

```text
methods/DarkIR/options/inference/real_lsrw.yml
```

假設檔案在：

```text
MyDrive/DarkIR_checkpoints/DarkIR_384.pt
```

複製到 Colab：

```python
%cd /content/image_processing_final_project

!mkdir -p methods/DarkIR/models
!cp /content/drive/MyDrive/DarkIR_checkpoints/DarkIR_384.pt methods/DarkIR/models/
!ls -lh methods/DarkIR/models
```

### 4.2 EVSSM 權重

假設 EVSSM 權重在：

```text
MyDrive/EVSSM_checkpoints/
```

其中包含：

```text
net_g_realblur_j.pth
net_g_realblur_r.pth
net_g_GoPro.pth
```

複製到 Colab：

```python
%cd /content/image_processing_final_project

!mkdir -p checkpoints
!cp /content/drive/MyDrive/EVSSM_checkpoints/net_g_realblur_j.pth checkpoints/
!cp /content/drive/MyDrive/EVSSM_checkpoints/net_g_realblur_r.pth checkpoints/
!cp /content/drive/MyDrive/EVSSM_checkpoints/net_g_GoPro.pth checkpoints/
!ls -lh checkpoints
```

如果你的 Drive 路徑不同，請修改上面的 `/content/drive/MyDrive/...`。

## 5. Dry Run

先確認 DarkIR 能讀到第 8 張：

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/Pipeline/01_darkir_from08 \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop \
  --start-index 8 \
  --limit 1 \
  --dry-run
```

## 6. 單張測試：第 8 張

### 6.1 跑 DarkIR

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/Pipeline/01_darkir_from08 \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop \
  --start-index 8 \
  --limit 1
```

如果 OOM，改用：

```text
--tile-size 2048 --overlap 320
```

如果 A100 80GB 很空，可以試：

```text
--tile-size 4096 --overlap 640
```

### 6.2 將 DarkIR 結果混回原圖

先試 `alpha=0.7`：

```python
%%bash
python scripts/blend_darkir_for_evssm.py \
  --original img \
  --darkir results/Pipeline/01_darkir_from08 \
  --output results/Pipeline/02_darkir_alpha07_for_evssm_from08 \
  --alpha 0.7 \
  --start-index 8 \
  --limit 1
```

建議比較：

```text
alpha 0.6: 更接近原圖，壓亮度更強
alpha 0.7: 折衷，建議先看
alpha 0.8: 更接近 DarkIR，增亮/修復效果更強
```

如果要產生 `alpha=0.6`：

```python
%%bash
python scripts/blend_darkir_for_evssm.py \
  --original img \
  --darkir results/Pipeline/01_darkir_from08 \
  --output results/Pipeline/02_darkir_alpha06_for_evssm_from08 \
  --alpha 0.6 \
  --start-index 8 \
  --limit 1
```

### 6.3 用混合結果跑 EVSSM

```python
%%bash
python scripts/evssm_infer_folder.py \
  --input results/Pipeline/02_darkir_alpha07_for_evssm_from08 \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output results/Pipeline/03_darkir_alpha07_evssm_from08 \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop \
  --limit 1
```

注意：這個 input 資料夾只包含第 8 張，所以 EVSSM 這裡不用 `--start-index 8`。

## 7. 跑全部圖片

### 7.1 跑全部 DarkIR

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/Pipeline/01_darkir_all \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop
```

### 7.2 全部混回原圖

```python
%%bash
python scripts/blend_darkir_for_evssm.py \
  --original img \
  --darkir results/Pipeline/01_darkir_all \
  --output results/Pipeline/02_darkir_alpha07_for_evssm_all \
  --alpha 0.7
```

### 7.3 跑全部 EVSSM

```python
%%bash
python scripts/evssm_infer_folder.py \
  --input results/Pipeline/02_darkir_alpha07_for_evssm_all \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output results/Pipeline/03_darkir_alpha07_evssm_all \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop
```

## 8. 高亮保護版：DarkIR 接 EVSSM

這個版本會用原圖產生高亮 soft mask。DarkIR 前先把高亮區稍微壓暗，DarkIR 後再用同一張 mask 把高亮區混回原圖，避免 DarkIR 把燈、反光、白色區域推到過曝。

預設先用：

```text
--threshold 0.82 --softness 0.12 --highlight-scale 0.75
```

### 8.1 第 8 張：產生 DarkIR protected input

```python
%%bash
python scripts/protect_highlights_for_darkir.py \
  --mode prepare \
  --original img \
  --output results/PipelineHighlight/01_darkir_protected_input_from08 \
  --threshold 0.82 \
  --softness 0.12 \
  --highlight-scale 0.75 \
  --start-index 8 \
  --limit 1
```

### 8.2 第 8 張：用 protected input 跑 DarkIR

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input results/PipelineHighlight/01_darkir_protected_input_from08 \
  --output results/PipelineHighlight/02_darkir_from08 \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop \
  --limit 1
```

### 8.3 第 8 張：用高亮 mask 合成 DarkIR 與原圖

```python
%%bash
python scripts/protect_highlights_for_darkir.py \
  --mode composite \
  --original img \
  --darkir results/PipelineHighlight/02_darkir_from08 \
  --output results/PipelineHighlight/03_darkir_highlight_protected_for_evssm_from08 \
  --threshold 0.82 \
  --softness 0.12 \
  --highlight-scale 0.75 \
  --start-index 8 \
  --limit 1
```

### 8.4 第 8 張：把合成結果送進 EVSSM

```python
%%bash
python scripts/evssm_infer_folder.py \
  --input results/PipelineHighlight/03_darkir_highlight_protected_for_evssm_from08 \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output results/PipelineHighlight/04_darkir_highlight_protected_evssm_from08 \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop \
  --limit 1
```

### 8.5 跑全部圖片

```python
%%bash
python scripts/protect_highlights_for_darkir.py \
  --mode prepare \
  --original img \
  --output results/PipelineHighlight/01_darkir_protected_input_all \
  --threshold 0.82 \
  --softness 0.12 \
  --highlight-scale 0.75
```

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input results/PipelineHighlight/01_darkir_protected_input_all \
  --output results/PipelineHighlight/02_darkir_all \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop
```

```python
%%bash
python scripts/protect_highlights_for_darkir.py \
  --mode composite \
  --original img \
  --darkir results/PipelineHighlight/02_darkir_all \
  --output results/PipelineHighlight/03_darkir_highlight_protected_for_evssm_all \
  --threshold 0.82 \
  --softness 0.12 \
  --highlight-scale 0.75
```

```python
%%bash
python scripts/evssm_infer_folder.py \
  --input results/PipelineHighlight/03_darkir_highlight_protected_for_evssm_all \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output results/PipelineHighlight/04_darkir_highlight_protected_evssm_all \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop
```

如果高亮還是太亮，可以試：

```text
--threshold 0.78 --highlight-scale 0.65
```

如果高亮區被壓得太暗，可以試：

```text
--threshold 0.86 --highlight-scale 0.85
```

## 9. 下載結果

下載單張測試結果：

```python
from google.colab import files
files.download("results/Pipeline/03_darkir_alpha07_evssm_from08/08_KFC_Rider_Rainy_Night_Delivery.jpg")
```

壓縮全部 pipeline 結果：

```python
!zip -r darkir_evssm_pipeline_results.zip results/Pipeline
```

下載 zip：

```python
from google.colab import files
files.download("darkir_evssm_pipeline_results.zip")
```

高亮保護版結果可以這樣下載：

```python
!zip -r darkir_evssm_highlight_pipeline_results.zip results/PipelineHighlight
from google.colab import files
files.download("darkir_evssm_highlight_pipeline_results.zip")
```

## 10. 常見問題

### DarkIR 結果太亮

降低 alpha：

```text
--alpha 0.6
```

### DarkIR 效果被削弱太多

提高 alpha：

```text
--alpha 0.8
```

### DarkIR 或 EVSSM OOM

先降 tile：

```text
--tile-size 2048 --overlap 320
```

或用較小 tile：

```text
--tile-size 1536 --overlap 256
```

### 想從任意圖片開始

兩個 inference 腳本都支援：

```text
--start-index N
--limit M
```

例如第 8 張：

```text
--start-index 8 --limit 1
```
