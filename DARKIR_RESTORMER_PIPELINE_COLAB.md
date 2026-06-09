# DarkIR 接 Restormer 再接 DiffBIR 的 Colab 完整流程
#
這份文件說明如何在 Google Colab 上執行：

```text
原圖 img/
  -> DarkIR
  -> alpha 混回原圖，降低過亮感
  -> Restormer
  -> DiffBIR diffusion 後處理
  -> 最終結果下載
```

文件中有兩種接法：

```text
baseline: 原圖 -> DarkIR -> alpha 混回原圖 -> Restormer Motion Deblurring -> DiffBIR SR x1 去模糊/修復
高亮保護版: 原圖產生 soft mask -> DarkIR 前壓高亮 -> DarkIR -> 高亮區混回原圖 -> Restormer Motion Deblurring -> DiffBIR SR x1 去模糊/修復
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

## 2. 進入專案、切換分支並更新 submodules

先設定你要使用的分支名稱：

```python
BRANCH_NAME = "PASTE_BRANCH_NAME_HERE"
assert BRANCH_NAME != "PASTE_BRANCH_NAME_HERE", "Set BRANCH_NAME before running this cell."
```

如果專案已經在 Colab，先進入既有 repo，再切到指定分支：

```python
%cd /content/image_processing_final_project
!git fetch origin
!git switch {BRANCH_NAME}
!git pull origin {BRANCH_NAME}
!git submodule update --init --recursive
```

如果要重新 clone，新 repo 會在 `git clone --branch {BRANCH_NAME}` 這一步直接切到指定分支，不需要再另外 `git switch`：

```python
%cd /content
REPO_URL = "PASTE_YOUR_REPO_URL_HERE"
assert REPO_URL != "PASTE_YOUR_REPO_URL_HERE", "Set REPO_URL before running this cell."
!git clone --recursive --branch {BRANCH_NAME} {REPO_URL} image_processing_final_project
%cd /content/image_processing_final_project
!git submodule update --init --recursive
```

確認檔案存在：

```python
!test -f methods/DarkIR/archs/DarkIR.py && echo "DarkIR ok"
!test -f methods/Restormer/demo.py && echo "Restormer ok"
!test -f scripts/setup_diffbir_colab.sh && echo "DiffBIR setup script ok"
!find img -maxdepth 1 -type f | head
```

## 3. 安裝環境

先安裝 DarkIR 依賴：

```python
!bash scripts/setup_darkir_colab.sh
```

再安裝 Restormer demo 需要的依賴：

```python
!bash scripts/setup_restormer_colab.sh
```

最後安裝 DiffBIR。這一步會 clone `XPixelGroup/DiffBIR` 到 `methods/DiffBIR`，並安裝 DiffBIR 需要的套件。Colab 通常已經有 CUDA 版 PyTorch，所以這個 setup script 不會覆蓋 torch：

```python
!bash scripts/setup_diffbir_colab.sh
```

檢查三邊 import：

```python
import sys, torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())

sys.path.insert(0, "/content/image_processing_final_project/methods/DarkIR/archs")
from DarkIR import DarkIR
print("DarkIR import ok")

from runpy import run_path
load_arch = run_path("/content/image_processing_final_project/methods/Restormer/basicsr/models/archs/restormer_arch.py")
print("Restormer import ok", load_arch["Restormer"])

sys.path.insert(0, "/content/image_processing_final_project/methods/DiffBIR")
from diffbir.inference import BIDInferenceLoop
print("DiffBIR import ok", BIDInferenceLoop)
```

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

### 4.2 Restormer 權重

建議先用 Restormer 的 Motion Deblurring 權重接在 DarkIR 後面：

```text
motion_deblurring.pth
```

Restormer demo 預期它放在：

```text
methods/Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth
```

假設你的權重在：

```text
MyDrive/Restormer_checkpoints/motion_deblurring.pth
```

複製到 Colab：

```python
%cd /content/image_processing_final_project

!mkdir -p methods/Restormer/Motion_Deblurring/pretrained_models
!cp /content/drive/MyDrive/Restormer_checkpoints/motion_deblurring.pth methods/Restormer/Motion_Deblurring/pretrained_models/
!ls -lh methods/Restormer/Motion_Deblurring/pretrained_models
```

如果你想試 Restormer Real Denoising，權重檔名和位置是：

```text
methods/Restormer/Denoising/pretrained_models/real_denoising.pth
```

對應 task 會是：

```text
Real_Denoising
```

### 4.3 DiffBIR 權重

DiffBIR 的 pretrained weights 會在第一次執行 `inference.py` 時自動下載，不需要先從 Google Drive 複製。

建議 Colab 先保留足夠空間，因為 DiffBIR 會下載 Stable Diffusion、ControlNet、SwinIR/SCUNet 相關權重。

## 5. 單張測試：第 8 張

### 5.1 跑 DarkIR

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/PipelineRestormer/01_darkir_from08 \
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

### 5.2 將 DarkIR 結果混回原圖

先試 `alpha=0.7`：

```python
%%bash
python scripts/blend_with_original.py \
  --original img \
  --darkir results/PipelineRestormer/01_darkir_from08 \
  --output results/PipelineRestormer/02_darkir_alpha07_for_restormer_from08 \
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

### 5.3 用混合結果跑 Restormer Motion Deblurring

Restormer demo 要在 `methods/Restormer` 目錄下執行，所以這個 cell 會先 `cd` 進去。`input_dir` 和 `result_dir` 使用絕對路徑。

```python
%%bash
cd /content/image_processing_final_project/methods/Restormer
python demo.py \
  --task Motion_Deblurring \
  --input_dir /content/image_processing_final_project/results/PipelineRestormer/02_darkir_alpha07_for_restormer_from08 \
  --result_dir /content/image_processing_final_project/results/PipelineRestormer/03_darkir_alpha07_restormer_from08 \
  --tile 3072 \
  --tile_overlap 512
```

Restormer 會把結果存在 task 子資料夾中：

```text
results/PipelineRestormer/03_darkir_alpha07_restormer_from08/Motion_Deblurring/
```

如果 OOM，改成：

```text
--tile 2048 --tile_overlap 320
```

### 5.4 把 Restormer 結果送進 DiffBIR

這裡用 DiffBIR 的 `sr` task 做盲影像修復，並設定 `--upscale 1`。因為第 8 張原始尺寸約是 `5304x7952`，DiffBIR 直接吃原尺寸會在 Colab 被 killed，所以先把 Restormer 輸出縮到 `max-side 2048` 給 DiffBIR，最後再 resize 回 Restormer 輸出尺寸。

先準備 DiffBIR input：

```python
%%bash
python scripts/resize_for_diffbir.py prepare \
  --input results/PipelineRestormer/03_darkir_alpha07_restormer_from08/Motion_Deblurring \
  --output results/PipelineRestormer/04a_diffbir_input_from08 \
  --max-side 2048
```

再跑 DiffBIR。`--captioner none` 比 LLaVA 省 VRAM，也比較保守：

```python
%%bash
cd /content/image_processing_final_project/methods/DiffBIR
python -u inference.py \
  --task sr \
  --upscale 1 \
  --version v2.1 \
  --captioner none \
  --pos_prompt '' \
  --neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures, artifacts' \
  --cfg_scale 4 \
  --noise_aug 0 \
  --steps 8 \
  --input /content/image_processing_final_project/results/PipelineRestormer/04a_diffbir_input_from08 \
  --output /content/image_processing_final_project/results/PipelineRestormer/04b_diffbir_small_from08 \
  --device cuda \
  --precision fp16 \
  --cleaner_tiled \
  --cleaner_tile_size 128 \
  --cleaner_tile_stride 64 \
  --vae_encoder_tiled \
  --vae_encoder_tile_size 128 \
  --vae_decoder_tiled \
  --vae_decoder_tile_size 128 \
  --cldm_tiled \
  --cldm_tile_size 256 \
  --cldm_tile_stride 128
```

最後把 DiffBIR 結果 resize 回 Restormer 輸出尺寸：

```python
%%bash
python scripts/resize_for_diffbir.py restore \
  --diffbir results/PipelineRestormer/04b_diffbir_small_from08 \
  --reference results/PipelineRestormer/03_darkir_alpha07_restormer_from08/Motion_Deblurring \
  --output results/PipelineRestormer/04_darkir_alpha07_restormer_diffbir_from08
```

最終輸出會在：

```text
results/PipelineRestormer/04_darkir_alpha07_restormer_diffbir_from08/
```

## 6. 跑全部圖片

### 6.1 跑全部 DarkIR

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/PipelineRestormer/01_darkir_all \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop
```

### 6.2 全部混回原圖

```python
%%bash
python scripts/blend_with_original.py \
  --original img \
  --darkir results/PipelineRestormer/01_darkir_all \
  --output results/PipelineRestormer/02_darkir_alpha07_for_restormer_all \
  --alpha 0.7
```

### 6.3 跑全部 Restormer

```python
%%bash
cd /content/image_processing_final_project/methods/Restormer
python demo.py \
  --task Motion_Deblurring \
  --input_dir /content/image_processing_final_project/results/PipelineRestormer/02_darkir_alpha07_for_restormer_all \
  --result_dir /content/image_processing_final_project/results/PipelineRestormer/03_darkir_alpha07_restormer_all \
  --tile 3072 \
  --tile_overlap 512
```

### 6.4 跑全部 DiffBIR

先準備 DiffBIR input：

```python
%%bash
python scripts/resize_for_diffbir.py prepare \
  --input results/PipelineRestormer/03_darkir_alpha07_restormer_all/Motion_Deblurring \
  --output results/PipelineRestormer/04a_diffbir_input_all \
  --max-side 2048
```

再跑 DiffBIR：

```python
%%bash
cd /content/image_processing_final_project/methods/DiffBIR
python -u inference.py \
  --task sr \
  --upscale 1 \
  --version v2.1 \
  --captioner none \
  --pos_prompt '' \
  --neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures, artifacts' \
  --cfg_scale 4 \
  --noise_aug 0 \
  --steps 8 \
  --input /content/image_processing_final_project/results/PipelineRestormer/04a_diffbir_input_all \
  --output /content/image_processing_final_project/results/PipelineRestormer/04b_diffbir_small_all \
  --device cuda \
  --precision fp16 \
  --cleaner_tiled \
  --cleaner_tile_size 128 \
  --cleaner_tile_stride 64 \
  --vae_encoder_tiled \
  --vae_encoder_tile_size 128 \
  --vae_decoder_tiled \
  --vae_decoder_tile_size 128 \
  --cldm_tiled \
  --cldm_tile_size 256 \
  --cldm_tile_stride 128
```

最後 resize 回 Restormer 輸出尺寸：

```python
%%bash
python scripts/resize_for_diffbir.py restore \
  --diffbir results/PipelineRestormer/04b_diffbir_small_all \
  --reference results/PipelineRestormer/03_darkir_alpha07_restormer_all/Motion_Deblurring \
  --output results/PipelineRestormer/04_darkir_alpha07_restormer_diffbir_all
```

## 7. 高亮保護版：DarkIR 接 Restormer

這個版本會用原圖產生高亮 soft mask。DarkIR 前先把高亮區稍微壓暗，DarkIR 後再用同一張 mask 把高亮區混回原圖，最後 Restormer 的 input 會吃 composite output，不直接吃 DarkIR output。

預設先用：

```text
--threshold 0.82 --softness 0.12 --highlight-scale 0.75
```

### 7.1 第 8 張：產生 DarkIR protected input

```python
%%bash
python scripts/protect_highlights_for_darkir.py \
  --mode prepare \
  --original img \
  --output results/PipelineRestormerHighlight/01_darkir_protected_input_from08 \
  --threshold 0.82 \
  --softness 0.12 \
  --highlight-scale 0.75 \
  --start-index 8 \
  --limit 1
```

### 7.2 第 8 張：用 protected input 跑 DarkIR

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input results/PipelineRestormerHighlight/01_darkir_protected_input_from08 \
  --output results/PipelineRestormerHighlight/02_darkir_from08 \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop \
  --limit 1
```

### 7.3 第 8 張：用高亮 mask 合成 DarkIR 與原圖

```python
%%bash
python scripts/protect_highlights_for_darkir.py \
  --mode composite \
  --original img \
  --darkir results/PipelineRestormerHighlight/02_darkir_from08 \
  --output results/PipelineRestormerHighlight/03_darkir_highlight_protected_for_restormer_from08 \
  --threshold 0.82 \
  --softness 0.12 \
  --highlight-scale 0.75 \
  --start-index 8 \
  --limit 1
```

### 7.4 第 8 張：把合成結果送進 Restormer Motion Deblurring

```python
%%bash
cd /content/image_processing_final_project/methods/Restormer
python demo.py \
  --task Motion_Deblurring \
  --input_dir /content/image_processing_final_project/results/PipelineRestormerHighlight/03_darkir_highlight_protected_for_restormer_from08 \
  --result_dir /content/image_processing_final_project/results/PipelineRestormerHighlight/04_darkir_highlight_protected_restormer_from08 \
  --tile 3072 \
  --tile_overlap 512
```

輸出會在：

```text
results/PipelineRestormerHighlight/04_darkir_highlight_protected_restormer_from08/Motion_Deblurring/
```

### 7.5 第 8 張：把 Restormer 結果送進 DiffBIR

先準備 DiffBIR input：

```python
%%bash
python scripts/resize_for_diffbir.py prepare \
  --input results/PipelineRestormerHighlight/04_darkir_highlight_protected_restormer_from08/Motion_Deblurring \
  --output results/PipelineRestormerHighlight/05a_diffbir_input_from08 \
  --max-side 2048
```

再跑 DiffBIR：

```python
%%bash
cd /content/image_processing_final_project/methods/DiffBIR
python -u inference.py \
  --task sr \
  --upscale 1 \
  --version v2.1 \
  --captioner none \
  --pos_prompt '' \
  --neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures, artifacts' \
  --cfg_scale 4 \
  --noise_aug 0 \
  --steps 8 \
  --input /content/image_processing_final_project/results/PipelineRestormerHighlight/05a_diffbir_input_from08 \
  --output /content/image_processing_final_project/results/PipelineRestormerHighlight/05b_diffbir_small_from08 \
  --device cuda \
  --precision fp16 \
  --cleaner_tiled \
  --cleaner_tile_size 128 \
  --cleaner_tile_stride 64 \
  --vae_encoder_tiled \
  --vae_encoder_tile_size 128 \
  --vae_decoder_tiled \
  --vae_decoder_tile_size 128 \
  --cldm_tiled \
  --cldm_tile_size 256 \
  --cldm_tile_stride 128
```

最後 resize 回 Restormer 輸出尺寸：

```python
%%bash
python scripts/resize_for_diffbir.py restore \
  --diffbir results/PipelineRestormerHighlight/05b_diffbir_small_from08 \
  --reference results/PipelineRestormerHighlight/04_darkir_highlight_protected_restormer_from08/Motion_Deblurring \
  --output results/PipelineRestormerHighlight/05_darkir_highlight_protected_restormer_diffbir_from08
```

輸出會在：

```text
results/PipelineRestormerHighlight/05_darkir_highlight_protected_restormer_diffbir_from08/
```

### 7.6 跑全部圖片

```python
%%bash
python scripts/protect_highlights_for_darkir.py \
  --mode prepare \
  --original img \
  --output results/PipelineRestormerHighlight/01_darkir_protected_input_all \
  --threshold 0.82 \
  --softness 0.12 \
  --highlight-scale 0.75
```

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input results/PipelineRestormerHighlight/01_darkir_protected_input_all \
  --output results/PipelineRestormerHighlight/02_darkir_all \
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
  --darkir results/PipelineRestormerHighlight/02_darkir_all \
  --output results/PipelineRestormerHighlight/03_darkir_highlight_protected_for_restormer_all \
  --threshold 0.82 \
  --softness 0.12 \
  --highlight-scale 0.75
```

```python
%%bash
cd /content/image_processing_final_project/methods/Restormer
python demo.py \
  --task Motion_Deblurring \
  --input_dir /content/image_processing_final_project/results/PipelineRestormerHighlight/03_darkir_highlight_protected_for_restormer_all \
  --result_dir /content/image_processing_final_project/results/PipelineRestormerHighlight/04_darkir_highlight_protected_restormer_all \
  --tile 3072 \
  --tile_overlap 512
```

```python
%%bash
python scripts/resize_for_diffbir.py prepare \
  --input results/PipelineRestormerHighlight/04_darkir_highlight_protected_restormer_all/Motion_Deblurring \
  --output results/PipelineRestormerHighlight/05a_diffbir_input_all \
  --max-side 2048
```

```python
%%bash
cd /content/image_processing_final_project/methods/DiffBIR
python -u inference.py \
  --task sr \
  --upscale 1 \
  --version v2.1 \
  --captioner none \
  --pos_prompt '' \
  --neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures, artifacts' \
  --cfg_scale 4 \
  --noise_aug 0 \
  --steps 8 \
  --input /content/image_processing_final_project/results/PipelineRestormerHighlight/05a_diffbir_input_all \
  --output /content/image_processing_final_project/results/PipelineRestormerHighlight/05b_diffbir_small_all \
  --device cuda \
  --precision fp16 \
  --cleaner_tiled \
  --cleaner_tile_size 128 \
  --cleaner_tile_stride 64 \
  --vae_encoder_tiled \
  --vae_encoder_tile_size 128 \
  --vae_decoder_tiled \
  --vae_decoder_tile_size 128 \
  --cldm_tiled \
  --cldm_tile_size 256 \
  --cldm_tile_stride 128
```

```python
%%bash
python scripts/resize_for_diffbir.py restore \
  --diffbir results/PipelineRestormerHighlight/05b_diffbir_small_all \
  --reference results/PipelineRestormerHighlight/04_darkir_highlight_protected_restormer_all/Motion_Deblurring \
  --output results/PipelineRestormerHighlight/05_darkir_highlight_protected_restormer_diffbir_all
```

如果高亮還是太亮，可以試：

```text
--threshold 0.78 --highlight-scale 0.65
```

如果高亮區被壓得太暗，可以試：

```text
--threshold 0.86 --highlight-scale 0.85
```

## 8. 可選：改用 Restormer Real Denoising

如果你覺得 DarkIR 後面主要剩雜訊，不是模糊，可以試 Restormer 的 `Real_Denoising`。

先放權重：

```text
methods/Restormer/Denoising/pretrained_models/real_denoising.pth
```

然後跑：

```python
%%bash
cd /content/image_processing_final_project/methods/Restormer
python demo.py \
  --task Real_Denoising \
  --input_dir /content/image_processing_final_project/results/PipelineRestormer/02_darkir_alpha07_for_restormer_from08 \
  --result_dir /content/image_processing_final_project/results/PipelineRestormer/03_darkir_alpha07_restormer_denoise_from08 \
  --tile 3072 \
  --tile_overlap 512
```

輸出會在：

```text
results/PipelineRestormer/03_darkir_alpha07_restormer_denoise_from08/Real_Denoising/
```

## 9. 下載結果

下載單張測試的 DiffBIR 最終結果：

```python
from google.colab import files
files.download("/content/image_processing_final_project/results/PipelineRestormer/04_darkir_alpha07_restormer_diffbir_from08/08_KFC_Rider_Rainy_Night_Delivery.png")
```

如果不確定輸出檔名，先列出：

```python
!find /content/image_processing_final_project/results/PipelineRestormer -type f | head -50
```

壓縮全部 pipeline 結果：

```python
%cd /content/image_processing_final_project
!zip -r darkir_restormer_diffbir_pipeline_results.zip results/PipelineRestormer
```

下載 zip：

```python
from google.colab import files
files.download("darkir_restormer_diffbir_pipeline_results.zip")
```

高亮保護版結果可以這樣下載：

```python
%cd /content/image_processing_final_project
!zip -r darkir_restormer_diffbir_highlight_pipeline_results.zip results/PipelineRestormerHighlight
from google.colab import files
files.download("darkir_restormer_diffbir_highlight_pipeline_results.zip")
```

## 10. 常見問題

### CUDA is not available

這代表目前 Colab runtime 沒有拿到 GPU，不是權重或圖片路徑沒有用到。

先回到第 1 步確認：

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

切換後建議重新執行：

```python
!nvidia-smi
import torch
print("cuda available:", torch.cuda.is_available())
```

如果 `cuda available` 還是 `False`，代表目前 runtime 仍然不是 GPU，請重新連線或重開 runtime。

### Restormer 找不到權重

Motion Deblurring 權重必須在：

```text
methods/Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth
```

確認：

```python
!ls -lh methods/Restormer/Motion_Deblurring/pretrained_models
```

### Restormer 輸出在哪裡

Restormer demo 會自動在 `result_dir` 下再建立 task 子資料夾。例如：

```text
--result_dir results/PipelineRestormer/03_xxx
--task Motion_Deblurring
```

實際輸出會在：

```text
results/PipelineRestormer/03_xxx/Motion_Deblurring/
```

### DiffBIR 輸出在哪裡

DiffBIR 不會像 Restormer 一樣自動建立 task 子資料夾。它會把圖片直接存在 `--output` 指定的資料夾，例如：

```text
results/PipelineRestormer/04_darkir_alpha07_restormer_diffbir_from08/
```

同時會產生一個 `prompt.csv`，記錄本次使用的 prompt。

### DiffBIR 安裝失敗

DiffBIR 官方環境以 Python 3.10 和 PyTorch 2.2.2 為基準；如果 Colab 的 Python 或 torch 版本變動導致安裝失敗，建議先重開一個乾淨 GPU runtime，再從第 1 步重新跑。

本文件的 `scripts/setup_diffbir_colab.sh` 會保留 Colab 內建 torch，只安裝 DiffBIR 其他依賴，避免把可用的 CUDA torch 換掉。

### DiffBIR exit status 137

`returned non-zero exit status 137` 通常代表 Colab runtime 因為 RAM 或 GPU 記憶體不足，直接把 DiffBIR process 殺掉。這不是輸入路徑錯誤。

如果 log 裡出現類似：

```text
input_size: torch.Size([1, 3, 5304, 7952])
split to 41x62 = 2542 tiles
Executing Encoder Task Queue: ... /231322
```

代表圖片尺寸太大，DiffBIR 即使用 tiled VAE 也會產生非常大的 task queue，Colab 很容易直接 kill。

本文件的 DiffBIR 步驟已經改成先用 `scripts/resize_for_diffbir.py prepare --max-side 2048` 縮圖，DiffBIR 跑完後再用 `restore` resize 回 Restormer 輸出尺寸。這樣最終圖片大小仍然會跟 Restormer/原圖一致。

先用第 8 張單張測試，不要直接跑全部圖片。如果單張可以跑，全部圖片再接著跑。

如果還是 137，把 DiffBIR tile 再降一級：

```text
--cleaner_tile_size 96 --cleaner_tile_stride 48
--vae_encoder_tile_size 96
--vae_decoder_tile_size 96
--cldm_tile_size 192 --cldm_tile_stride 96
```

如果仍然 137，請改用 Colab High-RAM runtime 或更大 VRAM GPU。DiffBIR 會載入 Stable Diffusion/ControlNet 相關權重，普通 Colab runtime 很容易被殺掉。

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

### OOM

先降 tile：

```text
DarkIR:    --tile-size 2048 --overlap 320
Restormer: --tile 2048 --tile_overlap 320
DiffBIR:   --cldm_tile_size 192 --cldm_tile_stride 96
```
