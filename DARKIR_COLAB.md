# DarkIR 在 Google Colab 上的建置與 Inference 流程

這份教本是給 Colab notebook 使用的。所有程式區塊都可以直接貼到 Colab cell 執行；如果 cell 第一行是 `%%bash`，它必須放在該 cell 的第一行。

如果你要跑「DarkIR -> 混回原圖 -> EVSSM」串接流程，請看 [DARKIR_EVSSM_PIPELINE_COLAB.md](DARKIR_EVSSM_PIPELINE_COLAB.md)。

本教本預設你要使用的模型是：

```text
DarkIR_384.pt
```

這個 checkpoint 建議搭配：

```text
methods/DarkIR/options/inference/real_lsrw.yml
```

原因是 `DarkIR_384.pt` 是 `width: 32` 架構，而 `real_lsrw.yml` 也是 `width: 32`。不要用 `LOLBlur.yml` 搭配 `DarkIR_384.pt`，因為 `LOLBlur.yml` 是 `width: 64`，權重和模型結構會不匹配。

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

## 2. 進入專案並更新 DarkIR submodule

如果專案已經在 Colab 裡：

```python
%cd /content/image_processing_final_project
!git pull
!git submodule update --init --recursive methods/DarkIR
```

如果你是重新 clone：

```python
%cd /content
REPO_URL = "PASTE_YOUR_REPO_URL_HERE"
assert REPO_URL != "PASTE_YOUR_REPO_URL_HERE", "Set REPO_URL before running this cell."
!git clone --recursive {REPO_URL} image_processing_final_project
%cd /content/image_processing_final_project
!git submodule update --init --recursive methods/DarkIR
```

確認 DarkIR 和圖片存在：

```python
!test -f methods/DarkIR/archs/DarkIR.py && echo "DarkIR submodule ok"
!find img -maxdepth 1 -type f | head
```

## 3. 安裝 DarkIR 依賴

```python
!bash scripts/setup_darkir_colab.sh
```

這個腳本會保留 Colab 目前的 PyTorch，不會強制安裝 DarkIR 官方 `requirements.txt` 裡指定的 `torch==2.5.1`。在 Colab 裡重裝 PyTorch 很容易破壞 GPU runtime，所以這裡只安裝 DarkIR 額外需要的套件。

## 4. 從 Google Drive 複製 `DarkIR_384.pt`

先掛載 Google Drive：

```python
from google.colab import drive
drive.mount("/content/drive")
```

假設你的 checkpoint 放在：

```text
MyDrive/DarkIR_checkpoints/DarkIR_384.pt
```

執行：

```python
%cd /content/image_processing_final_project

!mkdir -p methods/DarkIR/models
!cp /content/drive/MyDrive/DarkIR_checkpoints/DarkIR_384.pt methods/DarkIR/models/
!ls -lh methods/DarkIR/models
```

如果你的檔案在其他資料夾，請修改這段路徑：

```text
/content/drive/MyDrive/DarkIR_checkpoints/DarkIR_384.pt
```

最後你應該會看到：

```text
methods/DarkIR/models/DarkIR_384.pt
```

## 5. Dry Run 確認路徑

先確認腳本會讀到圖片、config 和 checkpoint，但不真的跑模型：

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/DarkIR384_dryrun \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --start-index 1 \
  --limit 1 \
  --dry-run
```

測第 8 張大圖的 dry run：

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/DarkIR384_dryrun_from08 \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --start-index 8 \
  --limit 1 \
  --dry-run
```

## 6. 先跑第一張 Whole-Image

這是不切 tile、不縮圖，直接整張圖片送進 DarkIR：

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/DarkIR384_whole_first \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --start-index 1 \
  --limit 1
```

如果成功，輸出會在：

```text
results/DarkIR/DarkIR384_whole_first/
```

## 7. 測第 8 張大圖

第 8 張是目前 `img/` 裡很大的圖片之一。建議先用 tiled inference 測：

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/DarkIR384_tile3072_from08 \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop \
  --start-index 8 \
  --limit 1
```

如果 `tile-size 3072` OOM，就改成比較保守的：

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/DarkIR384_tile2048_from08 \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 2048 \
  --overlap 320 \
  --blend crop \
  --start-index 8 \
  --limit 1
```

如果 A100 80GB 還很空，可以試更大的：

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/DarkIR384_tile4096_from08 \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 4096 \
  --overlap 640 \
  --blend crop \
  --start-index 8 \
  --limit 1
```

## 8. 跑全部圖片

如果第 8 張測試結果和記憶體都 OK，可以跑全部：

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/DarkIR384_tile3072_all \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop
```

如果想從第 8 張開始跑到最後：

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/DarkIR384_tile3072_from08_to_end \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop \
  --start-index 8
```

## 9. 下載結果

壓縮結果：

```python
!zip -r darkir384_results.zip results/DarkIR
```

下載：

```python
from google.colab import files
files.download("darkir384_results.zip")
```

## 10. DarkIR 接 EVSSM：先混回原圖降低過亮感

如果 DarkIR 結果有些區域過亮，可以不要直接把 DarkIR 最終圖丟給 EVSSM，而是先和原圖混合：

```text
blended = alpha * DarkIR(input) + (1 - alpha) * input
```

建議先試：

```text
alpha = 0.6, 0.7, 0.8
```

`alpha` 越小，越接近原圖，亮度會被壓低更多；`alpha` 越大，越接近 DarkIR 結果，增亮和去雜訊效果會更強。

### 10.1 先產生 DarkIR 結果

例如先用 `DarkIR_384.pt` 跑第 8 張：

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/DarkIR384_tile3072_from08 \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop \
  --start-index 8 \
  --limit 1
```

### 10.2 將 DarkIR 結果和原圖混合

先 dry run：

```python
%%bash
python scripts/blend_darkir_for_evssm.py \
  --original img \
  --darkir results/DarkIR/DarkIR384_tile3072_from08 \
  --output results/Pipeline/DarkIR384_alpha07_for_EVSSM_from08 \
  --alpha 0.7 \
  --start-index 8 \
  --limit 1 \
  --dry-run
```

正式產生 EVSSM input：

```python
%%bash
python scripts/blend_darkir_for_evssm.py \
  --original img \
  --darkir results/DarkIR/DarkIR384_tile3072_from08 \
  --output results/Pipeline/DarkIR384_alpha07_for_EVSSM_from08 \
  --alpha 0.7 \
  --start-index 8 \
  --limit 1
```

如果還是太亮，改用：

```text
--alpha 0.6
```

如果太暗或 DarkIR 效果被削弱太多，改用：

```text
--alpha 0.8
```

### 10.3 用混合結果當 EVSSM input

```python
%%bash
python scripts/evssm_infer_folder.py \
  --input results/Pipeline/DarkIR384_alpha07_for_EVSSM_from08 \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output results/Pipeline/DarkIR384_alpha07_EVSSM_RealBlurJ_from08 \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop \
  --limit 1
```

注意：這裡的 EVSSM input 資料夾已經只包含第 8 張，所以不需要再加 `--start-index 8`。如果你的混合資料夾包含全部圖片，就可以照常用 `--start-index` 和 `--limit`。

### 10.4 跑全部圖片

先跑全部 DarkIR：

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/DarkIR384_tile3072_all \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 3072 \
  --overlap 512 \
  --blend crop
```

混回原圖：

```python
%%bash
python scripts/blend_darkir_for_evssm.py \
  --original img \
  --darkir results/DarkIR/DarkIR384_tile3072_all \
  --output results/Pipeline/DarkIR384_alpha07_for_EVSSM_all \
  --alpha 0.7
```

丟進 EVSSM：

```python
%%bash
python scripts/evssm_infer_folder.py \
  --input results/Pipeline/DarkIR384_alpha07_for_EVSSM_all \
  --checkpoint checkpoints/net_g_realblur_j.pth \
  --output results/Pipeline/DarkIR384_alpha07_EVSSM_RealBlurJ_all \
  --tile-size 1536 \
  --overlap 260 \
  --blend crop
```

## 11. 常見問題

### 找不到 checkpoint

確認檔案在：

```python
!ls -lh methods/DarkIR/models
```

應該要看到：

```text
DarkIR_384.pt
```

### checkpoint 和 config 不匹配

`DarkIR_384.pt` 請使用：

```text
methods/DarkIR/options/inference/real_lsrw.yml
```

不要使用：

```text
methods/DarkIR/options/inference/LOLBlur.yml
```

因為 `LOLBlur.yml` 是 `width: 64`，而 `DarkIR_384.pt` 通常是 `width: 32`。

### CUDA OOM

優先嘗試：

```text
--tile-size 2048 --overlap 320 --blend crop
```

或使用半精度：

```text
--precision fp16
```

例如：

```python
%%bash
python scripts/darkir_infer_folder.py \
  --input img \
  --output results/DarkIR/DarkIR384_tile2048_fp16_from08 \
  --config methods/DarkIR/options/inference/real_lsrw.yml \
  --checkpoint methods/DarkIR/models/DarkIR_384.pt \
  --tile-size 2048 \
  --overlap 320 \
  --blend crop \
  --precision fp16 \
  --start-index 8 \
  --limit 1
```
