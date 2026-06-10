# Colab：改良版 DarkIR + Restormer + DiffBIR Pipeline

這份流程使用 repo 內已寫好的腳本：

```text
scripts/colab_setup_darkir_restormer_diffbir_v2.sh
scripts/colab_run_darkir_restormer_diffbir_v2.sh
```

Pipeline 預設不是所有模型硬串，而是內容分流：

```text
img/
-> resize max_dim=1440
-> DarkIR gate：預設只處理 13
-> Restormer
-> OMDNet
-> DiffBIR gate：預設跳過 02,04,05,07,08,09,11,12,13
-> sharpen
-> Real-ESRGAN x4
```

判斷邏輯：

- `13` 是 DarkIR 最值得保留的低光/玻璃反射案例。
- `05,07,08,11,12` 文字、車牌、招牌多，DiffBIR 容易改字，所以跳過。
- `04,09` 是嚴重 failure case，DiffBIR 容易 hallucinate 假細節，所以跳過。

## 0. Colab GPU 與 Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!nvidia-smi
```

## 1. Clone 專案

把 `PROJECT_REPO` 換成你的 GitHub repo。`PROJECT_BRANCH` 換成目前放 pipeline 的分支。

```bash
%%bash
set -euo pipefail

PROJECT_REPO="https://github.com/<your-user>/<your-repo>.git"
PROJECT_BRANCH="fftformer-colab-experiment"

cd /content
if [ ! -d image_processing_final_project ]; then
  git clone "$PROJECT_REPO" image_processing_final_project
fi

cd /content/image_processing_final_project
git fetch origin "$PROJECT_BRANCH"
git checkout "$PROJECT_BRANCH"
git pull --ff-only origin "$PROJECT_BRANCH"
```

## 2. 準備 Drive 資料

預設腳本會從這些位置找圖片和權重：
預設腳本會自動 clone 外部模型 repo，其中 OMDNet 來源是 `https://github.com/yudingchuan/OMDNet.git`。Drive 只需要放圖片和權重；`OMDNET_DIR` 是 clone 失敗或你要用備份版本時的 fallback。

```text
/content/drive/MyDrive/deblur_project/img/
/content/drive/MyDrive/deblur_weights/
```

權重預設結構：

```text
/content/drive/MyDrive/deblur_weights/
  DarkIR_384.pt
  motion_deblurring.pth
  model_ckpt_epoch_219.ckpt
  RealESRGAN_x4plus.pth
  DiffBIR/
    ...官方 v2.1 需要的 weights...
```

如果你的 Drive 路徑不同，在執行 setup 時用 env var 改掉。

## 3. 一鍵環境安裝與檢查

這支腳本會做：

- 安裝 Python deps
- clone `DarkIR/`, `Restormer/`, `OMDNet/`, `DiffBIR/`, `Real-ESRGAN/`
- 從 Drive 複製圖片與權重
- 過濾 DiffBIR requirements 裡的 `xformers`
- 檢查必要檔案並 `py_compile` pipeline

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project
bash scripts/colab_setup_darkir_restormer_diffbir_v2.sh
```

若你的 Drive 路徑不同：

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project

WEIGHT_DIR="/content/drive/MyDrive/my_weights" \
IMAGE_DIR="/content/drive/MyDrive/my_images" \
OMDNET_DIR="/content/drive/MyDrive/my_OMDNet" \
bash scripts/colab_setup_darkir_restormer_diffbir_v2.sh
```

如果 Colab 已有可用 torch，不想重裝 torch：

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project

INSTALL_TORCH=0 \
bash scripts/colab_setup_darkir_restormer_diffbir_v2.sh
```

## 4. 第 8 張 smoke test

第 8 張預設跳過 DarkIR 與 DiffBIR，主要測保守文字路線：

```text
resize -> Restormer -> OMDNet -> sharpen
```

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project
bash scripts/colab_run_darkir_restormer_diffbir_v2.sh smoke08
```

看輸出：

```python
from IPython.display import display
from PIL import Image

display(Image.open('/content/image_processing_final_project/results/smoke_darkir_restormer_diffbir_v2_08/step7_sharpen_post/08.png'))
```

## 5. 第 13 張 DarkIR smoke test

第 8 張不會進 DarkIR，所以一定要再跑第 13 張確認 DarkIR gate 真的能跑。

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project
bash scripts/colab_run_darkir_restormer_diffbir_v2.sh smoke13
```

看 DarkIR 前後：

```python
from IPython.display import display
from PIL import Image

base = '/content/image_processing_final_project/results/smoke_darkir_v2_13'
for path in [
    f'{base}/step1_resize/13.png',
    f'{base}/step2_darkir_gate/13.png',
    f'{base}/step7_sharpen_post/13.png',
]:
    print(path)
    display(Image.open(path))
```

## 6. 候選組完整實驗

先跑 `07,10,13,14,15`，不要一開始全 15 張。這組會測到文字保護、DarkIR selective、DiffBIR selective、人臉/夜景候選。

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project
bash scripts/colab_run_darkir_restormer_diffbir_v2.sh candidates
```

## 7. 全 15 張正式跑

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project
bash scripts/colab_run_darkir_restormer_diffbir_v2.sh full
```

Colab 中斷後可續跑，例如從 DiffBIR gate，也就是 step 6 開始：

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project
bash scripts/colab_run_darkir_restormer_diffbir_v2.sh resume 6
```

## 8. 參數實驗

測試更多 DarkIR：

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project
bash scripts/colab_run_darkir_restormer_diffbir_v2.sh ablation_darkir
```

測試更保守 DiffBIR 強度：

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project
bash scripts/colab_run_darkir_restormer_diffbir_v2.sh ablation_diffbir
```

測試不跑 OMDNet：

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project
bash scripts/colab_run_darkir_restormer_diffbir_v2.sh ablation_no_omdnet
```

自訂參數時用 `custom`，後面直接接 Python pipeline args：

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project

bash scripts/colab_run_darkir_restormer_diffbir_v2.sh custom \
  --images 01,06,14 \
  --diffbir_strength 0.75 \
  --out_dir results/custom_darkir_v2
```

## 9. 打包下載

正式結果：

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project
bash scripts/colab_run_darkir_restormer_diffbir_v2.sh package_full
```

smoke/candidate/ablation 結果：

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project
bash scripts/colab_run_darkir_restormer_diffbir_v2.sh package_experiments
```

若要改輸出到其他 Drive 位置：

```bash
%%bash
set -euo pipefail

cd /content/image_processing_final_project

DRIVE_OUT="/content/drive/MyDrive/deblur_outputs" \
bash scripts/colab_run_darkir_restormer_diffbir_v2.sh package_full
```
