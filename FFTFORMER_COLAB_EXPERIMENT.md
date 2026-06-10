# Colab：FFTformer 實測流程

目的：在獨立分支 `fftformer-colab-experiment` 上重跑 FFTformer，確認它是否能突破 Restormer / NAFNet。`all-pipelines` 既有結果顯示三者幾乎一樣，所以這份流程的重點是可重現比較，而不是期待它一定勝出。

官方 repo：<https://github.com/kkkls/FFTformer>

## 1. Colab setup

```bash
cd /content
git clone <你的專案 repo> image_processing_final_project
cd /content/image_processing_final_project
git checkout fftformer-colab-experiment

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python pillow numpy scikit-image einops basicsr tqdm pyyaml lmdb
```

## 2. Clone FFTformer

官方 README 標示依賴 Python / PyTorch / scikit-image / opencv-python / TensorBoard / einops，測試入口是 `test.sh`，pretrained model 在 repo 的 `pretrain_model/` 發布。

```bash
cd /content/image_processing_final_project
git clone https://github.com/kkkls/FFTformer.git FFTformer
```

確認權重：

```bash
ls -lh FFTformer/pretrain_model/
```

應至少有 GoPro 或 RealBlur 權重。此專案腳本預設使用：

```text
FFTformer/pretrain_model/fftformer_GoPro.pth
```

若檔名不同，請改 `code/pipeline/run_fftformer.py` 的 `load_model()` 權重路徑，或在 Colab 中建立同名 symlink。

## 3. 第 8 張 smoke test

先不跑 ESRGAN，確認 FFTformer 本體能通：

```bash
python code/pipeline/run_fftformer.py \
  --input_dir img \
  --images 08 \
  --skip_esrgan \
  --out_dir results/fftformer_smoke_08
```

輸出會在：

```text
results/fftformer_smoke_08/pre_esrgan/08_1440.png
```

## 4. 重跑 all-pipelines 的重點圖

```bash
python code/pipeline/run_fftformer.py \
  --input_dir img \
  --images 04,06,07,09 \
  --extra_04_720 \
  --out_dir results/pipeline_fftformer_colab
```

這會產出：

```text
results/pipeline_fftformer_colab/04_1440.png
results/pipeline_fftformer_colab/04_720.png
results/pipeline_fftformer_colab/06_1440.png
results/pipeline_fftformer_colab/07_1440.png
results/pipeline_fftformer_colab/09_1440.png
```

若 Colab VRAM 不足：

```bash
python code/pipeline/run_fftformer.py \
  --input_dir img \
  --images 04,06,07,09 \
  --tile 192 \
  --overlap 32 \
  --skip_esrgan \
  --out_dir results/pipeline_fftformer_colab_tile192
```

## 5. 對照方法

用既有結果對比：

```text
results/pipeline_fftformer/
results/pipeline_v23_nafnet/step5_realesrgan/
results/pipeline_v18/step6_realesrgan/
results/_final_compare/_3way_backbone.png
```

判讀重點：

- 04：即使跑 720，騎士仍像鬼影，表示主體資訊已遺失。
- 07：FFTformer 可以得到銳利車體，但不比 v18 保守路線明顯更好。
- 09：zoom blur 是 spatially-variant kernel，FFTformer 不能根本解決。
- 06：與 Restormer / NAFNet 相近，可作 backbone ceiling 證據。

## 6. 結論寫法

建議在報告中把 FFTformer 寫成「驗證 backbone ceiling」：

```text
Although FFTformer introduces frequency-domain self-attention and is theoretically attractive for motion blur, our side-by-side experiment on real night photos shows no clear advantage over Restormer and NAFNet. The hard cases remain hard across all three backbones, suggesting that the limiting factor is the out-of-distribution blur pattern and information loss, not the backbone architecture.
```
