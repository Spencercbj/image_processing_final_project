# 夜間影像去模糊 — Term Project

對 15 張夜間拍攝的模糊照片（`img/`）做去模糊 + 增亮 + 細節增強。
本 branch (`all pipelines`) 收錄了我們試過的**所有 pipeline、所有程式碼、精選結果**。

- 完整實驗紀錄：[`experiment_log.md`](experiment_log.md)
- 隔夜衝刺總結 + 最終建議：[`OVERNIGHT_RESULTS.md`](OVERNIGHT_RESULTS.md)
- 待辦/想法清單：[`ideas.md`](ideas.md)

---

## 1. 環境安裝

GPU：RTX 4060 Laptop (8GB)。CUDA 12.4。Python 3.10。

```bash
# 方法 A：conda（推薦）
conda env create -f environment.yml
conda activate deblur

# 方法 B：pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
```

### ⚠️ 踩坑（務必注意）
1. **絕對不要 `pip install xformers`** — 它會把 CUDA 版 torch 換成 CPU-only，整個環境壞掉。
2. **直接用 `python.exe` 完整路徑**跑（例：`C:\...\envs\deblur\python.exe`），不要用 `conda run`（Windows cp950 編碼會報錯）。
3. **OMDNet 一定要用 PIL `Image.LANCZOS` 縮到 1440** 再餵進去（不要用 cv2 resize），否則 gate 機制表現異常。
4. 8GB VRAM：高解析度的模型都要 **tiling**（程式裡都已內建）。

---

## 2. 需要另外下載的模型權重

程式碼放在 `code/`，但**模型本體與權重沒有進 repo**（太大）。各 pipeline 用到的模型如下，請自行 clone + 下載權重放到對應位置：

| 模型 | 用途 | 來源 | 權重 |
|------|------|------|------|
| **Restormer** | 運動去模糊（主力 backbone） | github.com/swz30/Restormer | `Motion_Deblurring/pretrained_models/motion_deblurring.pth` |
| **OMDNet** | 局部運動去模糊（gate 機制） | CVPR 2026 | `OMDNet/checkpoints/test1/model_ckpt_epoch_219.ckpt` |
| **Real-ESRGAN** | 超解析 x4 / 細節增強 | github.com/xinntao/Real-ESRGAN | `RealESRGAN_x4plus.pth` |
| **GFPGAN** | 人臉增強（ESRGAN 內） | github.com/TencentARC/GFPGAN | `GFPGANv1.3` + facelib |
| **CodeFormer** ⭐ | 人臉修復（比 GFPGAN 穩） | github.com/sczhou/CodeFormer | `weights/CodeFormer/codeformer.pth` |
| **NAFNet** | 去模糊 backbone（替代 Restormer） | github.com/megvii-research/NAFNet | `NAFNet-GoPro-width64.pth` |
| **FFTformer** | 頻域去模糊 | github.com/kkkls/FFTformer | `pretrain_model/fftformer_GoPro.pth`（repo 內附） |
| **DiffBIR** | Diffusion 盲修復 | github.com/XPixelGroup/DiffBIR | v2.1 weights（SD2.1 + IRControlNet） |
| **DarkIR** | 低光+去模糊聯合 | CVPR 2025 | 官方權重 |
| **HVI-CIDNet** | 低光增強 | CVPR 2025 | 官方權重 |
| **MPRNet** | 多階段去模糊 | github.com/swz30/MPRNet | `model_deblurring.pth` |
| **UFPNet** | 非均勻去模糊 | github.com/Fangzhenxuan/UFPDeblur | `net_g_latest.pth` |

> CodeFormer 踩坑：bundled `basicsr/__init__.py` 會 import `.version`，需手動建 `CodeFormer/basicsr/version.py`（`__version__='1.3.2'; __gitsha__='unknown'; version_info=(1,3,2)`）。

---

## 3. 試過的 Pipeline（完整版見 `experiment_log.md`）

### 演進主線
| 版本 | 流程 | 重點 / 結論 |
|------|------|------------|
| v1–v2 | HVI-CIDNet → MISCFilter → Restormer → PASD/SDEdit → ESRGAN | PASD hallucination 嚴重；MISCFilter 對 real blur 弱 → 棄用 |
| v3–v4 | HVI-CIDNet → Wiener/Restormer → ESRGAN | 均勻 deconv 傷清晰區；PSF 難估 → 棄用 |
| v5–v7 | (低光) → Restormer → **降解析** → OMDNet → ESRGAN | **發現：降解析度後 OMDNet 去模糊更強** |
| v8–v12 | 比較實驗：低光方法 / OMD 解析度(1440 vs 1920) / Sharpen 強度 | 結論：**1440 + 無低光 + Sharpen 0.5** 最佳 |
| **v13** | Restormer → OMDNet(1440) → **DiffBIR(s=0.8)** → ESRGAN | DiffBIR 修細節，但高強度有 hallucination |
| **v14** | CLAHE → Restormer → OMDNet → DiffBIR(0.85) → ESRGAN | 文字圖跳過 DiffBIR；06/14 最佳 |
| v15 | Restormer → OMDNet → Sharpen → ESRGAN | **文字專用**（無 DiffBIR，文字最清楚） |
| v16 | **DarkIR** → Restormer → OMDNet → ESRGAN | 低光+去模糊聯合；13 最佳 |
| v17 | 先 resize 再 CLAHE → Restormer → OMDNet → ESRGAN | early-resize 實驗 |
| **v18** | resize1440 → Restormer → OMDNet → **NLMeans** → Sharpen → ESRGAN(+GFPGAN 臉) | **最終量產版**，全 15 張 |
| v19 | resize → Restormer → OMDNet → DiffBIR(0.85) → GFPGAN | 人臉專用 |
| v20 | resize → **Richardson-Lucy deconv** → Restormer → OMDNet → ESRGAN | 04/09 嚴重模糊；ringing 嚴重 → 棄 |
| v21 | resize → **UFPNet → MPRNet → Restormer → OMDNet** → ESRGAN | cascade 多模型堆疊 |
| v22 | resize → **DiffBIR SDEdit 低強度(0.4–0.6)** → ESRGAN | 救 04/09；hallucination/tile artifact |
| **v23** | resize1440 → **NAFNet(GoPro)** → OMDNet → Sharpen → ESRGAN | NAFNet 當 backbone：**≈ Restormer，無突破** |

### 隔夜新增實驗（2026-06-10）
| 實驗 | 流程 | 結論 |
|------|------|------|
| **CodeFormer** ⭐ | 既有最佳結果 → CodeFormer(w=0.5/0.7, 只換臉) | **本輪最大收穫**：10/14/15 人臉明顯變好，勝 GFPGAN/v19 |
| **FFTformer** | resize → FFTformer(GoPro) → ESRGAN | 與 Restormer/NAFNet 三者**幾乎一樣** → 瓶頸是資料不是架構 |
| **04 低解析** | resize 480/720 → Restormer/NAFNet/OMD → ESRGAN | 騎士仍鬼影 → failure case |
| **09 log-polar** | warpPolar(LOG) → 1D RL deconv → inverse | zoom blur 部分可解（字變清楚），邊緣 ringing |
| **DiffBIR full** | start=noise, strength=1.0 | 與 SDEdit 不同路徑；04 仍無解 |
| **03 增亮** | v18 結果 → shadow-lift + CLAHE | 去模糊本來就夠，純後製增亮 |

### 關鍵結論
- **Restormer ≈ NAFNet ≈ FFTformer**：三個 SOTA backbone 天花板相同。
- **DiffBIR 會毀文字**：07 v13 把車牌「TDC-1769」生成成「HAAFIP」→ 文字圖必須跳過。
- **CodeFormer 是人臉最佳解**，比 GFPGAN 穩、不易換臉。
- **Failure cases**：04（kernel 超出訓練分布）、09（spatially-variant zoom blur）、11（長曝光鬼影=資訊遺失）。

---

## 4. 程式碼結構 `code/`

- `code/pipeline/` — 所有 pipeline 主程式（`run_pipeline_v*.py`、`run_03_brighten.py`、`run_04_lowres.py`、`run_09_logpolar.py`、`run_fftformer.py`、`run_diffbir_full.py`、`montage.py` 等）
- `code/Restormer/`、`code/NAFNet/`、`code/OMDNet/` … — 各模型的推理 wrapper（含 tiling）
- `code/freq_analyze/`、`code/deconvolution/`、`code/wiener/` — 頻率分析與傳統去卷積

執行範例（注意用完整 python 路徑）：
```bash
<python.exe> code/pipeline/run_pipeline_v18.py --images 01,05,07
```

---

## 5. 結果 `results/` 與精選 `z_*`

為了控制體積，repo 只收 **最終輸出 + before/after 對比 + 精選**（不含逐步中間檔，原始 7GB → 約 1.5GB）。

- `upload_picks/` — **互評上傳的 2 張**（14 人像、07 計程車，全解析度）
- `z_candidates/` — 每張圖的最佳候選
- `z_results/` — 早期各 pipeline 精選
- `results/_final_compare/` — **before/after 對比圖**（做 PPT 用）
- `results/pipeline_v18/` — 主力量產版全 15 張
- `results/pipeline_{v23_nafnet,fftformer,diffbir_full,03_brighten,04_lowres,09_logpolar,codeformer_faces}/` — 隔夜新實驗
