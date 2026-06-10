# 影像去模糊 Pipeline 實驗紀錄

## 專案概述

- **目標**: 對 15 張夜間拍攝的模糊照片進行去模糊、增亮、細節增強
- **截止日期**: 2026-06-11 (結果), 2026-06-12 (報告)
- **測試圖片**: 01~15（早期實驗用 01, 05, 08, 12, 15；後期全部 15 張）

---

## 15 張照片描述

| # | 檔名 | 描述 | 模糊類型 | 難度 |
|---|------|------|---------|------|
| 01 | Urban_Light_Trails_Walking_Figure | 都市街頭，一人側身行走，背景滿是水平光軌 | 水平運動模糊（背景）、人物輕微模糊 | 中 |
| 02 | Red_Sign_Reflections_Blurred_Glass | 透過玻璃拍攝街景，紅色霓虹招牌反射鏡像，前景嚴重散焦 | 散焦模糊 + 玻璃反射疊影 | 高 |
| 03 | Dim_Night_Market_Food_Alley | 昏暗夜市走道，紅燈泡照明，多人走動 | 低光 + 輕微人物移動模糊 | 中（主要是暗） |
| 04 | Cyclist_Passing_Warm_Storefront_Lights | 騎腳踏車的人經過暖色店面，背景燈光水平拖尾嚴重 | 強水平運動模糊（整體）~165° | 極高 |
| 05 | Red_Taxi_Through_City_Lights | 紅色計程車穿越城市路口，背景霓虹招牌水平模糊 | 水平運動模糊（背景）、車體相對清晰 | 中 |
| 06 | Shaking_Neon_Signs_Overhead_Night | 仰拍高樓霓虹招牌（綠/粉/白），整張嚴重晃動 | 多方向手震模糊、光暈嚴重 | 極高 |
| 07 | Yellow_Taxi_Neon_Rain_Street | 黃色計程車（TDC-1769）側面，背景是五顏六色霓虹招牌，雨天 | 水平運動模糊（背景）、車體較清晰、有文字 | 中 |
| 08 | KFC_Rider_Rainy_Night_Delivery | KFC 外送員騎紅色機車，雨天夜晚，背景水平光軌 | 水平運動模糊、雨滴 | 中 |
| 09 | White_Truck_Zoom_Blur_Rain | 白色卡車（Sita.co.uk），放射狀 zoom blur，雨天 | 放射狀 zoom blur ~8° | 極高 |
| 10 | Crowded_Night_Market_Face_Glow | 擁擠夜市人群，多張人臉清晰可見，霓虹招牌背景 | 輕微移動模糊、人群雜亂 | 低~中 |
| 11 | Fumachi_Night_Market_Ghost_Crowd | 福町夜市入口，長曝光人群呈半透明鬼影，招牌文字清楚 | 長曝光鬼影（人群）、靜態物清晰 | 中（人群難救） |
| 12 | Sidewalk_Signage_In_City_Lights | 人行道標誌牌（停/SIDEWALK/TOW-AWAY），大量中文招牌背景 | 幾乎無模糊，主要是文字清晰度 | 低（文字多） |
| 13 | FamilyMart_Window_Reflections_Night_Traffic | FamilyMart 便利商店門面，玻璃反射車輛，多層次疊影 | 玻璃反射疊影 + 低光 | 中~高 |
| 14 | Yellow_Chair_Alley_Motion_Portrait | 巷弄中女子坐在黃色塑膠椅上，霓虹燈光背景，人物晃動 | 人物運動模糊 + 手震 | 高（人臉） |
| 15 | Photographer_Reflected_In_Night_Glass | 攝影師透過玻璃的自拍，手持相機，玻璃反射疊影 | 玻璃反射疊影 + 運動模糊、低光 | 高（人臉） |

---

## 使用的模型/工具一覽

| 模型 | 用途 | 來源 |
|------|------|------|
| **HVI-CIDNet** | 低光增強 | CVPR 2025 |
| **CLAHE** | 低光增強（傳統方法） | OpenCV |
| **DarkIR** | 低光+去模糊聯合處理 | CVPR 2025 |
| **MISCFilter** | 盲去運動模糊 | ECCV 2024 |
| **Restormer** | 運動去模糊 | — |
| **OMDNet** | 局部運動去模糊（gate 機制） | CVPR 2026 |
| **Wiener Deconvolution** | 方向性去模糊（需估 PSF） | 傳統方法 |
| **Richardson-Lucy Deconv** | 迭代反卷積 | 傳統方法 |
| **UFPNet** | 去模糊 | — |
| **MPRNet** | 去模糊/去噪 | — |
| **PASD** | Diffusion 超解析 | ECCV 2024 |
| **SDEdit** | Diffusion 微修（img2img） | — |
| **DiffBIR** | Diffusion 修復 | — |
| **Real-ESRGAN** | 超解析/細節增強 | — |
| **GFPGAN** | 人臉增強 | — |
| **EasyOCR** | 文字區域偵測（用於 blending） | — |

---

## Pipeline 版本總覽

### Pipeline v1（已棄用）

```
原圖 → HVI-CIDNet → MISCFilter → Restormer → PASD → Real-ESRGAN
```

- **測試圖**: 01, 08, 12
- **已知問題**:
  - HVI-CIDNet tile overlap=64 太小，用 uniform weight → 格子狀 artifact
  - PASD hallucination 嚴重，改變太多細節
  - MISCFilter + Restormer 對嚴重 real-world motion blur 去模糊效果有限
  - 提前切換至 v2

---

### Pipeline v2（已棄用）

```
原圖 → HVI-CIDNet → MISCFilter → Restormer → SDEdit → Real-ESRGAN → EasyOCR Blending
```

- **測試圖**: 01, 08, 12
- **改進**:
  - HVI-CIDNet 改用 cosine blending + overlap=128 → 修好格子 artifact
  - SDEdit (strength=0.35) 取代 PASD → 較保守，減少 hallucination
  - 加入 EasyOCR Blending 保留文字
- **已知問題**:
  - SDEdit 在 768px 解析度下處理再 upscale 回來，解析度落差造成品質損失
  - EasyOCR blending 品質不佳

---

### Pipeline v3（已棄用）

```
原圖 → HVI-CIDNet → Wiener Deconv（per-image PSF）→ Restormer → Real-ESRGAN
```

- **測試圖**: 05, 06, 08, 15
- **改進**:
  - 移除 MISCFilter，改用 Wiener Deconv + 頻率分析估 PSF
  - 移除 SDEdit 和 EasyOCR Blending
  - Per-image 策略：依模糊嚴重程度調整 Wiener K
- **已知問題**:
  - 均勻 deconv 傷害本來就清楚的區域
  - PSF 估計不夠精確

---

### Pipeline v4（已棄用）

```
原圖 → HVI-CIDNet → Restormer（cosine blending）→ cv2 Denoise → Real-ESRGAN
```

- **測試圖**: 規劃全 15 張，但只產出 instructions.md
- **改進**:
  - 移除 Wiener deconv
  - Restormer 加 cosine blending
  - 新增 cv2 denoise step
- **已知問題**:
  - 結果未完整產出

---

### Pipeline v5（已棄用）

```
原圖 → HVI-CIDNet → Downscale (1/2x) → OMDNet → Real-ESRGAN (x2)
```

- **測試圖**: 05, 08, 15
- **重點**: 首次引入 OMDNet（gate 機制選擇性去模糊）
- **發現**:
  - 降低解析度後 OMDNet 效果更好（1/2 解析度 → 3x 去模糊效果）
  - 但 downscale 太多會損失細節

---

### Pipeline v6（已棄用）

```
原圖 → HVI-CIDNet → Restormer → Downscale → OMDNet → Real-ESRGAN
```

- **測試圖**: 05, 08, 15
- **改進**: 在 OMDNet 前加 Restormer

---

### Pipeline v7（已棄用）

```
原圖 → Restormer → OMDNet → Real-ESRGAN
```

- **測試圖**: 05, 08, 15
- **改進**: 移除低光增強，測試 baseline

---

### Pipeline v8a / v8b（實驗比較用）

```
v8 shared: 原圖 → HVI-CIDNet → Restormer → resize 1440 → OMDNet
v8a: → SDEdit (strength=0.25) → Sharpen(1.0) → Real-ESRGAN x4
v8b: → Sharpen(1.0) → Real-ESRGAN x4
```

- **測試圖**: 01, 05, 08, 11, 12, 15
- **用時**: 共享步驟 + a + b = 21.5 min
- **比較軸**: SDEdit 有 vs 無

---

### Pipeline v9（實驗比較用）

```
原圖 → CLAHE → Restormer → resize 1920 → OMDNet → Sharpen(1.5) → Real-ESRGAN x3
```

- **測試圖**: 01, 05, 08, 11, 12, 15
- **用時**: 30.3 min
- **比較軸**: CLAHE vs HVI-CIDNet vs none；1920 vs 1440 解析度

---

### Pipeline v10（實驗比較用）

```
原圖 → HVI-CIDNet(v8) → Restormer(v8) → resize 1920 → OMDNet → Sharpen(1.5) → Real-ESRGAN x3
```

- **測試圖**: 01, 05, 08, 11, 12, 15
- **用時**: 2.8 min（重用 v8 中間結果）
- **比較軸**: HVI-CIDNet + 1920 解析度

---

### Pipeline v11（實驗比較用 — baseline）

```
原圖 → Restormer → resize 1920 → OMDNet → Sharpen(1.5) → Real-ESRGAN x3
```

- **測試圖**: 01, 05, 08, 11, 12, 15
- **用時**: 27.7 min
- **比較軸**: 無低光增強 baseline

---

### Pipeline v12（過渡版本）

```
原圖 → Restormer → resize 1440 → OMDNet → Sharpen(1.5) → Real-ESRGAN x4
```

- **說明**: 無低光增強 + 1440 解析度，從 v8-v11 實驗得出的最佳組合

---

### Pipeline v13（核心版本）

```
原圖 → Restormer → OMDNet (PIL resize 1440 in-memory) → DiffBIR (s=0.8) → Real-ESRGAN x4
```

- **測試圖**: 01, 05, 08, 11, 12, 15
- **重點**: 引入 DiffBIR 取代 SDEdit，strength=0.8
- **變體**: 
  - `pipeline_v13_s09`: DiffBIR strength=0.9
  - `pipeline_v13_s095`: DiffBIR strength=0.95
- **效果**: DiffBIR 能修復細節但強度過高會引入 hallucination

---

### Pipeline v14（CLAHE + DiffBIR + 文字保護）

```
原圖 → CLAHE → Restormer → OMDNet (PIL 1440) → [Sharpen(0.3) → DiffBIR(0.85)] → Sharpen(0.5) → ESRGAN x4
```

- **測試圖**: 全 15 張
- **特色**: 文字多的圖片跳過 DiffBIR（避免文字扭曲）

---

### Pipeline v15（文字專用）

```
原圖 → Restormer → OMDNet (PIL 1440) → Sharpen(0.5) → Real-ESRGAN x4
```

- **用途**: 文字多的圖片，不用 CLAHE、不用 DiffBIR
- **效果**: 文字保留最清楚

---

### Pipeline v16（DarkIR）

```
原圖 → DarkIR → Restormer → OMDNet (PIL 1440) → Sharpen(0.5) → Real-ESRGAN x4
```

- **特色**: DarkIR (CVPR 2025) 同時處理低光+去模糊
- **效果**: 見 `results/pipeline_v16_darkir/`

---

### Pipeline v17（Early Resize）

```
原圖 → PIL resize 1440 → CLAHE → Restormer → OMDNet → Sharpen(0.5) → ESRGAN x4
```

- **測試**: 先縮圖再處理是否更好
- **效果**: 見 `results/pipeline_v17_earlyresize/`

---

### Pipeline v18（最終量產版）

```
原圖 → PIL resize 1440 → Restormer → OMDNet → NLMeans denoise → Sharpen(0.5) → ESRGAN x4
```

- **測試圖**: 全 15 張
- **特色**:
  - 人臉圖片 (01, 03, 10, 14, 15) 在 ESRGAN 步驟啟用 GFPGAN face enhance
  - 加入 NLMeans denoise 降噪
- **效果**: 全圖產出，結果在 `results/pipeline_v18/`

---

### Pipeline v19（人臉專用）

```
原圖 → resize 1440 → Restormer → OMDNet → DiffBIR SDEdit (s=0.85) → GFPGAN
```

- **測試圖**: 01, 03, 10, 14, 15（人臉圖片）
- **特色**: 改善 negative prompt 減少人臉周圍 ghosting/artifact

---

### Pipeline v20（RL Deconv 實驗）

```
原圖 → PIL resize 1440 → Richardson-Lucy deconv → Restormer → OMDNet → Real-ESRGAN
```

- **測試圖**: 04, 09（嚴重運動模糊）
- **說明**: 頻率分析得到 04 模糊角度 ~165°, 09 模糊角度 ~8°
- **效果**: `pipeline_v20_rl` 目錄為空，推測實驗未完成或效果不佳

---

### Pipeline v21（Cascade 多模型）

```
原圖 → PIL resize 1440 → UFPNet → MPRNet → Restormer → OMDNet → Real-ESRGAN x4
```

- **測試圖**: 先跑 04, 09；後擴展全部（跳過已跑的 04, 09）
- **特色**: 堆疊三種去模糊模型（UFPNet → MPRNet → Restormer）
- **人臉圖**: ESRGAN 步驟啟用 face_enhance

---

### Pipeline v22（DiffBIR 低強度）

```
v18 的 resize 1440 結果 → DiffBIR SDEdit (s=0.4, 0.5, 0.6) → Real-ESRGAN x4
```

- **測試圖**: 04, 09
- **說明**: 嘗試較低的 DiffBIR 強度修復嚴重模糊的圖片

---

### Pipeline v23（NAFNet backbone）

```
原圖 → PIL resize 1440 → NAFNet(GoPro) → OMDNet → Sharpen(0.5) → ESRGAN x4
```

- **測試圖**: 01, 04, 05, 06, 07, 08, 09
- **說明**: 首次啟用 NAFNet（GoPro 權重，CVPR 2022），取代 Restormer 當去模糊 backbone
- **效果**: 與 Restormer **效果相當**，無明顯勝出；對 04/09 一樣無解。結論：NAFNet ≈ Restormer，兩者天花板相同
- **結果**: `results/pipeline_v23_nafnet/step5_realesrgan/`

---

### 03 後製增亮（不再跑模型）

```
v18 最終結果 → shadow-lift (gamma 0.72, 保護高光) + 溫和 CLAHE(clip 1.2)
```

- **說明**: 03 的去模糊本來就夠，問題只是太暗。在 pipeline 最後增亮（噪點已被 NLMeans 清掉，不會放大）
- **效果**: 暗部細節拉出來、暖色燈泡不過曝。勝過 v16 DarkIR（後者較平、失夜市氛圍）
- **結果**: `results/pipeline_03_brighten/`（最佳 `03_e_lift_clahe.png`）

---

### 04 激進降解析度實驗

```
原圖 → PIL resize 480/720 → [Restormer | NAFNet | NAFNet→OMDNet] → ESRGAN x4
```

- **說明**: 試圖把 blur kernel 等比例縮短到模型訓練範圍內（v5 發現低解析度去模糊更強）
- **效果**: 店面結構略清楚，但騎士仍是鬼影 → **無突破**。確認 04 為 failure case
- **結果**: `results/pipeline_04_lowres/`

---

### 09 LOG-polar 去卷積（修正版）

```
估計 zoom 中心(高光抑制+中央限制) → warpPolar(WARP_POLAR_LOG) →
逐通道 1D 水平 Richardson-Lucy deconv (k=9~23) → inverse warpPolar
```

- **說明**: 關鍵修正——zoom blur 在 **log-polar** 下才會變成「沿半徑軸的均勻平移」，
  linear polar 不會（這是先前 polar 實驗失敗主因）。診斷圖 `diag_logpolar.png` 確認模糊大致變水平
- **效果**: 「Sita.co.uk」字樣有變清楚，但邊緣放射狀殘留 + ringing；非成品級
- **fallback**: 中央裁切 60%（卡車為最清晰主體）
- **結果**: `results/pipeline_09_logpolar/`

---

### DiffBIR full-mode（noise start，非 SDEdit）

```
resize 1440 → DiffBIR (start_point_type=noise, strength=1.0, steps=30) → ESRGAN x4
```

- **測試圖**: 04, 10
- **說明**: DiffBIR 原生盲修復路徑（SwinIR cleaner + IRControlNet 從 noise 生成），行為與 SDEdit/cond 不同
- **效果**: 04 仍無效（blur 太重，cleaner 救不回結構）
- **結果**: `results/pipeline_diffbir_full/`

---

### FFTformer（頻域 transformer，CVPR 2023）

```
原圖 → PIL resize 1440 (+04 另跑 720) → FFTformer(GoPro) → ESRGAN x4
```

- **測試圖**: 04, 06, 07, 09
- **說明**: ideas.md 最後一個未試方法。頻域 self-attention (FSAS)，理論上對長距離 motion blur 優於 spatial。OOM 風險高 → tile=256 小塊推理（成功，未爆顯存）
- **效果**: 與 Restormer / NAFNet **三者幾乎一模一樣**，04/09 同樣無解，07 一樣銳利。→ 結論：**Restormer ≈ NAFNet ≈ FFTformer**，瓶頸是資料（blur 超出訓練分布）不是架構
- **結果**: `results/pipeline_fftformer/`；三方對比 `results/_final_compare/_3way_backbone.png`

---

### CodeFormer 人臉修復（新下載，CVPR 2022 NeurIPS）

```
既有最佳結果(10:v18 / 14:v14 / 15:v13) → CodeFormer (w=0.5 / 0.7, upscale=1, bg 不動)
```

- **測試圖**: 10, 14, 15
- **說明**: 只換臉、不動背景。w 越高越保真。比 GFPGAN 穩、不易把臉變成另一個人
- **效果**: **本輪最大收穫**。14 人臉再上一個檔次（互評首選）；10 右下大臉去蠟感、明顯變銳，勝過 v19(DiffBIR)
- **結果**: `results/codeformer_faces/`（w07 / w05）→ 候選 `z_candidates/`

---

## 特殊實驗

### DarkIR + Restormer（早期實驗）

```
原圖 → DarkIR → Restormer
```

- **測試圖**: 全 15 張
- **效果**: 見 `results/pipeline_DarkIR_Restormer/`

### Restormer + DiffBIR SDEdit（早期實驗）

```
原圖 → Restormer → DiffBIR SDEdit
```

- **測試圖**: 01, 03, 04, 07, 11

### Image 09 Polar Coordinate 去模糊

```
原圖 → PIL resize 1440 → warpPolar → PIL resize 1440 → OMDNet → warpPolar inverse → NLMeans → Sharpen → ESRGAN x4
```

- **說明**: 09 號圖片有 zoom blur，嘗試用極座標轉換將放射狀模糊轉為線性模糊，讓 OMDNet 更好處理

### Image 15 人臉修復實驗

- `v13_omd_15_face` / `v19_omd_15_face`: 15 號圖以 v13/v19 pipeline 加 OMDNet 處理人臉
- `*_diffbir`: 上述結果再經 DiffBIR + GFPGAN 增強
- `original_15_face`: 直接 resize → Restormer → ESRGAN+GFPGAN（不經 OMDNet）

---

## 實驗比較軸整理

### 低光增強方法比較

| 方法 | 效果 | 問題 |
|------|------|------|
| **HVI-CIDNet** | 增亮效果好，色彩自然 | tile 不好處理、可能放大噪點 |
| **CLAHE** | 無 artifact、速度快 | 增亮效果較弱、色彩偏移 |
| **DarkIR** | 同時低光+去模糊 | 需單獨測試效果 |
| **None** | — | 某些圖片不需增亮反而更乾淨 |

**結論**: 最終版 (v18) 不使用低光增強，因部分圖片增亮後噪點加劇反而更差。

### OMDNet 解析度比較

| 解析度 | 去模糊效果 | 細節保留 |
|--------|-----------|---------|
| **1440** | 更好的去模糊效果 | 細節稍少，靠 ESRGAN x4 補回 |
| **1920** | 去模糊較弱 | 細節較多 |

**結論**: 1440 + ESRGAN x4 為最終選擇。

### Diffusion 模型比較

| 方法 | 優點 | 缺點 |
|------|------|------|
| **PASD** (v1) | 細節增強 | hallucination 嚴重 |
| **SDEdit** (v2, v8a) | 較保守、可控 | 768px 解析度限制 |
| **DiffBIR** (v13+) | 修復品質好 | 高強度仍有 hallucination；文字扭曲 |

**結論**: DiffBIR s=0.8~0.85 效果最佳，但文字多的圖需跳過。

### 去模糊模型比較

| 方法 | 效果 |
|------|------|
| **MISCFilter** | 對 real-world motion blur 效果有限 |
| **Restormer** | 穩定可靠的去模糊 baseline |
| **OMDNet** | gate 機制選擇性去模糊，保護清晰區域 |
| **Wiener/RL Deconv** | 需精確 PSF，傷害清晰區域 |
| **UFPNet + MPRNet + Restormer cascade** | v21 的做法，堆疊多模型 |

### Sharpen 強度比較

| 強度 | 效果 |
|------|------|
| **1.0** | v8 使用，偏弱 |
| **1.5** | v9-v12 使用，偏強 |
| **0.5** | v15+ 使用，適中，配合 DiffBIR 使用 |

---

## 最終架構選擇

基於大量實驗，最終主要使用以下 pipeline：

- **一般圖片 (v18)**: `resize 1440 → Restormer → OMDNet → NLMeans → Sharpen(0.5) → ESRGAN x4`
- **人臉圖片 (v18/v19)**: 上述 + GFPGAN face enhance
- **文字多的圖片 (v15)**: `Restormer → OMDNet → Sharpen(0.5) → ESRGAN x4`（跳過所有 diffusion）
- **嚴重模糊 (v21)**: `resize 1440 → UFPNet → MPRNet → Restormer → OMDNet → ESRGAN x4`
- **嚴重模糊 + DiffBIR (v22)**: 低強度 DiffBIR (s=0.4~0.6) 嘗試修復

---

## 朋友的做法（參考）

```
MPRNet denoise → OMDNet (max_dim=1440, whole image)
```

- 不用 tiling、不用 ESRGAN
- OMDNet 直接整張圖處理

---

## 已知限制與踩坑

1. **xformers**: 絕對不要 `pip install xformers`，會破壞 CUDA 環境
2. **conda run**: 不要用 `conda run`，直接用 python.exe 完整路徑
3. **OMDNet OOM**: 高解析度需 tiling + cosine blending 或 fallback 機制
4. **DiffBIR + 文字**: DiffBIR 會扭曲文字，文字多的圖片必須跳過
5. **HVI-CIDNet tiling**: 需 cosine blending + overlap >= 128 才不會有格子
6. **PASD**: hallucination 太嚴重，已完全棄用
7. **Wiener/RL Deconv**: 需精確 PSF 估計，實際效果不佳，已棄用

---

## 目前最佳結果（z_results/）

以下為從各 pipeline 中挑選出相對最好的結果：

| 檔名 | 原圖 | 使用的 Pipeline | 說明 |
|------|------|----------------|------|
| `01_v13.png` | 01 都市行人光軌 | v13: Restormer → OMDNet(1440) → DiffBIR(s=0.8) → ESRGAN x4 | DiffBIR 修復人物與光軌細節 |
| `02_v21.png` | 02 玻璃反射招牌 | v21: resize 1440 → UFPNet → MPRNet → Restormer → OMDNet → ESRGAN x4 | Cascade 多模型堆疊處理散焦+反射 |
| `05_v13.png` | 05 紅色計程車 | v13: Restormer → OMDNet(1440) → DiffBIR(s=0.8) → ESRGAN x4 | DiffBIR 版本 |
| `05_v18.png` | 05 紅色計程車 | v18: resize 1440 → Restormer → OMDNet → NLMeans → Sharpen → ESRGAN x4 | 無 DiffBIR 版本，文字較清楚 |
| `06_v14.png` | 06 仰拍霓虹招牌 | v14: CLAHE → Restormer → OMDNet(1440) → DiffBIR(s=0.85) → ESRGAN x4 | CLAHE 增亮 + DiffBIR 修復嚴重手震 |
| `08_rest_omd_esrgan.png` | 08 KFC 外送員 | Restormer → OMDNet → ESRGAN（早期實驗） | 簡單三步效果已不錯 |
| `12_res_omd_esrgan.png` | 12 人行道標誌 | Restormer → OMDNet → ESRGAN（早期實驗） | 文字多，不用 DiffBIR 保留文字清晰 |
| `13_v16.png` | 13 FamilyMart 玻璃反射 | v16: DarkIR → Restormer → OMDNet(1440) → Sharpen → ESRGAN x4 | DarkIR 同時處理低光+去模糊 |
| `14_v14.png` | 14 黃椅巷弄女子 | v14: CLAHE → Restormer → OMDNet(1440) → DiffBIR(s=0.85) → ESRGAN x4 | 人臉+運動模糊，DiffBIR 修復效果好 |
| `15_rest_face_omd_sr.png` | 15 攝影師玻璃自拍 | resize → Restormer → GFPGAN → OMDNet → ESRGAN | 人臉專用處理 |
| `15_v13_s9.png` | 15 攝影師玻璃自拍 | v13 (s=0.9): Restormer → OMDNet(1440) → DiffBIR(s=0.9) → ESRGAN x4 | 較高 DiffBIR 強度修復人臉 |

### 尚未選出最佳結果的圖片

| # | 原圖 | 狀態 |
|---|------|------|
| 03 | 昏暗夜市走道 | 未選出 |
| 04 | 騎腳踏車嚴重模糊 | 未選出（極高難度，v20/v21/v22 皆嘗試過） |
| 07 | 黃色計程車霓虹街 | 未選出 |
| 09 | 白色卡車 zoom blur | 未選出（極高難度，嘗試過 polar deconv） |
| 10 | 擁擠夜市人群 | 未選出 |
| 11 | 福町夜市鬼影人群 | 未選出 |
