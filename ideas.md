# 改進方向與待嘗試清單（2026-06-10）

> 針對尚未選出最佳結果的 6 張：03, 04, 07, 09, 10, 11。
> 截止：06/11 上傳 2 張互評圖 + PPT。

---

## 0. 策略層面（最重要）

**互評只需要交 2 張。** 不要再把時間沉在 04/09 這種物理上救不回來的圖上面——它們的價值在書面報告（當作 failure case 分析），不在互評分數。

- **互評建議交**: 06（嚴重手震 → v14 修復，before/after 對比最震撼）和 14（人臉+運動模糊 → v14，人眼對人臉最敏感，修好了很加分）。05/07 也是備選（車牌文字從糊到可讀，很直觀）。
- 04/09/11 在報告裡寫「為什麼難」：blur kernel 長度超過訓練分布（04）、zoom blur 非均勻（09）、長曝光鬼影是資訊永久遺失而非卷積模糊（11）。這種分析在書面報告（40%）比硬擠出一張爛圖更值錢。

---

## 1. 逐張建議

### 03 昏暗夜市 — 問題是「暗」不是「糊」

v18 的結果去模糊已經夠了，就是太暗。之前的結論是「先增亮會放大噪點」，但**在 pipeline 最後增亮**就沒這個問題（噪點已經被 Restormer/NLMeans 清掉了）：

```
v18 最終結果 → 溫和 CLAHE (clipLimit~1.5) 或 gamma 校正 (γ≈0.7~0.8)
```

- 成本極低（幾秒鐘），先試這個。
- 進階版：只對暗部增亮（luminance mask 上做 curve），保留燈泡的氛圍不過曝。
- 備選：拿 v16 (DarkIR) 的 03 來比較。

### 04 騎腳踏車 — blur 太長，模型訓練分布外

目前 v21 幾乎沒改善、v22 (DiffBIR) 直接變抽象畫。兩個還沒真正試過的方向：

1. **更激進的 downscale**：v5 已經發現「1/2 解析度 → 3x 去模糊效果」。04 的 blur 長度目視超過 80px，縮到 1440 還是太長。試 **max_dim = 480~720** 跑 Restormer/OMDNet，blur kernel 等比例縮短到模型能處理的範圍，再用 DiffBIR(s=0.4~0.5) + ESRGAN 補解析度。低解析度時 DiffBIR 的 hallucination 反而比較不明顯。
2. **承認它是 panning shot**：背景的水平拖尾其實是「追焦攝影」的美感，真正該救的只有騎士。
   ```
   SAM/手動遮罩騎士 → 只對騎士區域 deconv(165°)+Restormer → 貼回原圖 → 邊緣 feather blending
   ```
   背景保留原樣，視覺上「主體清楚+速度感」比全圖硬修自然得多。互評者看到的是一張好看的照片，不是一張修壞的照片。
3. 還沒試的模型：**FFTformer**（頻域方法，理論上對長 motion blur 比 spatial 方法強）、**NAFNet**（repo 已經 clone 了但實驗紀錄裡完全沒出現！）。

### 07 黃色計程車 — 其實已經可以收了

v18 的 07 看起來很好：車體銳利、TDC-1769 車牌可讀、背景 panning blur 有速度感。建議：

- 直接把 `pipeline_v18/step6_realesrgan/07.png` 收進 z_results。
- 如果想再比較：跑一張 v13 (DiffBIR s=0.8) 版本對比車牌/招牌文字有沒有被扭曲，沒有的話 DiffBIR 版的車體質感可能更好。07 是互評備選，值得這 10 分鐘。

### 09 白色卡車 zoom blur — 用 log-polar 而不是 linear polar

之前的 polar 實驗方向對了但少一步：**zoom blur 在 linear polar 下，blur 長度仍隨半徑變化**（離中心越遠拖尾越長），OMDNet 還是處理不了非均勻 blur。但在 **log-polar**（`cv2.warpPolar` 加 `WARP_POLAR_LOG` flag）下，純 zoom blur 變成**沿半徑軸的均勻平移模糊** → 一個固定 1D PSF 的 Wiener/RL deconv 就能處理：

```
估計 zoom 中心（找最清晰點，約在卡車駕駛座附近）
→ warpPolar (LOG) → 1D 水平 RL deconv（試 kernel 長度 10~30px）
→ inverse warpPolar → Restormer 清 ringing → ESRGAN
```

- 中心估計很關鍵，偏了會整個歪掉。可以對原圖做梯度能量圖找最大值。
- 備案：zoom blur 中心區域本來就最清楚——**裁切中心 60% 區域**修一張「局部結果」，報告裡說明 zoom blur 的特性。
- 如果都不行就放掉，09 留作報告的 failure case 分析（zoom blur = spatially-variant kernel，所有 uniform-kernel 方法理論上都失效）。

### 10 擁擠夜市人群 — 臉是重點，現在太蠟

v18 的 10 臉部偏軟、有蠟感（NLMeans + ESRGAN 平滑掉了皮膚紋理）。

1. 跑 **v19**（DiffBIR s=0.85 + GFPGAN）的 10 號——log 上 v19 只列了測試但 z_results 沒收，先看看結果。
2. **CodeFormer 取代 GFPGAN**：對嚴重退化的人臉通常比 GFPGAN 穩（GFPGAN 容易把臉修成另一個人），fidelity weight 調 0.7~0.8。
3. 右前方大臉是視覺焦點：**裁出來單獨修**（crop → Restormer → CodeFormer → 貼回 + feather），比全圖 face enhance 對小臉亂修安全。
4. 跳過 NLMeans（10 的噪點不重，denoise 是蠟感主因之一）。

### 11 福町夜市鬼影 — 鬼影救不了，換個目標

長曝光的半透明人影是**資訊永久遺失**（多個時刻疊加），任何 deblur 都無法恢復，DiffBIR 硬修只會生成假人（恐怖谷）。建議直接轉換目標：

- **把靜態結構當主體**：牌樓、招牌「福町夜市」、兩側攤位本來就清楚，用 **v15 文字版 pipeline**（無 DiffBIR）把這些修到極致銳利，鬼影人群保留當作長曝光的藝術效果。
- v18 的 11 看起來已經接近這個狀態，跟 v15 版本並排比一下招牌文字，選銳的那張收進 z_results。
- 報告裡明確寫：ghosting ≠ blur，這是 deblurring 的理論邊界。

---

## 2. 還沒動用的武器（時間允許再試）

| 工具 | 為什麼值得 | 適用圖 | 4060 8G 可行性 |
|------|-----------|--------|----------------|
| **NAFNet** | repo 已 clone 卻從未出現在實驗紀錄；GoPro benchmark 上比 Restormer 強 | 04, 09, 全部 | ✅ 和 Restormer 同量級，套用既有 tiling 程式碼即可 |
| **FFTformer** (CVPR 2023) | 頻域 transformer，對長距離 motion blur 理論上優於 spatial 方法 | 04, 06 | ⚠️ 唯一高風險：頻域 attention 高解析度下 VRAM 暴增，1440 全圖必 OOM，官方 tiling 支援不完整 → 最低優先 |
| **CodeFormer** | 比 GFPGAN 更穩的人臉修復 | 10, 14, 15 | ✅ 臉部以 512 crop 處理，約 3~4GB，依賴 basicsr（環境已有） |
| **SAM 遮罩 + 選擇性處理** | 主體/背景分開處理，背景 blur 當美感保留 | 04, 07, 01 | ✅ ViT-B 約 4GB / MobileSAM <1GB；**單張圖建議直接手畫遮罩更快** |
| **DiffBIR 完整模式** | 目前都用 SDEdit 模式；DiffBIR 本身的 stage-1 (BSRNet/SwinIR) + IRControlNet 是專為 blind restoration 設計的，行為和 SDEdit 不同 | 04, 10 | ✅ SDEdit 模式已能跑，峰值差不多，記得開 VAE tiling |

**硬體註記**：log-polar deconv、CLAHE/gamma、手動遮罩 blending 都是 CPU 操作，零 VRAM；04 的低解析度實驗（480~720）比現行 1440 更輕。優先清單 1~7 項全部在 8G 內安全，唯一紅線是 FFTformer。

---

## 3. 明天（06/11）前的優先順序

1. ☐ **選定互評 2 張**（建議 06 + 14，或換 05/07）— 這是分數所在
2. ☐ 03：v18 結果 + 後置增亮（10 分鐘）
3. ☐ 07：直接收 v18，順手跑 v13 對比（10 分鐘）
4. ☐ 11：v15 vs v18 並排比招牌，收一張（5 分鐘）
5. ☐ 10：跑 v19 的 10 號 / CodeFormer（~30 分鐘）
6. ☐ 04：低解析度實驗（480/720）（~30 分鐘）
7. ☐ 09：log-polar deconv（風險高，時間剩才做）
8. ☐ PPT：before/after 並排 + pipeline 圖 + failure case 分析
