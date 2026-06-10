# 隔夜實驗結果總結 (2026-06-10 夜)

> 依 `ideas.md` 把所有方法跑過一輪。以下是每張圖的最佳結果與建議。
> 候選圖都在 `z_candidates/`，對比圖在 `results/_final_compare/`。

---

## ⭐ 互評 2 張建議：**14 + 07**

> **兩張全解析度已放在 `互評上傳_recommended/`，早上直接上傳即可**
> （`14_portrait_restored.png`、`07_taxi_deblurred.png`）。要換再從下表挑。

| 圖 | 結果檔 | 為什麼選它 |
|----|--------|-----------|
| **14** 黃椅女子 | `z_candidates/14_v14_cf07.png` | **before/after 最震撼**：原圖人物幾乎是鬼影，修復後人臉清晰、椅子/背景銳利。新加的 **CodeFormer** 把臉部細節再拉一個檔次。 |
| **07** 黃色計程車 | `z_results/07_v18.png` | **最乾淨的技術示範**：車牌「TDC-1769」「萬世旺」從糊到可讀，車身銳利，背景保留水平 panning 速度感。臉 + 車兩種 deblur 一起秀，多樣性高。 |

> 原本 ideas.md 建議 06+14，但實測 06 的 before/after 改善幅度「中等」（霓虹晃動殘留多），
> 07 的「車牌可讀」對比更直觀、更有說服力，所以改推 **14 + 07**。最終由你決定。

---

## 各圖最佳結果

| # | 最佳檔案 | 方法 | 備註 |
|---|---------|------|------|
| 03 | `z_candidates/03_v18_brightened.png` | v18 + shadow-lift + 溫和CLAHE | 去模糊本來就夠，純粹後製增亮、保留暖色燈泡 |
| 07 | `z_results/07_v18.png` | v18 | 車牌可讀，强候選 |
| 10 | `z_candidates/10_v18_cf05.png` | v18 + **CodeFormer w0.5** | 人臉去蠟感，右下大臉明顯變銳利，勝過 v19(DiffBIR) |
| 11 | `z_candidates/11_v15.png` | v15 文字版 | 招牌「福町夜市」最清楚；鬼影人群無法救（報告 failure case） |
| 14 | `z_candidates/14_v14_cf07.png` | v14 + **CodeFormer w0.7** | 互評首選 |
| 15 | `z_candidates/15_v13_cf07.png` | v13 + CodeFormer w0.7 | 玻璃後人臉，輕度改善 |

---

## 新嘗試的方法與結論（給報告用）

1. **CodeFormer 人臉修復**（新下載）：對 10/14/15 都有效，比 GFPGAN 穩、不易變成另一個人。w0.7 保真、w0.5 更銳但略「美顏感」。→ **最大收穫**。
2. **NAFNet (GoPro, CVPR2022)**（首次使用）：當 Restormer 的替代 backbone，實測與 Restormer **效果相當**，沒有明顯勝出，對 04/09 一樣無解。
2b. **FFTformer (頻域 transformer, CVPR2023)**（最後一個未試方法）：tile=256 成功跑完（未 OOM）。與 Restormer/NAFNet **三者幾乎一模一樣** → **瓶頸是資料不是架構**。三方對比 `results/_final_compare/_3way_backbone.png`
3. **04 激進降解析度 (480/720)**：理論上縮短 blur kernel，但騎士資訊已遺失，仍是鬼影 → 無突破。
4. **09 log-polar 去卷積**（修正版）：log-polar 下 zoom blur 確實大致變成水平均勻模糊，1D RL deconv 能讓「Sita.co.uk」字樣變清楚一些，但邊緣放射狀殘留 + ringing。→ 方法部分有效，可當報告亮點，但非成品級。fallback 是中央裁切。
5. **DiffBIR full-mode (noise start)**：與 SDEdit 模式不同的原生盲修復路徑；對 04 仍無效（blur 太重）；對 10 與 CodeFormer 版相當但有 hallucinate 人群風險，不採用。
6. **07 v13 (DiffBIR s0.8) 對照**：DiffBIR 把車牌「TDC-1769」整個生成成「HAAFIP」、「萬世旺」變亂碼 → **文字完全被幻覺破壞**。坐實「文字圖必須跳過 DiffBIR」，是報告很好的反例。所以 07 確定用 **v18**。
   對照圖：`results/_final_compare/_07_v18_vs_v13.png`

### Failure cases（報告 40% 的價值所在）
- **04 騎士**：blur kernel 長度超出所有模型訓練分布，且非單純 panning（主體也糊），資訊永久遺失。所有方法皆無解。
- **09 卡車 zoom blur**：spatially-variant kernel，uniform-kernel 方法理論上失效；log-polar 部分緩解但有 ringing。
- **11 鬼影人群**：長曝光多時刻疊加 = 資訊永久遺失，deblur 理論邊界。

---

## 對比圖位置
- 互評候選 before/after：`results/_final_compare/14_ba.png`, `_others_ba.png`(含07)
- 07 車牌特寫：`results/_final_compare/_07_plate.png`
- CodeFormer 人臉對比：`results/codeformer_faces/_faces07.png`, `_full14.png`, `_10bigface.png`
- 09 log-polar：`results/pipeline_09_logpolar/_cmp.png`, `_cmp2.png`, `diag_logpolar.png`
- 04 低解析度：`results/pipeline_04_lowres/_cmp.png`
- NAFNet vs Restormer：`results/_cmp_naf_vs_v18.png`
