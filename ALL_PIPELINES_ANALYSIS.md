# all-pipelines 實驗分析與後續方法

日期：2026-06-10  
來源：`all-pipelines` 分支的 `experiment_log.md`、`OVERNIGHT_RESULTS.md`、`ideas.md`、`results/` 與 `z_*` 精選結果。

## 1. 最重要結論

1. **主力 deblur backbone 已不是瓶頸。** Restormer、NAFNet、FFTformer 三者在本資料集上幾乎同級；04、09 仍失敗，代表問題多半是模糊型態超出訓練分布，而不是 backbone 不夠新。
2. **DiffBIR 必須分流使用。** 它能補細節，但會破壞文字。07 的 v13 對照把車牌與中文招牌生成錯字，是最明確反例；文字圖、車牌圖、招牌圖要跳過 DiffBIR。
3. **人臉後處理用 CodeFormer 比 GFPGAN / DiffBIR 人臉路線穩。** 10、14、15 的最佳候選都證明 CodeFormer 能改善臉部銳利度，而且比較不會換臉。
4. **1440 max-dim 是目前最好的工程折衷。** OMDNet 在較低解析度去模糊更強，ESRGAN 再補解析度；1920 保細節但去模糊較弱。
5. **DarkIR 不應全圖無腦前置。** 13 的 DarkIR 結果有價值，但 03 的最佳解是 v18 後製增亮；一般圖不加低光增強反而更乾淨。

## 2. 每類圖片的策略

| 類型 | 圖片 | 建議 |
|---|---|---|
| 一般可修復夜景 | 01, 06, 14 | `resize1440 -> Restormer -> OMDNet -> DiffBIR(s=0.8~0.85) -> sharpen -> ESRGAN` |
| 文字/車牌/招牌多 | 05, 07, 08, 11, 12 | 跳過 DiffBIR，走 `Restormer -> OMDNet -> sharpen -> ESRGAN` |
| 低光/玻璃反射 | 13 | 可保留 DarkIR，但 DiffBIR 仍建議跳過，避免便利商店招牌與反射文字被改寫 |
| 人臉重點 | 10, 14, 15 | 最後加 CodeFormer；10 用 w=0.5 較銳，14/15 用 w=0.7 較保真 |
| failure case | 04, 09, 11 | 04/11 當不可逆資訊遺失案例；09 可展示 log-polar deconv 部分有效但不作主成品 |

## 3. 改善後的 DarkIR-Restormer-DiffBIR 方法

新版腳本：`code/pipeline/run_darkir_restormer_diffbir_v2.py`

預設流程：

```text
img/ input
-> PIL resize max_dim=1440
-> DarkIR gate：預設只處理 13，其餘直接複製
-> Restormer
-> OMDNet
-> DiffBIR gate：預設跳過 02,04,05,07,08,09,11,12,13
-> post-sharpen
-> Real-ESRGAN x4
```

這樣改的原因：

- DarkIR 改成 **selective low-light stage**，避免把所有夜景都拉平或放大噪點。
- DiffBIR 改成 **selective detail stage**，只處理適合生成式補細節的圖。
- 文字與車牌走保守路線，因為實驗已證明 diffusion 對可讀文字很危險。
- 04/09 預設不交給 DiffBIR，避免嚴重模糊被 hallucination 放大。

第 8 張 smoke test：

```bash
python code/pipeline/run_darkir_restormer_diffbir_v2.py --images 08 --skip_esrgan
```

全圖：

```bash
python code/pipeline/run_darkir_restormer_diffbir_v2.py
```

若想比較「全 DarkIR」和「selective DarkIR」：

```bash
python code/pipeline/run_darkir_restormer_diffbir_v2.py \
  --images 03,06,13,14,15 \
  --darkir_ids 03,06,13,14,15 \
  --out_dir results/pipeline_darkir_all_lowlight_v2
```

## 4. FFTformer 實測結論與 branch

已建立並切到新分支：`fftformer-colab-experiment`。

本分支中 `code/pipeline/run_fftformer.py` 已改成可指定輸入資料夾與圖片 ID 的實驗工具：

```bash
# 第 8 張 smoke test，先不跑 ESRGAN
python code/pipeline/run_fftformer.py --images 08 --skip_esrgan

# 重跑原本重點測試：04/06/07/09，並保留 04 的 720 低解析對照
python code/pipeline/run_fftformer.py --images 04,06,07,09 --extra_04_720
```

實驗判讀：

- FFTformer 的頻域 attention 理論上適合 motion blur，但本資料集實測並沒有突破 Restormer/NAFNet。
- `results/_final_compare/_3way_backbone.png` 是報告中最好的證據圖：三者近似，失敗圖同樣失敗。
- 後續不要把時間主要花在換 backbone；更值得做的是分流、遮罩、文字保護、人臉後處理與 failure-case 分析。

## 5. 建議報告表述

- 成功案例：14 + CodeFormer、07 保守文字路線。
- 方法貢獻：不是單一模型更強，而是 **依內容分流**：文字保守、人臉後修、低光選擇性、嚴重不可逆模糊不強行 hallucinate。
- 失敗案例：04/09/11 說明物理限制。這會比硬修出一張假圖更有說服力。
