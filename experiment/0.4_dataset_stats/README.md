# 0.4 資料集統計 → 論文 3.1

內容與圖的擺放位置寫在 [../3_dataset/README.md](../3_dataset/README.md) 的 3.1 節，這裡只記怎麼跑。

## 執行

```bash
python experiment/0.4_dataset_stats/get_data.py
python experiment/0.4_dataset_stats/draw.py
```

## `get_data.py` 讀什麼、寫什麼

- 讀：`data/tky_raw.txt`（原始打卡）、`data/tky_clean.csv`（清理後 POI）、
  `data/patch/patches.npz`（切好的 patch）
- 寫：
  - `summary.csv`：純量摘要（各階段數量、時間跨度、bbox、切格參數與結果）
  - `describe.csv`：六個量的敘述統計（mean / std / min / p10 / p25 / median / p75 / p90 / p99 / max）
  - `category.csv`：10 大類的 POI 數、細類數、打卡數、佔比、在幾 % 的 patch 出現過
  - `checkin.csv`、`cell.csv`、`patch.csv`：畫圖用的原始數列

## `draw.py` 產出的圖

論文正文用的圖已經在 [../3_dataset/](../3_dataset/) 的 notebook 重畫過（尺寸調整請去那裡）；這裡的版本留著當草稿與備用。

| 檔名 | 論文 | 內容 |
|---|---|---|
| `0.4.1_pipeline.png` | 未用（改寫成表 3.1） | 各處理階段的資料量（log 縱軸） |
| `0.4.2_checkin_ranksize.png` | **圖 3.1.2** | 每 POI 打卡次數的 rank–size（log–log） |
| `0.4.3_category_share.png` | **圖 3.1.3** | 類別佔比：POI 數 vs 打卡數 |
| `0.4.4_cell_poi.png` | **圖 3.3.3** | 100 m 格點的 POI 數與 `MIN_POI` 門檻 |
| `0.4.5_patch_poi.png` | **圖 3.1.4** | 每個 patch 的 POI 數 |
| `0.4.6_categories_per_patch.png` | **圖 3.1.5** | 每個 patch 出現幾種類別 |
| `0.4.7_entropy_per_patch.png` | **圖 3.1.6** | 每個 patch 的 Shannon entropy |
