# 0.6 網格切分方式 → 論文 3.3

切法的完整說明與圖的擺放位置寫在 [../3_dataset/README.md](../3_dataset/README.md) 的 3.3 節，這裡只記怎麼跑。

## 執行

```bash
python experiment/0.6_grid_construction/get_data.py
python experiment/0.6_grid_construction/draw.py
```

## `get_data.py` 做什麼

原地重跑 `data/patch/build_patches.py` 的流程（投影 → 切 100 m 格 → 取 Chebyshev 50 m
鄰域 → `MIN_POI` 篩選 → 聚合成計數向量），把每一步的中間結果存下來：

- `cells.csv`：每個佔用格點的座標、POI 數、是否保留
- `local.csv`：範例區 3×3 格內的 POI
- `patch.csv`：範例 patch 內 POI 相對格心的位移
- `count.csv`：範例 patch 的 10 維類別計數向量
- `selected.csv`：範例格的編號、格心座標與所有幾何參數

範例格是程式依規則挑的（八鄰格都有 POI、類別數不低於中位數、POI 數最接近中位數），
不是手選特例，所以換資料或換參數重跑會自己選出新的典型格。

## `draw.py` 產出的圖

論文正文用的圖已經在 [../3_dataset/](../3_dataset/) 的 notebook 重畫過（尺寸調整請去那裡）；這裡的版本留著當草稿與備用。

| 檔名 | 論文 | 內容 |
|---|---|---|
| `0.6.1_kept_centers.png` | 備用 | 全區格點：灰＝丟棄 17,925，紅＝保留 1,233 |
| `0.6.2_local_grid.png` | **圖 3.3.1** | 局部 3×3 格與範例 patch、格心 |
| `0.6.3_patch_window.png` | **圖 3.3.2** | 範例 patch 的 ±50 m 相對座標與 Chebyshev 視窗 |
| `0.6.4_count_vector.png` | 備用 | 聚合成的 10 維計數向量 |
