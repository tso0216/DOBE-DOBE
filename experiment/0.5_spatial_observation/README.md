# 0.5 空間分布觀察 → 論文 3.2

觀察內容與圖的擺放位置寫在 [../3_dataset/README.md](../3_dataset/README.md) 的 3.2 節，這裡只記怎麼跑。

## 執行

```bash
python experiment/0.5_spatial_observation/get_data.py
python experiment/0.5_spatial_observation/draw.py
```

## `get_data.py` 怎麼算

- 全部 POI 投影到 EPSG:6677 後，用 250 m 網格算密度、用 1 km 粗網格挑放大區與做密度分層
- 「都心」＝ POI 數最多的那個 1 km 格（1,402 個 POI），徑向剖面與距離都以它為原點
- 三塊放大區都是 800 × 800 m，分別取 1 km 格 POI 數的 p100 / p90 / p50
- 密度分層＝把 1,195 個 1 km 格依 POI 數切成五等分，各層分別算類別組成
- 寫出 `density.csv`、`zoom.csv`、`radial.csv`、`composition.csv`、`core.csv`

## `draw.py` 產出的圖

論文正文用的圖已經在 [../3_dataset/](../3_dataset/) 的 notebook 重畫過（尺寸調整請去那裡）；這裡的版本留著當草稿與備用。

| 檔名 | 論文 | 內容 |
|---|---|---|
| `0.5.1_density_map.png` | **圖 3.2.2** | 250 m 網格的 POI 密度熱圖，標出 A/B/C 放大區 |
| `0.5.2_zoom_windows.png` | **圖 3.2.3** | 三塊 800 × 800 m 放大區並排（p100 / p90 / p50） |
| `0.5.3_radial_profile.png` | **圖 3.2.4** | 離都心的徑向密度剖面（log 縱軸） |
| `0.5.4_composition_by_density.png` | **圖 3.2.5** | 五個密度分層各自的類別組成 |
