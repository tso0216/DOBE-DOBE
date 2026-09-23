"""重跑一次 data/patch/build_patches.py 的切格流程，把中間結果留下來，
用來畫「怎麼把 POI 切成 patch」的示意圖。"""
import os
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import (CAT_COL, CATEGORIES, CENTER_STEP, CRS,  # noqa: E402
                            CSV, HALF_WIDTH, MIN_POI, N_CAT)

N_SIDE = 3      # 局部示意圖要畫幾乘幾個格子

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(CSV)
tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
x, y = tr.transform(df["lon"].values, df["lat"].values)
cats = df[CAT_COL].values

# ---- 步驟一：投影後用 CENTER_STEP 切格，統計每格 POI 數 ----
cell = np.floor(np.column_stack([x, y]) / CENTER_STEP).astype(np.int64)
cells, inv, n_poi = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
kept = n_poi >= MIN_POI
pd.DataFrame({"cell_x": cells[:, 0], "cell_y": cells[:, 1],
              "n_poi": n_poi, "kept": kept.astype(int)}).to_csv(
    os.path.join(out_dir, "cells.csv"), index=False)

# ---- 步驟二：挑一格當範例。要求八個鄰格都有 POI、類別數不少於中位數，
# 且自身 POI 數最接近中位數，這樣示意圖才既典型又看得出機能混合 ----
cat_index = {c: i for i, c in enumerate(CATEGORIES)}
cat_id = np.array([cat_index[c] for c in cats])
onehot = np.zeros((len(cells), N_CAT), dtype=bool)
onehot[inv, cat_id] = True
n_cat = onehot.sum(axis=1)

occupied = {tuple(c) for c in cells}
neighbors = np.array([
    sum(((c[0] + i, c[1] + j) in occupied)
        for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
    for c in cells])

med = np.median(n_poi[kept])
med_cat = np.median(n_cat[kept])
ok = kept & (neighbors == 8) & (n_cat >= med_cat)
cand = np.flatnonzero(ok if ok.any() else kept)
pick = cand[np.argmin(np.abs(n_poi[cand] - med))]
sel = cells[pick]

# ---- 步驟三：以選中格為中心取 N_SIDE x N_SIDE 的局部區域 ----
origin = sel - N_SIDE // 2
x0, y0 = origin * CENTER_STEP
x1, y1 = x0 + N_SIDE * CENTER_STEP, y0 + N_SIDE * CENTER_STEP
m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
pd.DataFrame({"x": x[m], "y": y[m], "category": cats[m]}).to_csv(
    os.path.join(out_dir, "local.csv"), index=False)

# ---- 步驟四：選中格的 patch。中心為格心，視窗是 Chebyshev 半徑 HALF_WIDTH ----
center = (sel + 0.5) * CENTER_STEP
inside = (np.abs(x - center[0]) <= HALF_WIDTH) & (np.abs(y - center[1]) <= HALF_WIDTH)
pd.DataFrame({"dx": x[inside] - center[0], "dy": y[inside] - center[1],
              "category": cats[inside]}).to_csv(
    os.path.join(out_dir, "patch.csv"), index=False)

# ---- 步驟五：丟掉幾何、只留類別計數，就是模型的輸入向量 ----
vec = np.bincount([cat_index[c] for c in cats[inside]], minlength=N_CAT)
pd.DataFrame({"category": CATEGORIES, "count": vec}).to_csv(
    os.path.join(out_dir, "count.csv"), index=False)

inv_tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
clon, clat = inv_tr.transform(center[0], center[1])
pd.DataFrame([{"cell_x": sel[0], "cell_y": sel[1],
               "center_x": center[0], "center_y": center[1],
               "center_lat": clat, "center_lon": clon,
               "n_poi": int(inside.sum()), "x0": x0, "y0": y0,
               "n_side": N_SIDE, "cell_size": CENTER_STEP,
               "half_width": HALF_WIDTH, "min_poi": MIN_POI}]).to_csv(
    os.path.join(out_dir, "selected.csv"), index=False)

print(f"佔用格點 {len(cells)}，通過 MIN_POI={MIN_POI} 的 {kept.sum()}")
print(f"範例格 (cell_x, cell_y) = {tuple(sel)}，格心 ({clat:.5f}, {clon:.5f})")
print(f"局部 {N_SIDE}x{N_SIDE} 區域內 {m.sum()} 個 POI，patch 視窗內 {inside.sum()} 個")
print(f"類別計數向量 {vec.tolist()}")
