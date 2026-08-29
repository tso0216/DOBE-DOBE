"""繪製真實 POI 點位，並疊上與 common/dataset.py 一致的 100 公尺格點，
框出一小片區域（約 16 個格子），其中一格特別標示為選中的格子。
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from pyproj import Transformer

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import CAT_COL, CAT_COLORS, CATEGORIES, CRS, CSV, CENTER_STEP  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

N_SIDE = 3  # 3x3 = 9 個格子

out_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(CSV)

transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
x, y = transformer.transform(df['lon'].values, df['lat'].values)

cell_x = np.floor(x / CENTER_STEP).astype(np.int64)
cell_y = np.floor(y / CENTER_STEP).astype(np.int64)

# 用 POI 數最多的格子當中心，往左下延伸出一塊 N_SIDE x N_SIDE 的區域
cells, counts = np.unique(np.column_stack([cell_x, cell_y]), axis=0, return_counts=True)
busiest = cells[np.argmax(counts)]
origin = busiest - 1  # 讓最熱鬧的格子落在區塊中央附近
selected = busiest

x0, y0 = origin * CENTER_STEP
x1, y1 = x0 + N_SIDE * CENTER_STEP, y0 + N_SIDE * CENTER_STEP

mask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
xs, ys, cats = x[mask], y[mask], df[CAT_COL].values[mask]

fig, ax = plt.subplots(figsize=(8, 8))
for cat, color in zip(CATEGORIES, CAT_COLORS):
    sub = cats == cat
    ax.scatter(xs[sub], ys[sub], s=50, alpha=0.7, color=color, zorder=2)

for i in range(N_SIDE + 1):
    ax.axvline(x0 + i * CENTER_STEP, color='gray', linewidth=0.8, zorder=1)
    ax.axhline(y0 + i * CENTER_STEP, color='gray', linewidth=0.8, zorder=1)

sel_x, sel_y = selected * CENTER_STEP
ax.add_patch(Rectangle(
    (sel_x, sel_y), CENTER_STEP, CENTER_STEP,
    facecolor='#e6194b', alpha=0.15, edgecolor="#77021c", linewidth=8, zorder=1.5))

ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '0.4_grid_illustration.png'), dpi=250, bbox_inches='tight')
print(f"框選區域內共 {mask.sum()} 個 POI，選中格子座標 (cell_x, cell_y) = {tuple(selected)}")
