"""把 get_data.py 的中間結果畫成數張獨立的切格流程圖，對應論文 3.3 節。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import CAT_COLORS, CATEGORIES  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'

out_dir = os.path.dirname(os.path.abspath(__file__))
cells = pd.read_csv(os.path.join(out_dir, 'cells.csv'))
local = pd.read_csv(os.path.join(out_dir, 'local.csv'))
patch = pd.read_csv(os.path.join(out_dir, 'patch.csv'))
count = pd.read_csv(os.path.join(out_dir, 'count.csv')).set_index('category').reindex(CATEGORIES)
s = pd.read_csv(os.path.join(out_dir, 'selected.csv')).iloc[0]

STEP, HALF, N_SIDE = s['cell_size'], s['half_width'], int(s['n_side'])
x0, y0 = s['x0'], s['y0']


def save(fig, name):
    """fig：matplotlib figure。name：輸出檔名（相對本資料夾）。回傳：None。"""
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    print(name)


# ---- 3.3.1 全區：哪些格點被留下來當 patch 中心 ----
fig, ax = plt.subplots(figsize=(7.5, 7.5))
drop = cells[cells['kept'] == 0]
keep = cells[cells['kept'] == 1]
ax.scatter(drop['cell_x'] * STEP / 1000, drop['cell_y'] * STEP / 1000, s=0.4,
           color='#cccccc', label=f'Discarded, POI < {int(s["min_poi"])} ({len(drop):,})')
ax.scatter(keep['cell_x'] * STEP / 1000, keep['cell_y'] * STEP / 1000, s=2.5,
           color='#e6194b', label=f'Kept as patch ({len(keep):,})')
ax.scatter([x0 / 1000], [y0 / 1000], s=160, facecolor='none', edgecolor='#3cb44b',
           linewidth=2, label='Zoom of the next figure')
ax.set_xlabel('EPSG:6677 X (km)')
ax.set_ylabel('EPSG:6677 Y (km)')
ax.set_aspect('equal')
ax.legend(fontsize=9, loc='upper left', markerscale=2.5)
ax.set_title(f'Step 1-2: project, cut into {int(STEP)} m cells,\n'
             f'keep only cells with at least {int(s["min_poi"])} POIs')
save(fig, '0.6.1_kept_centers.png')

# ---- 3.3.2 局部：格線怎麼落、選中的是哪一格 ----
fig, ax = plt.subplots(figsize=(7, 7))
for cat, color in zip(CATEGORIES, CAT_COLORS):
    sub = local[local['category'] == cat]
    ax.scatter(sub['x'], sub['y'], s=44, alpha=0.85, color=color, zorder=3, label=cat)
for i in range(N_SIDE + 1):
    ax.axvline(x0 + i * STEP, color='gray', linewidth=0.9, zorder=1)
    ax.axhline(y0 + i * STEP, color='gray', linewidth=0.9, zorder=1)
ax.add_patch(Rectangle((s['cell_x'] * STEP, s['cell_y'] * STEP), STEP, STEP,
                       facecolor='#e6194b', alpha=0.13, edgecolor='#77021c',
                       linewidth=3, zorder=2))
ax.scatter([s['center_x']], [s['center_y']], marker='+', s=220, color='#77021c',
           linewidth=2.5, zorder=4)
ax.set_xlim(x0, x0 + N_SIDE * STEP)
ax.set_ylim(y0, y0 + N_SIDE * STEP)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
ax.set_title(f'Step 2: a {N_SIDE}x{N_SIDE} block of {int(STEP)} m cells '
             f'({len(local)} POIs)\nred = the example cell, cross = its centre')
save(fig, '0.6.2_local_grid.png')

# ---- 3.3.3 單一 patch：座標改成相對格心的位移 ----
fig, ax = plt.subplots(figsize=(6, 6))
for cat, color in zip(CATEGORIES, CAT_COLORS):
    sub = patch[patch['category'] == cat]
    ax.scatter(sub['dx'], sub['dy'], s=110, alpha=0.9, color=color, zorder=3, label=cat)
ax.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False,
                       edgecolor='#77021c', linewidth=3, zorder=2))
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
ax.scatter([0], [0], marker='+', s=220, color='#77021c', linewidth=2.5, zorder=4)
ax.set_xlim(-HALF * 1.18, HALF * 1.18)
ax.set_ylim(-HALF * 1.18, HALF * 1.18)
ax.set_aspect('equal')
ax.set_xlabel('$\\Delta x$ from the cell centre (m)')
ax.set_ylabel('$\\Delta y$ from the cell centre (m)')
ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
ax.set_title(f'Step 3: one patch = Chebyshev radius {HALF:.0f} m\n'
             f'({int(s["n_poi"])} POIs, centre {s["center_lat"]:.4f}, {s["center_lon"]:.4f})')
save(fig, '0.6.3_patch_window.png')

# ---- 3.3.4 丟掉幾何，只留類別計數 ----
fig, ax = plt.subplots(figsize=(7.5, 4.5))
v = count['count'].values
pos = np.arange(len(CATEGORIES))[::-1]
ax.barh(pos, v, color=CAT_COLORS, height=0.72)
for p, n in zip(pos, v):
    ax.text(n + 0.12, p, str(int(n)), va='center', fontsize=10)
ax.set_yticks(pos)
ax.set_yticklabels(CATEGORIES, fontsize=9)
ax.set_xlim(0, max(v.max() * 1.25, 1))
ax.set_xlabel('POI count')
ax.set_title(f'Step 4: drop the geometry, keep the {len(CATEGORIES)}-dim count vector\n'
             f'x = {v.tolist()}')
save(fig, '0.6.4_count_vector.png')
