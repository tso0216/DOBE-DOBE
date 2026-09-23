"""把 get_data.py 的輸出畫成數張獨立的觀察圖，對應論文 3.2 節。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import CAT_COLORS, CATEGORIES  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'

DENSITY_CELL = 250
ZOOM_SIDE = 800
TAG = ['A', 'B', 'C']

out_dir = os.path.dirname(os.path.abspath(__file__))
dens = pd.read_csv(os.path.join(out_dir, 'density.csv'))
zoom = pd.read_csv(os.path.join(out_dir, 'zoom.csv'))
radial = pd.read_csv(os.path.join(out_dir, 'radial.csv'))
comp = pd.read_csv(os.path.join(out_dir, 'composition.csv'))
pcts = zoom['zoom_pct'].unique()


def save(fig, name):
    """fig：matplotlib figure。name：輸出檔名（相對本資料夾）。回傳：None。"""
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    print(name)


# ---- 3.2.1 密度熱圖 ----
fig, ax = plt.subplots(figsize=(8, 8))
gx, gy = dens['gx'].values, dens['gy'].values
img = np.full((gy.max() - gy.min() + 1, gx.max() - gx.min() + 1), np.nan)
img[gy - gy.min(), gx - gx.min()] = dens['n_poi'].values
extent = [gx.min() * DENSITY_CELL / 1000, (gx.max() + 1) * DENSITY_CELL / 1000,
          gy.min() * DENSITY_CELL / 1000, (gy.max() + 1) * DENSITY_CELL / 1000]
im = ax.imshow(img, origin='lower', extent=extent, cmap='magma',
               norm=LogNorm(vmin=1, vmax=np.nanmax(img)), interpolation='nearest')
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
             label=f'POIs per {DENSITY_CELL} m cell (log scale)')

for tag, pct in zip(TAG, pcts):
    z = zoom[zoom['zoom_pct'] == pct].iloc[0]
    cx, cy, side = z['cx'] / 1000, z['cy'] / 1000, ZOOM_SIDE / 1000
    ax.add_patch(Rectangle((cx - side / 2, cy - side / 2), side, side, fill=False,
                           edgecolor='#3cb44b', linewidth=2))
    ax.annotate(tag, (cx, cy), xytext=(cx + 3.0, cy + 3.0), color='#3cb44b',
                fontsize=15, ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.28', facecolor='white',
                          edgecolor='#3cb44b', linewidth=1.5),
                arrowprops=dict(arrowstyle='-', color='#3cb44b', linewidth=1.2))
ax.set_xlabel('EPSG:6677 X (km)')
ax.set_ylabel('EPSG:6677 Y (km)')
ax.set_aspect('equal')
ax.set_title(f'POI density over Tokyo\n'
             f'(max {int(np.nanmax(img))}, median {int(np.median(dens["n_poi"]))} '
             f'POIs per {DENSITY_CELL} m cell; white = no POI)')
save(fig, '0.5.1_density_map.png')

# ---- 3.2.2 三塊同尺寸放大區 ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
for ax, tag, pct in zip(axes, TAG, pcts):
    z = zoom[zoom['zoom_pct'] == pct]
    for cat, color in zip(CATEGORIES, CAT_COLORS):
        sub = z[z['category'] == cat]
        ax.scatter(sub['dx'], sub['dy'], s=18, alpha=0.85, color=color,
                   label=cat if ax is axes[0] else None)
    ax.set_xlim(-ZOOM_SIDE / 2, ZOOM_SIDE / 2)
    ax.set_ylim(-ZOOM_SIDE / 2, ZOOM_SIDE / 2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{tag}  |  1 km-cell density p{pct}  |  {len(z)} POIs\n"
                 f"{z['dist_core_km'].iloc[0]:.1f} km from the core", fontsize=10)
axes[0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, -0.04),
               ncol=3, markerscale=1.8, frameon=False)
fig.suptitle(f'Three {ZOOM_SIDE} x {ZOOM_SIDE} m windows at different densities', y=0.98)
save(fig, '0.5.2_zoom_windows.png')

# ---- 3.2.3 徑向密度剖面 ----
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(radial['r_km'], radial['density'], 'o-', color='#e6194b', markersize=4)
ax.set_yscale('log')
ax.set_xlabel('Distance from the densest 1 km cell (km)')
ax.set_ylabel('POIs / km$^2$ (log scale)')
ax.set_title(f"Radial density profile: {radial['density'].iloc[0]:.0f} at the core, "
             f"{radial[radial['r_km'] > 20]['density'].mean():.1f} beyond 20 km")
ax.grid(alpha=0.3, which='both')
save(fig, '0.5.3_radial_profile.png')

# ---- 3.2.4 密度分層下的類別組成 ----
fig, ax = plt.subplots(figsize=(9, 4.8))
left = np.zeros(len(comp))
pos = np.arange(len(comp))
for cat, color in zip(CATEGORIES, CAT_COLORS):
    v = comp[cat].values
    ax.barh(pos, v, left=left, color=color, height=0.7, label=cat)
    for p, l, w in zip(pos, left, v):
        if w >= 6:
            ax.text(l + w / 2, p, f'{w:.0f}', ha='center', va='center',
                    fontsize=8, color='white')
    left += v
ax.set_yticks(pos)
ax.set_yticklabels([f'{r.label}\n({r.n_cell} cells)' for r in comp.itertuples()], fontsize=8)
ax.set_xlim(0, 100)
ax.set_xlabel('Share of POIs (%)')
ax.set_ylabel('POIs per 1 km cell (quintile)')
ax.set_title('Category composition by local density')
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
save(fig, '0.5.4_composition_by_density.png')
