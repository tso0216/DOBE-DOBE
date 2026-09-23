"""為空間分布的觀察圖備料：密度網格、三塊放大區、離都心的徑向剖面、
以及不同密度分層下的類別組成。"""
import os
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import CATEGORIES, CRS, CSV  # noqa: E402

DENSITY_CELL = 250      # 密度圖的網格邊長(公尺)
COARSE_CELL = 1000      # 挑放大區、做密度分層用的粗網格邊長(公尺)
ZOOM_SIDE = 800         # 每塊放大區的邊長(公尺)
ZOOM_PCT = [100, 90, 50]   # 放大區取粗網格 POI 數的第幾百分位
RING = 1000             # 徑向剖面的環寬(公尺)
N_LEVEL = 5             # 密度分層數

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(CSV)
tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
x, y = tr.transform(df["lon"].values, df["lat"].values)
cats = df["category"].values

# ---- 1. 密度網格 ----
gx = np.floor(x / DENSITY_CELL).astype(np.int64)
gy = np.floor(y / DENSITY_CELL).astype(np.int64)
key, inv = np.unique(np.column_stack([gx, gy]), axis=0, return_inverse=True)
dens = np.bincount(inv)
pd.DataFrame({"gx": key[:, 0], "gy": key[:, 1], "n_poi": dens}).to_csv(
    os.path.join(out_dir, "density.csv"), index=False)

# ---- 2. 粗網格：挑放大區、定義都心、做密度分層 ----
cx = np.floor(x / COARSE_CELL).astype(np.int64)
cy = np.floor(y / COARSE_CELL).astype(np.int64)
ckey, cinv = np.unique(np.column_stack([cx, cy]), axis=0, return_inverse=True)
cnt = np.bincount(cinv)

# 都心＝POI 最多的那個 1 km 格的中心
core = (ckey[np.argmax(cnt)] + 0.5) * COARSE_CELL

order = np.argsort(cnt)
zooms = []
for pct in ZOOM_PCT:
    rank = np.clip(round(len(order) * pct / 100) - 1, 0, len(order) - 1)
    cell = ckey[order[rank]]
    center = (cell + 0.5) * COARSE_CELL
    m = ((np.abs(x - center[0]) <= ZOOM_SIDE / 2) &
         (np.abs(y - center[1]) <= ZOOM_SIDE / 2))
    sub = pd.DataFrame({"zoom_pct": pct,
                        "cx": center[0], "cy": center[1],
                        "dx": x[m] - center[0], "dy": y[m] - center[1],
                        "category": cats[m]})
    sub["dist_core_km"] = np.hypot(center[0] - core[0], center[1] - core[1]) / 1000
    zooms.append(sub)
    print(f"p{pct:>3} 放大區：1 km 格內 {cnt[order[rank]]:>4} 個 POI，"
          f"{ZOOM_SIDE} m 視窗內 {m.sum():>4} 個，離都心 {sub['dist_core_km'].iloc[0]:.1f} km")
pd.concat(zooms).to_csv(os.path.join(out_dir, "zoom.csv"), index=False)

# ---- 3. 徑向剖面：離都心多遠、密度掉多快 ----
r = np.hypot(x - core[0], y - core[1])
ring = (r // RING).astype(np.int64)
n_ring = int(ring.max()) + 1
poi_ring = np.bincount(ring, minlength=n_ring)
r_in = np.arange(n_ring) * RING
area = np.pi * ((r_in + RING) ** 2 - r_in ** 2) / 1e6      # km^2
pd.DataFrame({"r_km": (r_in + RING / 2) / 1000, "n_poi": poi_ring,
              "density": poi_ring / area}).to_csv(
    os.path.join(out_dir, "radial.csv"), index=False)

# ---- 4. 密度分層下的類別組成 ----
edges = np.unique(np.percentile(cnt, np.linspace(0, 100, N_LEVEL + 1)))
level = np.clip(np.digitize(cnt, edges[1:-1]), 0, len(edges) - 2)
poi_level = level[cinv]

rows = []
for lv in range(len(edges) - 1):
    m = poi_level == lv
    share = pd.Series(cats[m]).value_counts(normalize=True).reindex(CATEGORIES).fillna(0) * 100
    rows.append(dict(level=lv,
                     label=f"{edges[lv]:.0f}–{edges[lv + 1]:.0f}",
                     n_cell=int((level == lv).sum()), n_poi=int(m.sum()),
                     **share.to_dict()))
pd.DataFrame(rows).to_csv(os.path.join(out_dir, "composition.csv"), index=False)

np.savetxt(os.path.join(out_dir, "core.csv"), core[None, :], delimiter=",",
           header="core_x,core_y", comments="")
print(f"密度網格 {len(key)} 格（{DENSITY_CELL} m），粗網格 {len(ckey)} 格（{COARSE_CELL} m）")
print(f"都心格中心 (x, y) = ({core[0]:.0f}, {core[1]:.0f})，最高密度 {cnt.max()} POI/km²")
