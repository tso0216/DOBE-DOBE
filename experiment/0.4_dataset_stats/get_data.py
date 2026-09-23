"""從 raw 打卡紀錄、清理後 POI 清單、patch 檔算出資料集的各項統計量。"""
import collections
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import (CATEGORIES, CENTER_STEP, CRS, CSV,  # noqa: E402
                            HALF_WIDTH, MIN_POI, N_CAT, PATCHES)

RAW = os.path.join(os.path.dirname(__file__), "../../data/tky_raw.txt")

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)


def scan_raw():
    """回傳：(n_row, 每個 user 的打卡數 Counter, 每個 poi 的打卡數 Counter,
    最早與最晚打卡時間)。"""
    users, pois = collections.Counter(), collections.Counter()
    n_row, t_min, t_max = 0, None, None
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 8:
                continue
            n_row += 1
            users[p[0]] += 1
            pois[p[1]] += 1
            t = dt.datetime.strptime(p[7], "%a %b %d %H:%M:%S %z %Y")
            t_min = t if t_min is None or t < t_min else t_min
            t_max = t if t_max is None or t > t_max else t_max
    return n_row, users, pois, t_min, t_max


def describe(name, a):
    """name：統計量名稱。a：一維數列。回傳：一列 dict，含常見敘述統計。"""
    a = np.asarray(a, dtype=np.float64)
    return dict(quantity=name, n=len(a), mean=a.mean(), std=a.std(ddof=1),
                min=a.min(), p10=np.percentile(a, 10), p25=np.percentile(a, 25),
                median=np.median(a), p75=np.percentile(a, 75),
                p90=np.percentile(a, 90), p99=np.percentile(a, 99), max=a.max(),
                sum=a.sum())


n_row, user_cnt, raw_poi_cnt, t_min, t_max = scan_raw()
df = pd.read_csv(CSV)

# ---- 空間範圍（投影成公尺後量 bbox） ----
tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
x, y = tr.transform(df["lon"].values, df["lat"].values)

# ---- 格點：POI 佔用過的 100 m 格點，以及通過 MIN_POI 門檻的那些 ----
cell_id = np.floor(np.column_stack([x, y]) / CENTER_STEP).astype(np.int64)
cells, cell_poi = np.unique(cell_id, axis=0, return_counts=True)
kept = cell_poi >= MIN_POI

# ---- patch：實際餵進模型的那 N 個鄰域 ----
d = np.load(PATCHES)
offsets, cat, n_poi = d["offsets"], d["cat"], d["n_poi"]
n_patch = len(n_poi)

counts = np.zeros((n_patch, N_CAT), dtype=np.float64)
for i in range(n_patch):
    counts[i] = np.bincount(cat[offsets[i]:offsets[i + 1]], minlength=N_CAT)

n_nonzero = (counts > 0).sum(axis=1)                  # 每個 patch 出現幾種類別
p = counts / counts.sum(axis=1, keepdims=True)
plogp = np.zeros_like(p)
np.log(p, out=plogp, where=p > 0)
entropy = -(p * plogp).sum(axis=1)                    # 機能混合度
dominance = p.max(axis=1)                             # 最大類別佔比

# ---- 1. 純量摘要 ----
span_days = (t_max - t_min).days
summary = [
    ("raw_checkins", n_row),
    ("raw_users", len(user_cnt)),
    ("raw_unique_poi", len(raw_poi_cnt)),
    ("raw_span_days", span_days),
    ("raw_first_checkin", t_min.strftime("%Y-%m-%d")),
    ("raw_last_checkin", t_max.strftime("%Y-%m-%d")),
    ("clean_poi", len(df)),
    ("dropped_poi", len(raw_poi_cnt) - len(df)),
    ("leaf_category", df["category_name"].nunique()),
    ("top_category", df["category"].nunique()),
    ("lat_min", df["lat"].min()),
    ("lat_max", df["lat"].max()),
    ("lon_min", df["lon"].min()),
    ("lon_max", df["lon"].max()),
    ("bbox_width_km", (x.max() - x.min()) / 1000),
    ("bbox_height_km", (y.max() - y.min()) / 1000),
    ("poi_per_km2", len(df) / ((x.max() - x.min()) * (y.max() - y.min()) / 1e6)),
    ("cell_size_m", CENTER_STEP),
    ("patch_window_m", HALF_WIDTH * 2),
    ("min_poi", MIN_POI),
    ("occupied_cells", len(cells)),
    ("kept_patches", int(kept.sum())),
    ("kept_cell_ratio", kept.sum() / len(cells)),
    ("poi_in_patches", int(cell_poi[kept].sum())),
    ("poi_coverage", cell_poi[kept].sum() / len(df)),
    ("patch_in_npz", n_patch),
]
pd.DataFrame(summary, columns=["metric", "value"]).to_csv(
    os.path.join(out_dir, "summary.csv"), index=False)

# ---- 2. 敘述統計表 ----
pd.DataFrame([
    describe("checkins_per_user", list(user_cnt.values())),
    describe("checkins_per_poi", df["checkin_count"].values),
    describe("poi_per_occupied_cell", cell_poi),
    describe("poi_per_patch", n_poi),
    describe("categories_per_patch", n_nonzero),
    describe("entropy_per_patch", entropy),
    describe("dominance_per_patch", dominance),
]).to_csv(os.path.join(out_dir, "describe.csv"), index=False)

# ---- 3. 類別層級統計 ----
g = df.groupby("category")
cat_tbl = pd.DataFrame({
    "n_poi": g.size(),
    "n_leaf": g["category_name"].nunique(),
    "checkin_sum": g["checkin_count"].sum(),
    "checkin_mean": g["checkin_count"].mean(),
}).reindex(CATEGORIES)
cat_tbl["poi_ratio"] = cat_tbl["n_poi"] / cat_tbl["n_poi"].sum() * 100
cat_tbl["checkin_ratio"] = cat_tbl["checkin_sum"] / cat_tbl["checkin_sum"].sum() * 100
cat_tbl["patch_presence"] = (counts > 0).mean(axis=0) * 100      # 出現在幾 % 的 patch
cat_tbl["mean_per_patch"] = counts.mean(axis=0)
cat_tbl["std_per_patch"] = counts.std(axis=0, ddof=1)
cat_tbl.to_csv(os.path.join(out_dir, "category.csv"), index_label="category")

# ---- 4. 給 draw.py 畫分布用的原始數列 ----
pd.DataFrame({"checkin_count": df["checkin_count"].values}).to_csv(
    os.path.join(out_dir, "checkin.csv"), index=False)
pd.DataFrame({"poi_per_cell": cell_poi}).to_csv(
    os.path.join(out_dir, "cell.csv"), index=False)
pd.DataFrame({"n_poi": n_poi, "n_cat": n_nonzero, "entropy": entropy,
              "dominance": dominance}).to_csv(
    os.path.join(out_dir, "patch.csv"), index=False)

print(f"raw {n_row} 筆打卡 / {len(user_cnt)} 使用者 / {len(raw_poi_cnt)} 個 POI"
      f"（{t_min:%Y-%m-%d} ~ {t_max:%Y-%m-%d}，{span_days} 天）")
print(f"清理後 {len(df)} 個 POI，{df['category_name'].nunique()} 個細類 -> {N_CAT} 個大類")
print(f"佔用格點 {len(cells)} -> 保留 patch {kept.sum()}"
      f"（涵蓋 {cell_poi[kept].sum()} 個 POI，{cell_poi[kept].sum() / len(df):.1%}）")
print(f"每 patch POI 數 mean {n_poi.mean():.2f} std {n_poi.std(ddof=1):.2f} "
      f"median {np.median(n_poi):.0f} max {n_poi.max()}")
