import os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# fsq（Foursquare 打卡紀錄清乾淨後的唯一地點清單，類別是 FSQ 的最大類）固定參數
CSV = os.path.join(ROOT, "data", "tky_clean.csv")
CAT_COL = "category"
CATEGORIES = [
    "Dining and Drinking",
    "Retail",
    "Nightlife Spot",
    "Community and Government",
    "Travel and Transportation",
    "Business and Professional Services",
    "Landmarks and Outdoors",
    "Arts and Entertainment",
    "Health and Medicine",
    "Sports and Recreation",
]
CAT_ZH = [
    "餐飲", "零售", "夜生活", "社區/政府", "交通",
    "商業服務", "地標/戶外", "藝文娛樂", "醫療", "運動休閒",
]

HALF_WIDTH = 50.0
CELL = 15
CENTER_STEP = 100      # patch 中心的格點間距(公尺)
MIN_POI = 10            # 圓內少於這個數量的中心直接丟掉

GRID = int(HALF_WIDTH * 2 / CELL)
CRS = "EPSG:6677"       # 日本平面直角座標系第9系，涵蓋東京都，單位公尺

PALETTE = [
    "#e6194b", "#3cb44b", "#911eb4", "#4363d8", "#f58231",
    "#46f0f0", "#008080", "#f032e6", "#9a6324", "#808000",
]

N_CAT = len(CATEGORIES)
CAT_COLORS = PALETTE[:N_CAT]

PATCHES = os.path.join(ROOT, "data", "patch", "patches.npz")
FEATURES = os.path.join(ROOT, "data", "patch", "features.npz")
FEATURES_CSV = os.path.join(ROOT, "data", "patch", "features.csv")

# 記錄進 result.log 的 dataset 區塊，方便回頭比較不同次訓練用的參數
PATCH_PARAMS = {
    "csv": CSV,
    "cat_col": CAT_COL,
    "categories": CATEGORIES,
    "half_width": HALF_WIDTH,
    "cell": CELL,
    "center_step": CENTER_STEP,
    "min_poi": MIN_POI,
    "crs": CRS,
}


def result(version, name):
    path = os.path.join(ROOT, "model", version, "result", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path



def make_split(lat, lon, mode="random", seed=0, val_frac=0.15, test_frac=0.2,
               cell=0.02):
    n = len(lat)
    rng = np.random.default_rng(seed)

    if mode == "random":
        member = [np.array([i]) for i in range(n)]
    elif mode == "spatial":
        key = (np.floor(np.asarray(lat) / cell).astype(np.int64) * 1_000_000 +
               np.floor(np.asarray(lon) / cell).astype(np.int64))
        member = [np.flatnonzero(key == b) for b in np.unique(key)]

    n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
    bucket, filled = [[], [], []], [0, 0]   # filled：test、val 目前裝了幾個 patch
    for u in rng.permutation(len(member)):
        idx = member[u]
        which = 2 if filled[0] < n_test else (1 if filled[1] < n_val else 0)
        bucket[which].append(idx)
        if which >= 1:
            filled[2 - which] += len(idx)

    def cat(xs):
        a = np.concatenate(xs) if xs else np.empty(0, dtype=np.int64)
        return torch.from_numpy(np.sort(a.astype(np.int64)))

    return cat(bucket[0]), cat(bucket[1]), cat(bucket[2])


def make_kfold_split(lat, lon, mode="random", seed=0, test_frac=0.2, n_splits=5,
                     cell=0.02):
    """傳入：lat/lon（patch 中心座標，用於 spatial 分桶）、test_frac（獨立 test 集比例）、
    n_splits（k-fold 折數）。回傳：test_idx（獨立於所有 fold 的 test 集）、
    folds（長度 n_splits 的 [(train_idx, val_idx), ...] 列表，val 輪流取 test_idx 以外資料的 1/n_splits）。"""
    rng = np.random.default_rng(seed)

    if mode == "random":
        member = [np.array([i]) for i in range(len(lat))]
    elif mode == "spatial":
        key = (np.floor(np.asarray(lat) / cell).astype(np.int64) * 1_000_000 +
               np.floor(np.asarray(lon) / cell).astype(np.int64))
        member = [np.flatnonzero(key == b) for b in np.unique(key)]

    def cat(bucket_ids):
        xs = [member[u] for u in bucket_ids]
        a = np.concatenate(xs) if xs else np.empty(0, dtype=np.int64)
        return torch.from_numpy(np.sort(a.astype(np.int64)))

    order = rng.permutation(len(member))
    n_test = int(round(len(member) * test_frac))
    test_idx = cat(order[:n_test])

    pool = rng.permutation(order[n_test:])
    fold_buckets = np.array_split(pool, n_splits)

    folds = []
    for k in range(n_splits):
        val_idx = cat(fold_buckets[k])
        train_idx = cat(np.concatenate(
            [fold_buckets[j] for j in range(n_splits) if j != k]))
        folds.append((train_idx, val_idx))

    return test_idx, folds
