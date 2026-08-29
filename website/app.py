import os

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

from model_def import AE, N_CAT

BASE = os.path.dirname(os.path.abspath(__file__))
PATCHES = os.path.join(BASE, "data", "patches.npz")
CKPT = os.path.join(BASE, "weights", "fold3_mae.pt")

LATENT_DIM = 2
HALF_WIDTH = 50.0
CENTER_STEP = 100.0  # patch 中心格點間距（公尺），與 common/dataset.py 的 CENTER_STEP 一致
SEED = 0            # 與 v3 訓練切 split 用的種子一致（model/v3_ddae_tfidf/cfg.py）
TEST_FRAC = 0.2
K_SCORE = 8         # S 分數用的 kNN 鄰居數
OUTLIER_PCT = 99.5  # robust 距離離群門檻百分位
ZOOM_PCT = (1, 99)  # latent 圖初始視野
BATCH_N = 50        # +N 隨機一次灑幾顆

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
CAT_COLORS = [
    "#e6194b", "#3cb44b", "#911eb4", "#4363d8", "#f58231",
    "#46f0f0", "#008080", "#f032e6", "#9a6324", "#808000",
]


def make_test_idx(n, seed, test_frac):
    """傳入：patch 總數、切分種子、test 比例。回傳：test set 的全域 patch 索引（升冪，numpy int64）。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(round(n * test_frac))
    return np.sort(order[:n_test])


def counts_per_patch(p):
    """傳入：patches npz。回傳：(n, N_CAT) 每個 patch 的類別計數矩陣（float32）。"""
    offsets, cat = p["offsets"], p["cat"]
    n = len(offsets) - 1
    return np.stack([
        np.bincount(cat[offsets[i]:offsets[i + 1]], minlength=N_CAT)
        for i in range(n)
    ]).astype(np.float32)


def encode(counts):
    """傳入：(k, N_CAT) 類別計數。回傳：(k, LATENT_DIM) latent（numpy float32）。"""
    with torch.no_grad():
        z, _ = model(torch.from_numpy(np.ascontiguousarray(counts)))
    return z.numpy()


def robust_distance(z):
    """傳入：(..., LATENT_DIM) latent。回傳：以 test 全體中位數/MAD 標準化後到中心的距離。"""
    return np.linalg.norm((z - MED) / MAD, axis=-1)


def neighbor_score(z, nb):
    """傳入：(LATENT_DIM,) latent、(K_SCORE,) 鄰居在 z_test 內的索引。回傳：到鄰居的平均歐氏距離（float）。"""
    return float(np.linalg.norm(z_test[nb] - z[None, :], axis=1).mean())


# ---- 啟動時載入資料與模型 ----

p = np.load(PATCHES)
offsets, n_poi_all = p["offsets"], p["n_poi"]
n_all = len(n_poi_all)

test_idx = make_test_idx(n_all, SEED, TEST_FRAC)
is_test = np.zeros(n_all, dtype=bool)
is_test[test_idx] = True
test_pos = np.full(n_all, -1, dtype=np.int64)
test_pos[test_idx] = np.arange(len(test_idx))

counts_all = counts_per_patch(p)

model = AE(LATENT_DIM)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

z_test = encode(counts_all[test_idx])
MED = np.median(z_test, axis=0)
MAD = np.median(np.abs(z_test - MED), axis=0) * 1.4826

dist_all = robust_distance(z_test)
THR = float(np.percentile(dist_all, OUTLIER_PCT))

# S 分數的鄰居：每個 test patch 未擾動時在 z_test 內最近 K_SCORE 個（不含自己），全程固定
d2 = ((z_test[:, None, :] - z_test[None, :, :]) ** 2).sum(-1)
np.fill_diagonal(d2, np.inf)
nb_all = np.argsort(d2, axis=1)[:, :K_SCORE]

lo = np.percentile(z_test, ZOOM_PCT[0], axis=0)
hi = np.percentile(z_test, ZOOM_PCT[1], axis=0)

# 全部 POI 的地理絕對座標（patch 中心 + 圓內位移），給地理圖畫點用
poi_x = np.repeat(p["center_x"], n_poi_all) + p["dx"]
poi_y = np.repeat(p["center_y"], n_poi_all) + p["dy"]

app = Flask(__name__)


@app.route("/")
def index():
    """回傳：主頁面 HTML。"""
    return render_template("index.html")


@app.route("/api/init")
def api_init():
    """回傳：初始化 payload（地理點、latent 座標、robust 距離、門檻、類別、常數）。"""
    return jsonify(dict(
        geo=dict(
            x=p["center_x"].astype(float).round(1).tolist(),
            y=p["center_y"].astype(float).round(1).tolist(),
            is_test=is_test.astype(int).tolist(),
            test_pos=test_pos.tolist(),
        ),
        poi=dict(
            x=poi_x.astype(float).round(1).tolist(),
            y=poi_y.astype(float).round(1).tolist(),
            cat=p["cat"].astype(int).tolist(),
        ),
        latent=z_test.astype(float).tolist(),
        robust=dist_all.astype(float).tolist(),
        thr=THR,
        latent_lo=lo.astype(float).tolist(),
        latent_hi=hi.astype(float).tolist(),
        global_ids=test_idx.tolist(),
        n_poi=n_poi_all[test_idx].tolist(),
        categories=CATEGORIES,
        colors=CAT_COLORS,
        half_width=HALF_WIDTH,
        center_step=CENTER_STEP,
        k_score=K_SCORE,
        batch_n=BATCH_N,
        init_patch=int(np.argmin(dist_all)),
    ))


@app.route("/api/patch/<int:i>")
def api_patch(i):
    """傳入：test-local patch 索引。回傳：該 patch 的 base POI、初始 latent、S 分數基準、robust 距離。"""
    if not 0 <= i < len(test_idx):
        return jsonify(error="patch index out of range"), 400
    g = int(test_idx[i])
    s, e = int(offsets[g]), int(offsets[g + 1])
    z0 = encode(counts_all[g:g + 1])[0]
    return jsonify(dict(
        idx=i,
        global_id=g,
        n_poi=int(n_poi_all[g]),
        base=dict(
            dx=p["dx"][s:e].astype(float).round(2).tolist(),
            dy=p["dy"][s:e].astype(float).round(2).tolist(),
            cat=p["cat"][s:e].astype(int).tolist(),
        ),
        z=z0.astype(float).tolist(),
        robust=float(robust_distance(z0)),
        score=neighbor_score(z0, nb_all[i]),
    ))


@app.route("/api/encode", methods=["POST"])
def api_encode():
    """傳入（JSON）：idx（test-local patch 索引）、added（長度 N_CAT 的新增類別計數）。
    回傳：加料後的 latent、robust 距離、S 分數。"""
    d = request.get_json(force=True)
    i = int(d["idx"])
    added = np.asarray(d["added"], dtype=np.float32)
    if not 0 <= i < len(test_idx) or added.shape != (N_CAT,) or (added < 0).any():
        return jsonify(error="bad request"), 400
    g = int(test_idx[i])
    z = encode((counts_all[g] + added)[None, :])[0]
    return jsonify(dict(
        z=z.astype(float).tolist(),
        robust=float(robust_distance(z)),
        score=neighbor_score(z, nb_all[i]),
    ))


if __name__ == "__main__":
    print(f"checkpoint：{os.path.relpath(CKPT, BASE)}｜"
          f"patch 總數 {n_all}（test {len(test_idx)}）｜離群門檻 {THR:.2f}")
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5001)), debug=False)
