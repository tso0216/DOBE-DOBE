import importlib
import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from common.dataset import (CAT_COLORS, CAT_ZH, CELL, GRID, N_CAT,  # noqa: E402
                            PATCHES, HALF_WIDTH, make_kfold_split, result)




MODEL_VERSION = "v3"
MODELS = {
    "ae": dict(latent_dim=2, dir="v2_deep_ae", pt="ae_fsce.pt"),
    "vae": dict(latent_dim=2, dir="v2_deep_vae", pt="vae.pt", vae=True),
    "ddae": dict(latent_dim=2, dir="v2_ddae_base", pt="ae.pt"),
    "v3": dict(latent_dim=2, dir="v3_ddae_tfidf",
               ckpt=os.path.join(ROOT, "experiment", "model",
                                 "ddae_fsce_tfidf", "fold3_mae.pt")),
}
CKPT = ''

cfg = MODELS[MODEL_VERSION]
TEST_ONLY = True     # 只用該模型訓練時切出來的 test set（獨立於所有 fold）
PATCH_IDX = None     # None = 從 robust 距離最小（最典型）的 patch 起手；或填 patch 編號
BATCH_N = 50         # 「+N 隨機」一次灑幾顆
OUTLIER_PCT = 99.5   # 全體 robust 距離的離群門檻百分位
ZOOM_PCT = (1, 99)   # latent 圖初始視野
SEED = 0
DOT = 3.0
K_SCORE = 8          # S 分數用的 kNN 鄰居數（跟 experiment/1.3.2_affinity_latent 一致）

ModelClass = importlib.import_module(f"model.{cfg['dir']}.model").AE

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False


def robust_distance(z, ref):
    """用全體 latent 的 中位數 / MAD 標準化後，算 z 到 latent 中心的距離。"""
    med = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - med), axis=0) * 1.4826
    return np.linalg.norm((z - med) / mad, axis=-1)



def sample_disk(rng, k):
    """在半徑 HALF_WIDTH 的圓內均勻灑 k 個點。"""
    r = HALF_WIDTH * np.sqrt(rng.random(k))
    t = rng.random(k) * 2 * np.pi
    return r * np.cos(t), r * np.sin(t)


def patch_points(p, i):
    """回傳第 i 個 patch 的 (dx, dy, cat)。"""
    s, e = p["offsets"][i], p["offsets"][i + 1]
    return p["dx"][s:e], p["dy"][s:e], p["cat"][s:e]


def fit_to2d(z_all, latent_dim):
    """latent 超過 2 維時，用全體的 PCA 前 2 主成分作圖（距離仍用完整維度）。"""
    if latent_dim == 2:
        return lambda z: z
    mean = z_all.mean(axis=0)
    _, _, vt = np.linalg.svd(z_all - mean, full_matrices=False)
    return lambda z: (z - mean) @ vt[:2].T


def knn_neighbors(z_ref, k):
    """z_ref：(n, d) 全部 patch 未擾動的 latent。k：鄰居數。
    回傳 (n, k)：每個 patch 在 z_ref 內最近的 k 個鄰居索引（不含自己）。"""
    dist, idx = cKDTree(z_ref).query(z_ref, k=k + 1)
    keep = idx != np.arange(len(z_ref))[:, None]
    return np.array([row[m][:k] for row, m in zip(idx, keep)])


def neighbor_dist(z_eval, nb):
    """z_eval：(d,) 被評分的 latent。nb：(k,) 該 patch 的鄰居索引（在 z_all 內）。
    回傳 float：到那 k 個鄰居的平均歐氏距離。"""
    return float(np.linalg.norm(z_all[nb] - z_eval[None, :], axis=1).mean())


def subset_patches(p, idx):
    """傳入：patches npz、要保留的 patch 索引。回傳：只含這些 patch 的 dict（dx/dy/cat/offsets/n_poi）。"""
    offsets, dx, dy, cat, n_poi = (p["offsets"], p["dx"], p["dy"], p["cat"], p["n_poi"])
    dx_l, dy_l, cat_l = [], [], []
    for i in idx:
        s, e = offsets[i], offsets[i + 1]
        dx_l.append(dx[s:e])
        dy_l.append(dy[s:e])
        cat_l.append(cat[s:e])
    n_poi_sub = n_poi[idx]
    offsets_sub = np.concatenate([[0], np.cumsum(n_poi_sub)])
    return dict(dx=np.concatenate(dx_l), dy=np.concatenate(dy_l),
               cat=np.concatenate(cat_l), offsets=offsets_sub, n_poi=n_poi_sub)


def encode_all(model, p, vae=False, batch=512):
    """傳入：model、patches npz（含 cat/offsets）。回傳：z_all，全體 patch 的 latent。

    現場用同一個 model 重算，不吃預存的 latents.npz——避免拿到跟目前 checkpoint
    對不上的舊快取（不同次訓練的 latent space 座標系不會一樣）。
    """
    offsets, cat = p["offsets"], p["cat"]
    n = len(offsets) - 1
    zs = []
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            counts = np.stack([
                np.bincount(cat[offsets[i]:offsets[i + 1]], minlength=N_CAT)
                for i in range(s, e)
            ]).astype(np.float32)
            out = model(torch.from_numpy(counts))
            z = out[0] if not vae else out[2]
            zs.append(z.numpy())
    return np.concatenate(zs, axis=0)


# ---- 載入資料與模型 ----

cfg = MODELS[MODEL_VERSION]
p = np.load(PATCHES)
if TEST_ONLY:
    split_cfg = importlib.import_module(f"model.{cfg['dir']}.cfg")
    test_idx, _ = make_kfold_split(p["center_lat"], p["center_lon"], seed=split_cfg.SEED,
                                   test_frac=split_cfg.TEST_FRAC, n_splits=split_cfg.N_SPLITS)
    p = subset_patches(p, test_idx.numpy())
n_poi = p["n_poi"]

model = ModelClass(cfg["latent_dim"])
ckpt_path = CKPT if CKPT else cfg.get("ckpt") or result(cfg["dir"], cfg["pt"])
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model.eval()

z_all = encode_all(model, p, vae=cfg.get("vae", False))

dist_all = robust_distance(z_all, z_all)
thr = np.percentile(dist_all, OUTLIER_PCT)

# S 分數：每個 patch 到自己（未擾動時）最近 K_SCORE 個鄰居的平均 latent 距離，
# 鄰居的身分與位置全程固定，只有被評分那個點的 latent 會隨著加 POI 而動。
# 基準值不在這裡預先算：換 patch 時用 traj[0] 走同一條計算路徑，
# 這樣「沒加任何 POI」時 Δ 會嚴格等於 0，不會被浮點誤差染成紅／綠。
nb_all = knn_neighbors(z_all, K_SCORE)
to2d = fit_to2d(z_all, cfg["latent_dim"])
z2_all = to2d(z_all)

# 點擊找最近 patch：兩軸各自除以視野跨度，讓「最近」跟眼睛看到的一致
lo = np.percentile(z2_all, ZOOM_PCT[0], axis=0)
hi = np.percentile(z2_all, ZOOM_PCT[1], axis=0)
scale = np.where(hi - lo > 0, hi - lo, 1.0)
tree = cKDTree(z2_all / scale)

# ---- UI 佈局 ----

fig = plt.figure(figsize=(14, 7.6))
ax_lat = fig.add_axes([0.06, 0.07, 0.40, 0.72])
ax_map = fig.add_axes([0.53, 0.07, 0.29, 0.72])
ax_radio = fig.add_axes([0.86, 0.32, 0.12, 0.47])
ax_batch = fig.add_axes([0.86, 0.23, 0.12, 0.05])
ax_undo = fig.add_axes([0.86, 0.16, 0.12, 0.05])
ax_reset = fig.add_axes([0.86, 0.09, 0.12, 0.05])

# 分數面板：整個視窗最上方置中，底色跟著 Δ 的方向變
SCORE_STYLE = {                       # tag: (文字色, 底色, 框線色)
    "flat": ("#374151", "#f3f4f6", "#9ca3af"),
    "far": ("#c0392b", "#fdecea", "#c0392b"),
    "near": ("#2e7d32", "#e8f5e9", "#2e7d32"),
}
score_status = fig.text(0.50, 0.985, "", fontsize=17, ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.45", fc="#f3f4f6",
                                  ec="#9ca3af", lw=1.4))
score_detail = fig.text(0.50, 0.895, "", fontsize=9.5, ha="center", va="top",
                        color="#4b5563")
status = fig.text(0.06, 0.845, "", fontsize=10)

axis2 = "z" if cfg["latent_dim"] == 2 else "PC"
ax_lat.scatter(z2_all[:, 0], z2_all[:, 1], s=DOT * 3, c="#6b7280",
               linewidths=0, alpha=0.75, rasterized=True)
pad = (hi - lo) * 0.05
ax_lat.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
ax_lat.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
ax_lat.set_title(f"latent space（{MODEL_VERSION}，點一下換 patch）", fontsize=10)
ax_lat.set_xlabel(f"{axis2}1", fontsize=8)
ax_lat.set_ylabel(f"{axis2}2", fontsize=8)
ax_lat.tick_params(labelsize=7)
ax_lat.grid(alpha=0.15, linewidth=0.5)

traj_line, = ax_lat.plot([], [], c="#c0392b", lw=1.5, zorder=3)
star = ax_lat.scatter([], [], s=130, marker="*", c="#1a1a1a", zorder=4)
cur = ax_lat.scatter([], [], s=60, c="#c0392b", edgecolors="k",
                     linewidths=0.6, zorder=5)

state = dict(idx=-1, base=None, cat=0,
             added=[],    # list of (dx_arr, dy_arr, cat_id)，一格 = 一個 undo 批次
             traj=[],     # 完整維度 latent；len(traj) == len(added) + 1
             artists=[],  # 與 added 平行的 ax_map scatter handle
             score_base=0.0)  # 這個 patch 未加 POI 時的 S 分數
rng = np.random.default_rng(SEED)


def encode_current():
    dx0, dy0, cat0 = state["base"]
    dx = np.concatenate([dx0] + [b[0] for b in state["added"]])
    dy = np.concatenate([dy0] + [b[1] for b in state["added"]])
    cat = np.concatenate([cat0] + [np.full(len(b[0]), b[2], dtype=cat0.dtype)
                                   for b in state["added"]])
    with torch.no_grad():
        counts = np.bincount(cat, minlength=N_CAT)
        x = torch.from_numpy(counts).unsqueeze(0).float()
        out = model(x)
        z = out[0] if not cfg.get("vae") else out[2]
    return z[0].numpy()


def update():
    t2 = to2d(np.array(state["traj"]))
    traj_line.set_data(t2[:, 0], t2[:, 1])
    star.set_offsets(t2[:1])
    cur.set_offsets(t2[-1:])
    dist = robust_distance(state["traj"][-1], z_all)
    out = dist > thr
    n_added = sum(len(b[0]) for b in state["added"])
    status.set_text(f"patch {state['idx']}｜原始 POI {n_poi[state['idx']]}｜"
                    f"已加 {n_added} 顆｜robust 距離 {dist:.2f} / "
                    f"門檻 {thr:.2f}｜" + ("★ 離群" if out else "正常"))
    status.set_color("#c0392b" if out else "#1a1a1a")

    score_now = neighbor_dist(state["traj"][-1], nb_all[state["idx"]])
    score_base = state["score_base"]
    delta = score_now - score_base
    # 沒加東西（或加完又 undo/reset 回原狀）一律當作零，別讓浮點殘差染色
    tol = max(abs(score_base), 1.0) * 1e-9
    if not state["added"] or abs(delta) <= tol:
        delta, tag, word = 0.0, "flat", "未改動"
    elif delta > 0:
        tag, word = "far", "遠離鄰居"
    else:
        tag, word = "near", "靠近鄰居"
    fg, bg, ec = SCORE_STYLE[tag]
    score_status.set_text(f"S 分數 {score_now:.3f}    Δ {delta:+.3f}  {word}")
    score_status.set_color(fg)
    score_status.get_bbox_patch().set(facecolor=bg, edgecolor=ec)
    score_detail.set_text(f"k={K_SCORE} 近鄰平均 latent 距離｜"
                          f"無 POI 基準 {score_base:.3f}｜已加 {n_added} 顆")
    fig.canvas.draw_idle()


def draw_base_map():
    ax_map.clear()
    dx0, dy0, cat0 = state["base"]
    for k in range(N_CAT):
        m = cat0 == k
        if m.any():
            ax_map.scatter(dx0[m], dy0[m], s=16, c=CAT_COLORS[k],
                           linewidths=0, alpha=0.85)
    ax_map.set_xlim(-HALF_WIDTH * 1.1, HALF_WIDTH * 1.1)
    ax_map.set_ylim(-HALF_WIDTH * 1.1, HALF_WIDTH * 1.1)
    ax_map.set_aspect("equal")
    ax_map.set_title(f"patch {state['idx']} 的 POI（點一下加一顆）", fontsize=10)
    ax_map.set_xlabel("東西向位移 (m)", fontsize=8)
    ax_map.set_ylabel("南北向位移 (m)", fontsize=8)
    ax_map.tick_params(labelsize=7)
    ax_map.grid(alpha=0.15, linewidth=0.5)


def set_patch(i):
    state["idx"] = int(i)
    state["base"] = patch_points(p, state["idx"])
    state["added"], state["artists"] = [], []
    draw_base_map()
    state["traj"] = [encode_current()]
    state["score_base"] = neighbor_dist(state["traj"][0], nb_all[state["idx"]])
    update()


def add_batch(dx, dy):
    cat_id = state["cat"]
    state["added"].append((dx, dy, cat_id))
    state["traj"].append(encode_current())
    art = ax_map.scatter(dx, dy, s=34, marker="x", c=CAT_COLORS[cat_id],
                         linewidths=1.4, zorder=3)
    state["artists"].append(art)
    update()


def on_click(event):
    if getattr(fig.canvas.toolbar, "mode", "") != "":
        return  # pan/zoom 模式中，拖曳不觸發
    if event.inaxes is ax_lat:
        _, i = tree.query(np.array([event.xdata, event.ydata]) / scale)
        set_patch(i)
    elif event.inaxes is ax_map:
        add_batch(np.array([event.xdata], dtype=np.float32),
                  np.array([event.ydata], dtype=np.float32))


def on_radio(label):
    state["cat"] = CAT_ZH.index(label)


def on_batch(event):
    bx, by = sample_disk(rng, BATCH_N)
    add_batch(bx.astype(np.float32), by.astype(np.float32))


def on_undo(event):
    if not state["added"]:
        return
    state["added"].pop()
    state["traj"].pop()
    state["artists"].pop().remove()
    update()


def on_reset(event):
    for a in state["artists"]:
        a.remove()
    state["added"], state["artists"] = [], []
    state["traj"] = state["traj"][:1]
    update()


ax_radio.set_title("要加的類別", fontsize=9)
radio = RadioButtons(ax_radio, CAT_ZH)
for i, t in enumerate(radio.labels):
    t.set_color(CAT_COLORS[i])
    t.set_fontsize(9)
btn_batch = Button(ax_batch, f"+{BATCH_N} 隨機")
btn_undo = Button(ax_undo, "undo")
btn_reset = Button(ax_reset, "reset")
radio.on_clicked(on_radio)
btn_batch.on_clicked(on_batch)
btn_undo.on_clicked(on_undo)
btn_reset.on_clicked(on_reset)
fig.canvas.mpl_connect("button_press_event", on_click)

set_patch(int(np.argmin(dist_all)) if PATCH_IDX is None else PATCH_IDX)

print(f"model：{MODEL_VERSION}｜checkpoint：{ckpt_path}")
print(f"patch {state['idx']}：POI {n_poi[state['idx']]}，"
      f"robust 距離 {dist_all[state['idx']]:.2f}，離群門檻 {thr:.2f}")

plt.show()
