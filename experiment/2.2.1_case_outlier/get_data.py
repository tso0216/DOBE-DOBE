"""挑一個 outlier patch（與 latent 最近 k 鄰居的平均距離最大），窮舉所有「加入 1~B_MAX 個 POI」的組合，記錄每個預算下分數最低的組合、latent 位置與分數"""
import csv
import itertools
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import CATEGORIES, N_CAT, PATCHES, make_split  # noqa: E402
from model.v3_ddae_tfidf.dataset import Patches  # noqa: E402

CKPT = os.path.join(HERE, "..", "model", "ddae_fsce_tfidf", "fold3_mae.pt")
PICK = "outlier"              # outlier：挑分數最大的；common：挑分數最小的
OBJ = "min"                   # min：把分數壓到最低；max：把分數推到最高
K = 8                         # 取 latent 上最近的 k 個 patch 當鄰居
B_MAX = 10                    # 加入的 POI 數上限，窮舉 1~B_MAX 個的所有組合
BATCH = 16384                 # 編碼候選時的批次大小
SEED = 0


class AE(nn.Module):
    def __init__(self, latent_dim, hidden):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_CAT, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, N_CAT),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def load_model(path):
    """path：checkpoint 檔案路徑。回傳 AE，已載入權重並設為 eval。"""
    sd = torch.load(path, map_location="cpu")
    hidden = sd["encoder.0.weight"].shape[0]
    latent_dim = sd["decoder.0.weight"].shape[1]
    model = AE(latent_dim, hidden)
    model.load_state_dict(sd)
    model.eval()
    return model


def encode(model, x):
    """model：AE。x：(B, N_CAT) count tensor。回傳 (B, latent_dim) 的 numpy latent。"""
    with torch.no_grad():
        out = [model.encoder(x[i:i + BATCH]) for i in range(0, len(x), BATCH)]
    return torch.cat(out).numpy().astype(np.float64)


def loo_score(z_all, k):
    """z_all：(N, d) 全體 patch 的 latent。k：鄰居數。回傳 (N,) 每個 patch 到自己以外
    最近 k 個 patch 的平均距離（leave-one-out）。"""
    dist, idx = cKDTree(z_all).query(z_all, k=k + 1)
    mask = idx != np.arange(len(z_all))[:, None]    # 有重複座標時 self 未必排在第一欄
    nb_dist = np.array([row[m][:k] for row, m in zip(dist, mask)])
    return nb_dist.mean(axis=1)


def knn_score(tree, z, k):
    """tree：建在參考點集上的 cKDTree。z：(B, d) 要評分的座標。k：鄰居數。
    回傳 (B,) 每個座標到參考點集中最近 k 個點的平均距離。"""
    dist, _ = tree.query(z, k=k)
    return dist.mean(axis=1)


def all_additions(n_cat, budget):
    """n_cat：類別數。budget：加入的 POI 數上限。回傳 (M, n_cat) 的增量矩陣與 (M,) 的
    加入總數，涵蓋所有總和為 1~budget 的加法組合（重複組合，不含全 0）。"""
    rows, sizes = [], []
    for b in range(1, budget + 1):
        for combo in itertools.combinations_with_replacement(range(n_cat), b):
            rows.append(np.bincount(combo, minlength=n_cat))
            sizes.append(b)
    return np.array(rows, dtype=np.float32), np.array(sizes)


def exhaustive_path(model, x0, tree, k, budget, obj):
    """model：AE。x0：(N_CAT,) 目標 patch 的計數向量。tree：建在其他 patch latent 上的
    cKDTree。k：鄰居數。budget：加入的 POI 數上限。obj：min 壓低分數 / max 推高分數。
    回傳 [(計數向量 numpy, latent numpy, 分數), ...]，第 0 筆是原始狀態，第 b 筆是窮舉所有
    「剛好加 b 個 POI」的組合後最符合目標的那一組。"""
    z0 = encode(model, x0[None, :])
    path = [(x0.numpy().copy(), z0[0], float(knn_score(tree, z0, k)[0]))]

    deltas, sizes = all_additions(N_CAT, budget)
    cand = torch.from_numpy(deltas) + x0
    z = encode(model, cand)
    s = knn_score(tree, z, k)

    pick = np.argmin if obj == "min" else np.argmax
    for b in range(1, budget + 1):
        where = np.flatnonzero(sizes == b)
        i = int(where[pick(s[where])])
        path.append((cand[i].numpy().copy(), z[i], float(s[i])))
    return path


def main():
    data = Patches(PATCHES)
    _, _, test_idx = make_split(data.lat, data.lon, seed=SEED)
    eval_idx = test_idx.numpy()
    x_test = data.agg(test_idx)

    model = load_model(CKPT)
    z_test = encode(model, x_test)
    score = loo_score(z_test, K)

    pos = int(np.argmax(score) if PICK == "outlier" else np.argmin(score))
    target = eval_idx[pos]

    role = np.full(len(eval_idx), "other", dtype=object)
    role[pos] = "target"

    bg_path = os.path.join(HERE, "background.csv")
    with open(bg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch_id", "z1", "z2", "score", "n_poi", "role"])
        for j, i in enumerate(eval_idx):
            w.writerow([i, z_test[j, 0], z_test[j, 1], score[j],
                        data.n_poi[i], role[j]])

    # 參考點集扣掉目標自己，否則位移後它的原始位置會被當成鄰居把分數壓低
    keep = np.arange(len(eval_idx)) != pos
    tree = cKDTree(z_test[keep])
    path = exhaustive_path(model, x_test[pos], tree, K, B_MAX, OBJ)

    out_path = os.path.join(HERE, "data.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "z1", "z2", "score", "gain"] + CATEGORIES)
        for step, (x, z, s) in enumerate(path):
            gain = s - path[step - 1][2] if step else 0.0
            w.writerow([step, z[0], z[1], s, gain] + list(x))

    print(f"{PICK} patch id={target}，POI 數 {data.n_poi[target]}，"
          f"原始分數 {path[0][2]:.4f}（test 集 {len(eval_idx)} 個 patch，"
          f"中位數 {np.median(score):.4f}）")
    print("原始組成：" + "、".join(
        f"{CATEGORIES[c]}×{int(v)}" for c, v in enumerate(path[0][0]) if v > 0))
    x0 = path[0][0]
    for b, (x, _, s) in enumerate(path[1:], 1):
        add = "、".join(f"{CATEGORIES[c]}×{int(v)}"
                        for c, v in enumerate(x - x0) if v > 0)
        print(f"  加 {b:>2} 個：分數 {s:.4f}（{s - path[b - 1][2]:+.4f}）  {add}")
    print(f"已存 {out_path} 與 {bg_path}")


main()
