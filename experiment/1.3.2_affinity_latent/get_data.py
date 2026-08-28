"""統計 v3 模型下，每個 patch 分別加入各 POI 類別 1/2/3 個之後，與 latent 最近 k 個鄰居的平均距離變化"""
import csv
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
ADD_AMOUNT = [1, 2, 3]
UNIFORM = "UNIFORM"           # 總量對齊的對照組：同樣加 n 個 POI，但平均分給全部類別
K = 8                         # 取 latent 上最近的 k 個 patch 當鄰居
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
        return model.encoder(x).numpy().astype(np.float64)


def knn_neighbors(z_ref, k):
    """z_ref：(n, d) test patch 在擾動前的 latent，同時是被評分的點與鄰居的來源。k：鄰居數。
    回傳 (nb, nb_dist)：nb 為 (n, k) 的 z_ref 內索引，是每個 patch 最近的 k 個鄰居（不含自己）；
    nb_dist 為對應的 (n, k) latent 距離。"""
    dist, idx = cKDTree(z_ref).query(z_ref, k=k + 1)
    keep_mask = idx != np.arange(len(z_ref))[:, None]   # 有重複座標時 self 未必排在第一欄
    nb = np.array([row[m][:k] for row, m in zip(idx, keep_mask)])
    nb_dist = np.array([row[m][:k] for row, m in zip(dist, keep_mask)])
    return nb, nb_dist


def neighbor_dist(z_eval, z_ref, nb):
    """z_eval：(n, d) 被評分 patch 的 latent。z_ref：(n, d) 鄰居的 latent（擾動前，固定不動）。
    nb：(n, k) 鄰居索引。回傳 (n,) 每個 patch 到其 k 個鄰居的 latent 平均歐氏距離。"""
    diff = z_ref[nb] - z_eval[:, None, :]
    return np.sqrt((diff ** 2).sum(axis=2)).mean(axis=1)


def main():
    data = Patches(PATCHES)
    _, _, test_idx = make_split(data.lat, data.lon, seed=SEED)
    x_eval = data.agg(test_idx)
    eval_idx = test_idx.numpy()

    model = load_model(CKPT)
    z_ref = encode(model, x_eval)

    nb, nb_dist = knn_neighbors(z_ref, K)
    base = neighbor_dist(z_ref, z_ref, nb)

    out_path = os.path.join(HERE, "data.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patch_id", "category", "amount",
                         "dist_before", "dist_after", "delta"])
        for label in CATEGORIES + [UNIFORM]:
            for amt in ADD_AMOUNT:
                x_shift = x_eval.clone()
                if label == UNIFORM:
                    x_shift += amt / N_CAT
                else:
                    x_shift[:, CATEGORIES.index(label)] += amt
                after = neighbor_dist(encode(model, x_shift), z_ref, nb)
                for pos in range(len(eval_idx)):
                    writer.writerow([eval_idx[pos], label, amt,
                                     base[pos], after[pos],
                                     after[pos] - base[pos]])

    print(f"test patch {len(eval_idx)} 個，鄰居數 k={K}"
          f"（鄰居只從 test 集內、在擾動前的 latent 上選定並固定）")
    print(f"第 {K} 近鄰的 latent 距離：中位 {np.median(nb_dist[:, -1]):.4f}、"
          f"最大 {nb_dist[:, -1].max():.4f}")
    print(f"原始 latent 鄰居平均距離 {base.mean():.4f}")
    print(f"已存 {out_path}")


main()
