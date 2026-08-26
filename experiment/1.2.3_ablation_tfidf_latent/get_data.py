"""計算 TF-IDF 前後的分群標籤，及 Ours（dae_tfidf）模型的 2 維 latent 座標"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import N_CAT, PATCHES  # noqa: E402
from model.other.v2_ddae_base.dataset import Patches  # noqa: E402
from model.v3_ddae_tfidf.model import compute_tfidf_features  # noqa: E402

MODEL_DIR = os.path.join(HERE, "..", "model")
WEIGHT_NAME = "ddae_fsce_tfidf/fold3_mae.pt"
N_CLUSTERS = 8
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


def encode_all(model, data, batch=256):
    """model：AE。data：Patches。回傳 (N,2) 全部 patch 的 latent 座標。"""
    zs = []
    with torch.no_grad():
        for i in range(0, data.n, batch):
            idx = torch.arange(i, min(i + batch, data.n))
            zs.append(model(data.agg(idx))[0])
    return torch.cat(zs).numpy()


def cluster_labels(x_std):
    """x_std：(N, D) 已標準化的特徵矩陣。回傳 (N,) KMeans 分群標籤（K=N_CLUSTERS）。"""
    return KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(x_std)


def align_labels(z, labels_ref, labels_target):
    """z：(N,2) 兩邊共用的 latent 座標。labels_ref：對照基準的分群標籤。labels_target：待對齊的分群標籤。
    回傳重新編號後的 labels_target：群心在 z 空間裡離 labels_ref 的哪一群最近，就改編成那群的編號。"""
    k = N_CLUSTERS
    centroid_ref = np.stack([z[labels_ref == c].mean(0) for c in range(k)])
    centroid_tgt = np.stack([z[labels_target == c].mean(0) for c in range(k)])
    cost = np.linalg.norm(centroid_ref[:, None, :] - centroid_tgt[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    remap = np.empty(k, dtype=int)
    remap[col_ind] = row_ind
    return remap[labels_target]


def main():
    data = Patches(PATCHES)
    x_all = data.agg(torch.arange(data.n)).numpy()

    x_before = RobustScaler().fit_transform(np.log1p(x_all))
    x_tfidf, _ = compute_tfidf_features(x_all)
    x_after = RobustScaler().fit_transform(x_tfidf)

    labels_before = cluster_labels(x_before)
    labels_after = cluster_labels(x_after)

    model = load_model(os.path.join(MODEL_DIR, WEIGHT_NAME))
    z = encode_all(model, data)

    labels_after = align_labels(z, labels_before, labels_after)

    out = os.path.join(HERE, "data.csv")
    with open(out, "w") as f:
        f.write("z1,z2,cluster_before,cluster_after\n")
        for (z1, z2), cb, ca in zip(z, labels_before, labels_after):
            f.write(f"{z1},{z2},{cb},{ca}\n")
    print(f"已存 {out}")


main()
