"""計算 AE+entropy 在不同訓練進度快照下，近鄰保留率隨訓練進度的變化"""
import os
import sys

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import PATCHES  # noqa: E402
from model.v3_ddae_tfidf.cfg import LATENT_DIM, N_NEIGHBORS, device  # noqa: E402
from model.v3_ddae_tfidf.dataset import Patches  # noqa: E402
from model.v3_ddae_tfidf.model import AE, compute_tfidf_features  # noqa: E402

CKPT_DIR = os.path.join(HERE, "..", "model", "entropy_progress_ckpt")
FOLD = 3
SNAPSHOT_PERCENTS = list(range(5, 101, 5))


def ckpt_path(percent):
    return os.path.join(CKPT_DIR, f"epoch_pct{percent}_fold{FOLD}.pt")


def neighbor_preservation(z, neighbors_high, k):
    """z：(N, 2) 低維 latent。neighbors_high：每個點在高維空間的 k 近鄰 index set 清單。
    k：近鄰數。回傳全部點的平均近鄰保留率（高維/低維近鄰集合的重疊比例）。
    """
    knn_low = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(z)
    _, idx_low = knn_low.kneighbors(z)
    overlaps = [len(neighbors_high[i] & set(idx_low[i][1:])) / k for i in range(len(z))]
    return float(np.mean(overlaps))


def main():
    missing = [p for p in SNAPSHOT_PERCENTS if not os.path.exists(ckpt_path(p))]
    if missing:
        raise FileNotFoundError(f"找不到進度 {missing} 的權重，請確認 {CKPT_DIR} 底下的快照完整")

    data = Patches(PATCHES)
    x_all = data.agg(torch.arange(data.n))

    x_tfidf_all, _ = compute_tfidf_features(x_all.numpy())
    k = N_NEIGHBORS
    knn_high = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(x_tfidf_all)
    _, idx_high = knn_high.kneighbors(x_tfidf_all)
    neighbors_high = [set(row[1:]) for row in idx_high]

    out = os.path.join(HERE, "data.csv")
    with open(out, "w") as f:
        f.write("percent,neighbor_preservation\n")
        for percent in sorted(SNAPSHOT_PERCENTS):
            model = AE(LATENT_DIM).to(device)
            model.load_state_dict(torch.load(ckpt_path(percent), map_location=device))
            model.eval()
            with torch.no_grad():
                z = model.encode(x_all.to(device)).cpu().numpy()
            score = neighbor_preservation(z, neighbors_high, k)
            f.write(f"{percent},{score:.5f}\n")
            print(f"{percent}%: 近鄰保留率(k={k}) = {score:.4f}")
    print(f"已存 {out}")


main()
