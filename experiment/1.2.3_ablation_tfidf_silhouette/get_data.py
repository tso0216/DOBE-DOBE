"""計算 TF-IDF 處理前後的分群 silhouette 分數"""
import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import PATCHES  # noqa: E402
from model.other.v2_ddae_base.dataset import Patches  # noqa: E402
from model.v3_ddae_tfidf.model import compute_tfidf_features  # noqa: E402

N_CLUSTERS = 8
SEED = 0


def cluster_labels(x_std):
    """x_std：(N, D) 已標準化的特徵矩陣。回傳 (N,) KMeans 分群標籤（K=N_CLUSTERS）。"""
    return KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(x_std)


def main():
    data = Patches(PATCHES)
    x_all = data.agg(torch.arange(data.n)).numpy()

    x_before = RobustScaler().fit_transform(np.log1p(x_all))
    x_tfidf, _ = compute_tfidf_features(x_all)
    x_after = RobustScaler().fit_transform(x_tfidf)

    labels_before = cluster_labels(x_before)
    labels_after = cluster_labels(x_after)

    scores = [
        ("處理前 (log1p)", silhouette_score(x_before, labels_before)),
        ("處理後 (TF-IDF)", silhouette_score(x_after, labels_after)),
    ]

    out = os.path.join(HERE, "data.csv")
    with open(out, "w") as f:
        f.write("stage,silhouette\n")
        for stage, score in scores:
            f.write(f"{stage},{score:.5f}\n")
            print(f"{stage}: silhouette = {score:.4f}")
    print(f"已存 {out}")


main()
