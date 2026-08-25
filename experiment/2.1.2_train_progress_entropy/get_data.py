"""計算 AE+entropy 在不同訓練進度快照下，全部 patch 的 2 維 latent 座標與 TF-IDF KMeans 分群標籤"""
import os
import sys

import torch
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import PATCHES  # noqa: E402
from model.v3_ddae_tfidf.cfg import LATENT_DIM, N_CLUSTERS, SEED, device  # noqa: E402
from model.v3_ddae_tfidf.dataset import Patches  # noqa: E402
from model.v3_ddae_tfidf.model import AE, compute_tfidf_features  # noqa: E402

CKPT_DIR = os.path.join(HERE, "..", "model", "entropy_progress_ckpt")
SNAPSHOT_PERCENTS = [10, 50, 100]


def ckpt_path(percent):
    return os.path.join(CKPT_DIR, f"epoch_pct{percent}_seed{SEED}.pt")


def encode(percent, x):
    """percent：訓練進度，對應 ckpt_path 存的權重檔。x：(N, N_CAT) count tensor。
    回傳 (N, 2) numpy array 的 latent。
    """
    model = AE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(ckpt_path(percent), map_location=device))
    model.eval()
    with torch.no_grad():
        z = model.encode(x.to(device)).cpu().numpy()
    return z


def main():
    missing = [p for p in SNAPSHOT_PERCENTS if not os.path.exists(ckpt_path(p))]
    if missing:
        raise FileNotFoundError(f"找不到進度 {missing} 的權重，請確認 {CKPT_DIR} 底下的快照完整")

    data = Patches(PATCHES)
    x_all = data.agg(torch.arange(data.n))

    x_tfidf_all, _ = compute_tfidf_features(x_all.numpy())
    labels = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(x_tfidf_all)

    out = os.path.join(HERE, "data.csv")
    with open(out, "w") as f:
        f.write("percent,sample_id,z1,z2,cluster\n")
        for percent in sorted(SNAPSHOT_PERCENTS):
            z = encode(percent, x_all)
            for sample_id, ((z1, z2), c) in enumerate(zip(z, labels)):
                f.write(f"{percent},{sample_id},{z1},{z2},{c}\n")
            print(f"訓練進度 {percent}% 已編碼完成")
    print(f"已存 {out}")


main()
