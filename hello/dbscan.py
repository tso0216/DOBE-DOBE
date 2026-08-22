import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VERSION = "v2_ddae_base"

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model", VERSION))
from common.dataset import PATCHES  # noqa: E402
from dataset import Patches  # noqa: E402
from api import load_model, encode  # noqa: E402

OUT = os.path.join(HERE, "dbscan_latent.png")

EPS = 1.75         # log1p + z-score 後的鄰域半徑（5-NN 距離中位數約 1.43），
MIN_SAMPLES = 5    # 超過 1.6 整份資料會滲流成單一巨群
DOT = 6.0

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def cluster(x):
    """x：(N,N_CAT) 原始 count。回傳 (N,) 的 DBSCAN 標籤，-1 是 noise。"""
    # log1p 壓掉 count 的長尾，再逐類 z-score，避免餐飲這種大類主導歐氏距離
    xs = StandardScaler().fit_transform(np.log1p(x))
    return DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(xs)


def main():
    data = Patches(PATCHES)
    x = data.agg(torch.arange(data.n))

    label = cluster(x.numpy())
    z = encode(load_model(), x).numpy()

    ids = sorted(set(label.tolist()) - {-1})
    print(f"{data.n} 個 patch，{len(ids)} 群，noise {int((label == -1).sum())} 個")
    for c in ids:
        print(f"  cluster {c}: {int((label == c).sum())}")

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    m = label == -1
    ax.scatter(z[m, 0], z[m, 1], s=DOT * 0.7, c="#bbbbbb", linewidths=0,
               alpha=0.5, label=f"noise ({int(m.sum())})", rasterized=True)
    for k, c in enumerate(ids):
        m = label == c
        ax.scatter(z[m, 0], z[m, 1], s=DOT, color=cmap(k % 20), linewidths=0,
                   alpha=0.85, label=f"c{c} ({int(m.sum())})", rasterized=True)

    ax.set_title(f"{VERSION} latent（顏色＝原始 10 維標準化空間的 DBSCAN 分群，"
                 f"eps={EPS}）", fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, markerscale=2, loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
