"""繪製 get_data.py 輸出的分數差距分布：x 軸為加入後與沒加時的分數差距，y 軸為落在該差距的 patch 數，一個類別一張子圖、一個加入量一條線"""
import csv
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from common.dataset import CATEGORIES, CAT_ZH  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

COLORS = ["#4363d8", "#f58231", "#e6194b"]
UNIFORM = "UNIFORM"
UNIFORM_ZH = "均勻加（對照）"
NEIGHBOR = "空間最近 8 鄰居"
OUT = "1.3.1_affinity_geo_dist.png"
BINS = 20
NCOLS = 4


def load_deltas(path):
    """path：data.csv 路徑。回傳 {(category, amount): [與沒加時的分數差距, ...]}
    與由小到大的加入量清單。"""
    deltas = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["category"], int(row["amount"]))
            deltas.setdefault(key, []).append(float(row["delta"]))
    return deltas, sorted({a for _, a in deltas})


def curve(vs, edges):
    """vs：一組差距值。edges：直方圖分箱邊界。回傳 (各 bin 中心, 落在各 bin 的 patch 數)。"""
    cnt, _ = np.histogram(vs, bins=edges)
    return (edges[:-1] + edges[1:]) / 2, cnt


def label_zh(cat):
    """cat：data.csv 裡的類別名。回傳圖上顯示的中文標籤。"""
    return UNIFORM_ZH if cat == UNIFORM else CAT_ZH[CATEGORIES.index(cat)]


def main():
    deltas, amounts = load_deltas(os.path.join(HERE, "data.csv"))
    cats = [c for c in CATEGORIES if (c, amounts[0]) in deltas] + [UNIFORM]

    pool = [v for vs in deltas.values() for v in vs]
    lo, hi = np.percentile(pool, [1, 99])
    edges = np.linspace(lo, hi, BINS + 1)

    nrows = math.ceil(len(cats) / NCOLS)
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(3.6 * NCOLS, 2.9 * nrows),
                             squeeze=False, sharex=True, sharey=True)

    for ax, cat in zip(axes.flat, cats):
        ax.axvline(0, color="#888888", linestyle="--", linewidth=1.2)
        for i, amt in enumerate(amounts):
            x, y = curve(deltas[(cat, amt)], edges)
            ax.plot(x, y, color=COLORS[i % len(COLORS)], linewidth=1.4,
                    label=f"加入 {amt} 個")
        ax.set_title(label_zh(cat), fontsize=10)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(cats):]:
        ax.axis("off")

    for ax in axes[-1]:
        ax.set_xlabel("與沒加時的分數差距 Δ（負值＝距離縮短）")
    for ax in axes[:, 0]:
        ax.set_ylabel("patch 數")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"加入各 POI 類別後，與{NEIGHBOR}的 latent 平均距離相對沒加時的差距分布",
                 fontsize=12)
    fig.tight_layout()

    out = os.path.join(HERE, OUT)
    fig.savefig(out, dpi=150)
    print(f"已存 {out}")


main()
