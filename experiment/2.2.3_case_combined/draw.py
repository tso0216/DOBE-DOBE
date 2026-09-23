"""把 2.2.1（outlier）與 2.2.2（common）兩個案例的 data.csv / background.csv 合成論文用的英文 2x2 圖：
左欄為 latent 窮舉軌跡（背景點為 train+val 參考集，依分數上色），右欄為加入前後的類別組成"""
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.abspath(os.path.join(EXP, "..")))
from common.dataset import CATEGORIES  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 11,
})

CASES = [  # (資料夾, 標題, OBJ)
    ("2.2.2_case_common", "Common", "max"),
    ("2.2.1_case_outlier", "Outlier", "min"),
]
SHORT = [c.replace(" and ", " & ").replace(" Services", "") for c in CATEGORIES]
CMAP = "viridis"
OUT = "case_combined.png"


def load_background(path):
    """path：background.csv 路徑。回傳 (z, score)。"""
    z, score = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            z.append((float(row["z1"]), float(row["z2"])))
            score.append(float(row["score"]))
    return np.array(z), np.array(score)


def load_path(path):
    """path：data.csv 路徑。回傳 (z, score, comp)，第 0 列是原始狀態。"""
    z, score, comp = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            z.append((float(row["z1"]), float(row["z2"])))
            score.append(float(row["score"]))
            comp.append([float(row[c]) for c in CATEGORIES])
    return np.array(z), np.array(score), np.array(comp)


def main():
    data = []
    for folder, name, obj in CASES:
        d = os.path.join(EXP, folder)
        zb, sb = load_background(os.path.join(d, "background.csv"))
        zp, sp, comp = load_path(os.path.join(d, "data.csv"))
        best = int(np.argmin(sp) if obj == "min" else np.argmax(sp))
        data.append((name, zb, sb, zp, sp, comp, best))
    vmin = min(x[2].min() for x in data)
    vmax = max(x[2].max() for x in data)

    fig = plt.figure(figsize=(4.8, 5.4), dpi=300)
    # 第 3 欄是空白間隔，避免 colorbar 刻度壓到右欄的 y 軸標籤
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 0.05, 0.62, 1], wspace=0.05,
                          hspace=0.35, left=0.1, right=0.985, top=0.9, bottom=0.27)
    fig.suptitle("Latent Trajectory and Category Breakdown Comparison",
                 fontweight="bold", fontsize=12, y=0.985)

    idx = np.arange(1, len(CATEGORIES) + 1)
    ymax = max(x[5].max() for x in data) * 1.12
    for r, (name, zb, sb, zp, sp, comp, best) in enumerate(data):
        ax = fig.add_subplot(gs[r, 0])
        sc = ax.scatter(zb[:, 0], zb[:, 1], c=sb, s=4, cmap=CMAP, vmin=vmin,
                        vmax=vmax, linewidths=0, alpha=0.7, rasterized=True)
        ax.plot(zp[:best + 1, 0], zp[:best + 1, 1], color="#111111",
                linewidth=0.9, zorder=3)
        ax.scatter(zp[1:best, 0], zp[1:best, 1], s=14, c="#ffffff",
                   edgecolors="#111111", linewidths=0.7, zorder=4)
        ax.scatter(*zp[0], s=90, marker="*", c="#d62728", edgecolors="#111111",
                   linewidths=0.5, zorder=5)
        ax.scatter(*zp[best], s=30, marker="D", c="#ffdd57",
                   edgecolors="#111111", linewidths=0.5, zorder=5)
        for i, txt in ((0, f"Start: {sp[0]:.2f}"), (best, f"End: {sp[best]:.2f}")):
            ax.annotate(txt, zp[i], textcoords="offset points", xytext=(5, 4),
                        fontsize=8.5, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                  ec="none", alpha=0.7))
        ax.set_title(f"{name} Case ({sp[0]:.2f} → {sp[best]:.2f})", fontsize=11)
        ax.set_ylabel("z2")
        ax.grid(alpha=0.15)
        if r == len(data) - 1:
            ax.set_xlabel("z1")

        bx = fig.add_subplot(gs[r, 3])
        bx.bar(idx - 0.2, comp[0], width=0.4, color="#888888")
        bx.bar(idx + 0.2, comp[best], width=0.4, color="#e6194b")
        bx.set_title(f"{name} Breakdown", fontsize=11)
        bx.set_ylabel("POI Count")
        bx.set_ylim(0, ymax)
        bx.set_xticks(idx)
        bx.grid(axis="y", alpha=0.2)
        if r == len(data) - 1:
            bx.set_xlabel("POI Category")
            bx.tick_params(axis="x", labelsize=9)
        else:
            bx.set_xticklabels([])

    cax = fig.add_subplot(gs[:, 1])
    fig.colorbar(sc, cax=cax, label="Score")

    handles = [
        Line2D([], [], marker="*", ls="", ms=9, mfc="#d62728", mec="#111111",
               label="Start (Original)"),
        Line2D([], [], marker="o", ls="", ms=4, mfc="#ffffff", mec="#111111",
               label="Best per Budget"),
        Patch(color="#e6194b", label="After POI Addition"),
        Line2D([], [], marker="D", ls="", ms=5, mfc="#ffdd57", mec="#111111",
               label=f"End (+{data[0][6]} POIs)"),
        Patch(color="#888888", label="Original"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.095), frameon=True, columnspacing=1.0,
               handletextpad=0.4)
    items = [f"{i}. {s}" for i, s in enumerate(SHORT, 1)]
    key = "\n".join(" | ".join(items[a:b]) for a, b in ((0, 3), (3, 6), (6, 10)))
    fig.text(0.5, 0.012, key, ha="center", va="bottom", fontsize=8,
             color="#333333", linespacing=1.3)

    out = os.path.join(HERE, OUT)
    fig.savefig(out, dpi=300)
    print(f"已存 {out}")


main()
