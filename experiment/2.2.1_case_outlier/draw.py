"""繪製 get_data.py 輸出的窮舉軌跡：latent 散點依分數上色，各預算下的最佳位置連成軌跡（只標起點與終點），另附分數曲線與類別組成對照"""
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from common.dataset import CATEGORIES, CAT_ZH  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

CASE = "outlier"
OBJ = "min"                   # min：目標是把分數壓到最低；max：推到最高
OUT = "2.2.1_case_outlier.png"
CMAP = "viridis"


def load_background(path):
    """path：background.csv 路徑。回傳 (z, score) 兩個等長 array。"""
    z, score = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            z.append((float(row["z1"]), float(row["z2"])))
            score.append(float(row["score"]))
    return np.array(z), np.array(score)


def load_path(path):
    """path：data.csv 路徑。回傳 (z, score, comp)：z 為 (steps, 2) 軌跡座標、
    score 為各步分數、comp 為 (steps, N_CAT) 的類別組成，第 0 列是原始狀態。"""
    z, score, comp = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            z.append((float(row["z1"]), float(row["z2"])))
            score.append(float(row["score"]))
            comp.append([float(row[c]) for c in CATEGORIES])
    return np.array(z), np.array(score), np.array(comp)


def plot_latent(ax, zb, sb, zp, sp, best):
    """ax：要畫的軸。zb/sb：背景 patch 的座標與分數。zp/sp：軌跡座標與分數。
    best：達成目標的那一步。回傳 scatter 物件供建立 colorbar。"""
    sc = ax.scatter(zb[:, 0], zb[:, 1], c=sb, s=10, cmap=CMAP,
                    linewidths=0, alpha=0.7, rasterized=True)

    ax.plot(zp[:, 0], zp[:, 1], color="#111111", linewidth=1.2, zorder=3)
    ax.scatter(zp[1:-1, 0], zp[1:-1, 1], s=22, c="#ffffff",
               edgecolors="#111111", linewidths=1.0, zorder=4,
               label="各預算下的最佳位置")
    ax.scatter(zp[0, 0], zp[0, 1], s=170, marker="*", c="#d62728",
               edgecolors="#111111", linewidths=0.8, zorder=5, label="起點（原始）")
    ax.scatter(zp[best, 0], zp[best, 1], s=40, marker="D", c="#ffdd57",
               edgecolors="#111111", linewidths=0.8, zorder=5,
               label=f"終點（加 {best} 個）")
    for i, txt in ((0, f"起點 {sp[0]:.2f}"), (best, f"終點 {sp[best]:.2f}")):
        ax.annotate(txt, zp[i], textcoords="offset points", xytext=(10, 6),
                    fontsize=9, zorder=6)

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8, loc="best")
    return sc


def added_text(comp, best):
    """comp：(steps, N_CAT) 組成。best：終點那一步。回傳終點相對原始多加了哪些類別的中文字串。"""
    delta = comp[best] - comp[0]
    return "、".join(f"{CAT_ZH[c]}×{int(v)}" for c, v in enumerate(delta) if v > 0)


def main():
    zb, sb = load_background(os.path.join(HERE, "background.csv"))
    zp, sp, comp = load_path(os.path.join(HERE, "data.csv"))
    best = int(np.argmin(sp) if OBJ == "min" else np.argmax(sp))
    median = float(np.median(sb))

    fig = plt.figure(figsize=(15, 7), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_curve = fig.add_subplot(gs[0, 1])
    ax_comp = fig.add_subplot(gs[1, 1])

    sc = plot_latent(ax_map, zb, sb, zp, sp, best)
    fig.colorbar(sc, ax=ax_map, label="分數（與 latent 最近 8 鄰居的平均距離）")
    ax_map.set_title("latent 空間中的窮舉軌跡（點依分數上色，軌跡串起每個預算的最佳組合）",
                     fontsize=11)

    steps = np.arange(len(sp))
    ax_curve.plot(steps, sp, marker="o", color="#3a6ea5", linewidth=1.4)
    ax_curve.axhline(median, color="#888888", linestyle="--", linewidth=1.0,
                     label=f"全體中位數 {median:.2f}")
    ax_curve.scatter([best], [sp[best]], s=90, marker="D", c="#ffdd57",
                     edgecolors="#111111", linewidths=0.8, zorder=5)
    ax_curve.set_xlabel("加入的 POI 數")
    ax_curve.set_ylabel("分數")
    ax_curve.set_xticks(steps)
    ax_curve.grid(alpha=0.2)
    ax_curve.legend(fontsize=8)
    ax_curve.set_title("每個預算窮舉後的最佳分數", fontsize=11)

    idx = np.arange(len(CATEGORIES))
    ax_comp.bar(idx - 0.2, comp[0], width=0.4, color="#888888", label="原始")
    ax_comp.bar(idx + 0.2, comp[best], width=0.4, color="#e6194b",
                label=f"加 {best} 個後")
    ax_comp.set_xticks(idx)
    ax_comp.set_xticklabels(CAT_ZH, rotation=30, ha="right", fontsize=8)
    ax_comp.set_ylabel("POI 數")
    ax_comp.grid(alpha=0.2, axis="y")
    ax_comp.legend(fontsize=8)
    ax_comp.set_title(f"類別組成對照（加入：{added_text(comp, best)}）", fontsize=11)

    fig.suptitle(f"{CASE} 案例：分數 {sp[0]:.2f} → {sp[best]:.2f}"
                 f"（窮舉加入 {best} 個 POI）", fontsize=13)
    out = os.path.join(HERE, OUT)
    fig.savefig(out, dpi=150)
    print(f"已存 {out}")


main()
