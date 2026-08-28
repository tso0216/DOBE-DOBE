"""繪製 get_data.py 輸出的各 POI 類別加入不同數量後，與空間最近 k 鄰居的 latent 平均距離變化、改善比例與扣除總量效應後的淨效應"""
import csv
import math
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from common.dataset import CATEGORIES, CAT_ZH  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

COLORS = ["#4363d8", "#f58231", "#e6194b"]
UNIFORM = "UNIFORM"
UNIFORM_ZH = "均勻加\n（對照）"


def load_deltas(path):
    """path：data.csv 路徑。回傳 {(category, amount): {patch_id: delta}} 與由小到大的加入量清單。"""
    deltas = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["category"], int(row["amount"]))
            deltas.setdefault(key, {})[int(row["patch_id"])] = float(row["delta"])
    return deltas, sorted({a for _, a in deltas})


def stats(vs):
    """vs：一組 delta（距離變化）數值。回傳 (平均、95% 信賴區間半寬、小於 0 即距離縮短的比例)。"""
    n = len(vs)
    mean = sum(vs) / n
    var = sum((v - mean) ** 2 for v in vs) / max(n - 1, 1)
    return mean, 1.96 * math.sqrt(var / n), sum(v < 0 for v in vs) / n


def label_zh(cat):
    """cat：data.csv 裡的類別名。回傳圖上顯示的中文標籤。"""
    return UNIFORM_ZH if cat == UNIFORM else CAT_ZH[CATEGORIES.index(cat)]


def grouped_bar(ax, categories, amounts, values, errs=None):
    """ax：要畫的軸。categories/amounts：分組的類別與加入量。values：{(category, amount): 值}。
    errs：可選的 {(category, amount): 誤差半寬}。回傳 None，直接畫在 ax 上。"""
    bar_w = 0.8 / len(amounts)
    for i, amt in enumerate(amounts):
        xs = [c + i * bar_w for c in range(len(categories))]
        ys = [values[(cat, amt)] for cat in categories]
        es = [errs[(cat, amt)] for cat in categories] if errs else None
        ax.bar(xs, ys, width=bar_w, yerr=es, capsize=2, label=f"加入 {amt} 個",
               color=COLORS[i % len(COLORS)],
               error_kw={"elinewidth": 0.8, "ecolor": "#444444"})
    ax.set_xticks([c + bar_w * (len(amounts) - 1) / 2 for c in range(len(categories))])
    ax.set_xticklabels([label_zh(cat) for cat in categories],
                       rotation=30, ha="right", fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    if UNIFORM in categories:
        ax.axvline(categories.index(UNIFORM) - 0.2, color="#888888",
                   linewidth=0.8, linestyle=":")


def main():
    deltas, amounts = load_deltas(os.path.join(HERE, "data.csv"))
    cats = [c for c in CATEGORIES if (c, amounts[0]) in deltas]
    n_patch = len(deltas[(cats[0], amounts[0])])

    agg = {k: stats(list(v.values())) for k, v in deltas.items()}
    # 淨效應：同一個 patch 上，加單一類別與加同樣總量但均勻分配的差
    net = {(cat, amt): stats([d - deltas[(UNIFORM, amt)][pid] for pid, d in
                              deltas[(cat, amt)].items()])
           for cat in cats for amt in amounts}

    fig, axes = plt.subplots(3, 1, figsize=(11, 13))

    grouped_bar(axes[0], cats + [UNIFORM], amounts,
                {k: v[0] for k, v in agg.items()},
                {k: v[1] for k, v in agg.items()})
    axes[0].axhline(0, color="#000000", linewidth=0.8)
    axes[0].set_ylabel("鄰居平均距離變化 Δ")
    axes[0].set_title(f"各 POI 類別加入後與空間最近 8 鄰居的 latent 平均距離變化"
                      f"（負值＝更貼近鄰里；誤差棒為 95% CI，n={n_patch}）", fontsize=11)
    axes[0].legend(fontsize=9)

    grouped_bar(axes[1], cats + [UNIFORM], amounts,
                {k: v[2] * 100 for k, v in agg.items()})
    axes[1].axhline(50, color="#000000", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("距離縮短的 patch 比例（%）")
    axes[1].set_title("加入後與鄰居距離縮短的 patch 比例（虛線為 50%）", fontsize=11)
    axes[1].legend(fontsize=9)

    grouped_bar(axes[2], cats, amounts,
                {k: v[0] for k, v in net.items()},
                {k: v[1] for k, v in net.items()})
    axes[2].axhline(0, color="#000000", linewidth=0.8)
    axes[2].set_ylabel("淨效應 Δ − Δ(均勻加)")
    axes[2].set_title("扣除總量效應後的淨效應（負值＝比同總量均勻加更貼近鄰里，逐 patch 配對相減）",
                      fontsize=11)
    axes[2].legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(HERE, "1.3.1_affinity_geo.png")
    fig.savefig(out, dpi=150)
    print(f"已存 {out}")


main()
