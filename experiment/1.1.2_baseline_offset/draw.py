"""繪製 get_data.py 輸出的 PCA / AE / VAE / Ours，各 POI 類別加入不同數量後的 latent 平均偏移"""
import csv
import math
import os

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

out_dir = os.path.dirname(os.path.abspath(__file__))
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#808000"]


def load_offsets(path):
    """path：data.csv 路徑。回傳 {(model, category): {amount: avg_offset}} 與依出現順序排列的 model 名稱、category 名稱清單。"""
    offsets = {}
    models, categories = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] not in models:
                models.append(row["model"])
            if row["category"] not in categories:
                categories.append(row["category"])
            key = (row["model"], row["category"])
            offsets.setdefault(key, {})[int(row["amount"])] = float(row["avg_offset"])
    return offsets, models, categories


def main():
    offsets, names, category = load_offsets(os.path.join(out_dir, "data.csv"))
    amounts = sorted({a for v in offsets.values() for a in v})

    ncols = min(len(category), 5)
    nrows = math.ceil(len(category) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    bar_w = 0.8 / len(names)
    x_base = range(len(amounts))
    for ax, cat in zip(axes.flat, category):
        for i, name in enumerate(names):
            ys = [offsets.get((name, cat), {}).get(a, 0.0) for a in amounts]
            ax.bar([x + i * bar_w for x in x_base], ys, width=bar_w,
                   label=name, color=COLORS[i % len(COLORS)])
        ax.set_title(cat, fontsize=10)
        ax.set_xticks([x + bar_w * (len(names) - 1) / 2 for x in x_base])
        ax.set_xticklabels(amounts)
        ax.grid(alpha=0.2, axis="y")
    for ax in axes.flat[len(category):]:
        ax.axis("off")
    last_row_used = len(category) - (nrows - 1) * ncols
    for ax in axes[-1][:last_row_used]:
        ax.set_xlabel("加入量")
    for ax in axes[:, 0]:
        ax.set_ylabel("latent 平均偏移")
    axes.flat[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("各 POI 類別加入不同數量後的 latent 平均偏移（baseline）")
    fig.tight_layout()

    out = os.path.join(out_dir, "1.1.1_baseline_offset.png")
    fig.savefig(out, dpi=150)


main()
