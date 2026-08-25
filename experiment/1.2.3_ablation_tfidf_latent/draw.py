"""繪製 get_data.py 輸出的 latent space：TF-IDF 處理前後分群對照"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

out_dir = os.path.dirname(os.path.abspath(__file__))
mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130

df = pd.read_csv(os.path.join(out_dir, "data.csv"))


def draw(ax, z1, z2, labels, title):
    """ax：matplotlib Axes。z1,z2：latent 座標。labels：分群標籤。title：子圖標題。"""
    cmap = plt.get_cmap("tab10")
    for k, c in enumerate(sorted(labels.unique())):
        m = labels == c
        ax.scatter(z1[m], z2[m], s=5, color=cmap(k % 10), linewidths=0,
                   alpha=0.8, label=f"c{c} ({int(m.sum())})", rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=6, markerscale=2, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), frameon=False)


fig, axes = plt.subplots(1, 2, figsize=(10, 6))
draw(axes[0], df["z1"], df["z2"], df["cluster_before"], "分群依據：log1p(count) 標準化（處理前）")
draw(axes[1], df["z1"], df["z2"], df["cluster_after"], "分群依據：TF-IDF 標準化（處理後）")
fig.suptitle("latent space：TF-IDF 處理前後分群對照")
fig.tight_layout()

out = os.path.join(out_dir, "1.2.3_ablation_tfidf_latent.png")
fig.savefig(out, bbox_inches="tight")
print(f"已存 {out}")
