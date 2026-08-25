"""繪製 get_data.py 輸出的 AE+entropy 不同訓練進度下的 2 維 latent space（依分群上色）"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

out_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(out_dir, "data.csv"))
percents = sorted(df["percent"].unique())
n_clusters = df["cluster"].nunique()
cmap = plt.get_cmap("tab10")

fig, axes = plt.subplots(1, len(percents), figsize=(4.2 * len(percents), 5), squeeze=False)
for ax, percent in zip(axes[0], percents):
    sub = df[df["percent"] == percent]
    for k in sorted(sub["cluster"].unique()):
        m = sub[sub["cluster"] == k]
        ax.scatter(m["z1"], m["z2"], s=6, color=cmap(k % 10), alpha=0.7,
                   linewidths=0, label=f"c{k} ({len(m)})")
    ax.set_title(f"訓練進度 {percent}%", fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
axes[0][-1].legend(fontsize=7, markerscale=2, loc="center left",
                   bbox_to_anchor=(1.02, 0.5), frameon=False)
n = df[df["percent"] == percents[0]].shape[0]
fig.suptitle(f"AE+entropy 不同訓練進度下的 2 維 latent space（依 TF-IDF KMeans 分群上色，全部資料，n={n}）")
fig.tight_layout()

out = os.path.join(out_dir, "2.1.2_train_progress_entropy.png")
fig.savefig(out, dpi=150)
print(f"已存 {out}")
