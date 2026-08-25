"""繪製 get_data.py 輸出的 AE+entropy 不同訓練進度下，加 POI 前後的 2 維重建位移對照"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

out_dir = os.path.dirname(os.path.abspath(__file__))
SHIFT_CATEGORY = 'Travel and Transportation'
SHIFT_AMOUNT = 5

df = pd.read_csv(os.path.join(out_dir, "data.csv"))
percents = sorted(df["percent"].unique())

fig, axes = plt.subplots(1, len(percents), figsize=(3 * len(percents), 4), squeeze=False)
for ax, percent in zip(axes[0], percents):
    sub = df[df["percent"] == percent]
    ax.plot(np.stack([sub["z1_before"], sub["z1_after"]]),
            np.stack([sub["z2_before"], sub["z2_after"]]),
            color="gray", alpha=0.15, linewidth=0.6, zorder=1)
    ax.scatter(sub["z1_before"], sub["z2_before"], s=6, color="#4363d8",
               alpha=0.7, linewidths=0, label="平移前", zorder=2)
    ax.scatter(sub["z1_after"], sub["z2_after"], s=6, color="#f58231",
               alpha=0.7, linewidths=0, label="平移後", zorder=2)
    ax.set_title(f"訓練進度 {percent}%", fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
axes[0][0].legend(fontsize=8, loc="best")
n = df[df["percent"] == percents[0]].shape[0]
fig.suptitle(f"AE+entropy 不同訓練進度下的 2 維重建：「{SHIFT_CATEGORY}」+{SHIFT_AMOUNT} 前後對照（全部資料，n={n}）")
fig.tight_layout()

out = os.path.join(out_dir, "2.1.2_train_progress_shift.png")
fig.savefig(out, dpi=150)
print(f"已存 {out}")
