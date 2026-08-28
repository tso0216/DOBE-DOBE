"""繪製 get_data.py 輸出的同時對所有類別加入 POI 後，每個 patch 的 latent 位移軌跡"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

out_dir = os.path.dirname(os.path.abspath(__file__))
NCOLS = 3
COLOR_BEFORE = "#4363d8"
COLOR_AFTER = "#e6194b"
COLOR_MID = "#9e9e9e"

df = pd.read_csv(os.path.join(out_dir, "data.csv"))
models = df["model"].unique().tolist()
steps = sorted(s for s in df["step"].unique() if s > 0)


def draw(ax, sub, title, target_step, target_amount):
    """ax：要畫的子圖。sub：單一 model 的資料（含 step/z1/z2 欄）。title：子圖標題。
    target_step：本張圖要標示為終點的 step。target_amount：該 step 對應的加入量。
    """
    for step in range(1, target_step):
        mid = sub[sub["step"] == step]
        ax.scatter(mid["z1"], mid["z2"], s=4, color=COLOR_MID,
                   alpha=0.8, linewidths=0, zorder=10)
    before = sub[sub["step"] == 0]
    after = sub[sub["step"] == target_step]
    ax.scatter(before["z1"], before["z2"], s=10, color=COLOR_BEFORE,
               alpha=0.85, linewidths=0, zorder=11, label="起點（原始）")
    ax.scatter(after["z1"], after["z2"], s=10, color=COLOR_AFTER,
               alpha=0.85, linewidths=0, zorder=12, label=f"終點（+{target_amount}）")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")


nrows = (len(models) + NCOLS - 1) // NCOLS
n_sample = df[(df["model"] == models[0]) & (df["step"] == 0)].shape[0]
for target_step in steps:
    target_amount = df.loc[df["step"] == target_step, "amount"].iloc[0]
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(5.0 * NCOLS, 4.6 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, models):
        draw(ax, df[df["model"] == name], name, target_step, target_amount)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle(f"同時對所有類別加入 {target_amount} 個 POI 後每個 patch 的 latent 位移（test set，n={n_sample}）")
    fig.tight_layout()

    out = os.path.join(out_dir, f"latent_shift_add{target_amount}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已存 {out}")
