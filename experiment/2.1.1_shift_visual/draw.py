"""繪製 get_data.py 輸出的各類別加入 POI 後，每個 patch 的 latent 位移軌跡（每類別各存一張圖）"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

out_dir = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(out_dir, "latent_shift")
NCOLS = 3
COLOR_BEFORE = "#4363d8"
COLOR_AFTER = "#e6194b"
COLOR_MID = "#9e9e9e"

df = pd.read_csv(os.path.join(out_dir, "data.csv"))
models = df["model"].unique().tolist()
categories = df["category"].unique().tolist()
max_step = df["step"].max()
max_amount = df.loc[df["step"] == max_step, "amount"].iloc[0]


def draw(ax, sub, title):
    """ax：要畫的子圖。sub：單一 model+category 的資料（含 step/z1/z2 欄）。title：子圖標題。"""
    for step in range(1, max_step):
        mid = sub[sub["step"] == step]
        ax.scatter(mid["z1"], mid["z2"], s=4, color=COLOR_MID,
                   alpha=0.8, linewidths=0, zorder=10)
    before = sub[sub["step"] == 0]
    after = sub[sub["step"] == max_step]
    ax.scatter(before["z1"], before["z2"], s=10, color=COLOR_BEFORE,
               alpha=0.85, linewidths=0, zorder=11, label="起點（原始）")
    ax.scatter(after["z1"], after["z2"], s=10, color=COLOR_AFTER,
               alpha=0.85, linewidths=0, zorder=12, label=f"終點（+{max_amount}）")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")


os.makedirs(PLOT_DIR, exist_ok=True)
nrows = (len(models) + NCOLS - 1) // NCOLS
for cat_idx, category in enumerate(categories):
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(5.0 * NCOLS, 4.6 * nrows), squeeze=False)
    cat_df = df[df["category"] == category]
    for ax, name in zip(axes.flat, models):
        draw(ax, cat_df[cat_df["model"] == name], name)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8, loc="best", frameon=False)
    n_sample = cat_df[(cat_df["model"] == models[0]) & (cat_df["step"] == 0)].shape[0]
    fig.suptitle(f"在「{category}」加入 POI 後每個 patch 的 latent 位移（test set，n={n_sample}）")
    fig.tight_layout()

    out = os.path.join(PLOT_DIR, f"{cat_idx:02d}_{category.replace(' ', '_')}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已存 {out}")
