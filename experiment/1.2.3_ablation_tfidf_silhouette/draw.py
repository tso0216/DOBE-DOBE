"""繪製 get_data.py 輸出的 TF-IDF 處理前後分群品質（silhouette）比較長條圖"""
import os

import matplotlib.pyplot as plt
import pandas as pd

out_dir = os.path.dirname(os.path.abspath(__file__))
mpl_font = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['font.sans-serif'] = mpl_font
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(os.path.join(out_dir, "data.csv"))
names = df["stage"].tolist()
vals = df["silhouette"].tolist()

fig, ax = plt.subplots(figsize=(5, 5))
colors = ["#4363d8", "#f58231"]
bars = ax.bar(names, vals, color=colors[:len(names)], alpha=0.85, width=0.5)
pad = 0.03 * max(abs(v) for v in vals)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + (pad if v >= 0 else -pad),
             f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
ax.axhline(0, color="black", linewidth=0.8)
ax.margins(y=0.15)
ax.set_ylabel("Silhouette score")
ax.set_title("TF-IDF 處理前後分群品質比較")
ax.grid(alpha=0.2, axis="y")
fig.tight_layout()

out = os.path.join(out_dir, "1.2.3_ablation_tfidf_silhouette.png")
fig.savefig(out, bbox_inches="tight")
print(f"已存 {out}")
