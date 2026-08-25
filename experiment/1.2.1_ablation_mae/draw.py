"""繪製 get_data.py 輸出的六個 AE/DAE 變體 ablation MAE 比較長條圖"""
import os

import matplotlib.pyplot as plt
import pandas as pd

out_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(out_dir, "data.csv"))

fig, ax = plt.subplots(figsize=(max(4, len(df) * 2), 4.5))
bars = ax.bar(df["model"], df["mae"], color="#4363d8", width=0.6)
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
ax.set_xlabel("Model")
ax.set_ylabel("MAE")
ax.set_ylim(0, df["mae"].max() * 1.15)
ax.grid(alpha=0.15, axis="y")
fig.tight_layout()

out = os.path.join(out_dir, "1.2.1_ablation_mae.png")
fig.savefig(out, dpi=150)
