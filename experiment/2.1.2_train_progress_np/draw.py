"""繪製 get_data.py 輸出的 AE+entropy 近鄰保留率隨訓練進度變化折線圖"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

out_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(out_dir, "data.csv"))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(df["percent"], df["neighbor_preservation"], marker="o", color="#4363d8")
ax.set_xlabel("訓練進度 (%)")
ax.set_ylabel("近鄰保留率")
ax.set_title("AE+entropy 近鄰保留率隨訓練進度變化（全部資料）", fontsize=11)
ax.grid(alpha=0.2)
fig.tight_layout()

out = os.path.join(out_dir, "2.1.2_train_progress_np.png")
fig.savefig(out, dpi=150)
print(f"已存 {out}")
