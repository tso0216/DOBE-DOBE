"""繪製 get_data.py 輸出的 cosine 相似度分佈直方圖"""
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

out_dir = os.path.dirname(os.path.abspath(__file__))
sim_dist = np.loadtxt(os.path.join(out_dir, 'data.csv'), delimiter=',', skiprows=1)

fig, ax = plt.subplots(figsize=(8, 5))
weights = np.ones_like(sim_dist) / len(sim_dist)
ax.hist(sim_dist, bins=50, weights=weights, color='skyblue', edgecolor='black')
ax.yaxis.set_major_formatter(PercentFormatter(1))

ax.set_title(f"model's input cosine similarity")
ax.set_xlabel('Cosine similarity')
ax.set_ylabel('佔比')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '0.3_input_similarity.png'), dpi=150)
