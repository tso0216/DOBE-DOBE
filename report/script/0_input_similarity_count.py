"""模型輸入 patch 間 cosine 相似度分佈 (僅限 fsq 資料集)"""
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from matplotlib.ticker import PercentFormatter

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 僅限 fsq 資料集
N_CAT = 10
CAT_ZH = ["餐飲", "零售", "夜生活", "社區/政府", "交通", "商業服務", "地標/戶外", "藝文娛樂", "醫療", "運動休閒"]
PATCHES = os.path.join(os.path.dirname(__file__), "../../data/patch/patches.npz")

out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(out_dir, exist_ok=True)

d = np.load(PATCHES)
offsets, cat = d['offsets'], d['cat']
n = len(offsets) - 1

x = np.zeros((n, N_CAT), dtype=np.float64)
for i in range(n):
    c = cat[offsets[i]:offsets[i + 1]]
    x[i] = np.bincount(c, minlength=N_CAT)

sim_dist = 1 - pdist(x, metric='cosine')

fig, ax = plt.subplots(figsize=(8, 5))
weights = np.ones_like(sim_dist) / len(sim_dist)
ax.hist(sim_dist, bins=50, weights=weights, color='skyblue', edgecolor='black')
ax.yaxis.set_major_formatter(PercentFormatter(1))

ax.set_title(f'模型輸入 patch 間 cosine 相似度分佈 (n={n}, pairs={len(sim_dist)})')
ax.set_xlabel('Cosine 相似度')
ax.set_ylabel('佔比')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '0_input_similarity_count.png'), dpi=150)
print(f'類別順序：{CAT_ZH}')
