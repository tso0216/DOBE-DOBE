import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import CAT_ZH, N_CAT, PATCHES  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.dirname(script_dir)
os.makedirs(out_dir, exist_ok=True)

d = np.load(PATCHES)
offsets, cat = d['offsets'], d['cat']
n = len(offsets) - 1

x = np.zeros((n, N_CAT), dtype=np.float64)
for i in range(n):
    c = cat[offsets[i]:offsets[i + 1]]
    x[i] = np.bincount(c, minlength=N_CAT)

sim = 1 - squareform(pdist(x, metric='cosine'))
order = leaves_list(linkage(pdist(x, metric='cosine'), method='average'))
sim_ordered = sim[order][:, order]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(sim_ordered, cmap='viridis', vmin=0, vmax=1)
ax.set_title(f'模型輸入間 cosine 相似度 (n={n})')
ax.set_xlabel('patch（依類排）')
ax.set_ylabel('patch（依類排）')
fig.colorbar(im, ax=ax, label='cosine similarity')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '0_input_similarity_matrix.png'), dpi=150)
print(f'類別順序：{CAT_ZH}')
print(f'已存 {os.path.join(out_dir, "0_input_similarity.png")}')
