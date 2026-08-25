"""模型輸入 patch 間 cosine 相似度分佈 (僅限 fsq 資料集)"""
import os
import numpy as np
from scipy.spatial.distance import pdist

# 僅限 fsq 資料集
N_CAT = 10
CAT_ZH = ["餐飲", "零售", "夜生活", "社區/政府", "交通", "商業服務", "地標/戶外", "藝文娛樂", "醫療", "運動休閒"]
PATCHES = os.path.join(os.path.dirname(__file__), "../../data/patch/patches.npz")

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)

d = np.load(PATCHES)
offsets, cat = d['offsets'], d['cat']
n = len(offsets) - 1

x = np.zeros((n, N_CAT), dtype=np.float64)
for i in range(n):
    c = cat[offsets[i]:offsets[i + 1]]
    x[i] = np.bincount(c, minlength=N_CAT)

sim_dist = 1 - pdist(x, metric='cosine')

np.savetxt(os.path.join(out_dir, 'data.csv'), sim_dist, delimiter=',', header='cosine_similarity', comments='')
print(f'類別順序：{CAT_ZH}')
