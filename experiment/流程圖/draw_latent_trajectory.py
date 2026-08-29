"""從目標 patch 出發，照 ADD 指定的類別逐顆加入 POI，
在 latent space 上把每加一顆的位置串成軌跡。

不做任何搜尋，加什麼完全由下面的 ADD 決定。
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow_common  # noqa: E402
from common.dataset import CAT_ZH, N_CAT  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 要加入的 POI 數，長度 10，順序同 common/dataset.py 的 CATEGORIES：
#  0 餐飲   1 零售   2 夜生活   3 社區/政府   4 交通
#  5 商業服務   6 地標/戶外   7 藝文娛樂   8 醫療   9 運動休閒
ADD = [6, 0, 0, 0, 0, 0, 0, 0, 0, 0]

ZOOM = 2.0   # 視野邊長（latent 單位），以目標 patch 為中心
DOT = 60     # 背景點大小

out_dir = os.path.dirname(os.path.abspath(__file__))

info = flow_common.load()
model, z_test, score, pos = info['model'], info['z_test'], info['score'], info['pos']
tree = info['tree']

add = np.asarray(ADD, dtype=np.float32)
assert len(add) == N_CAT, f"ADD 要有 {N_CAT} 個數字，目前 {len(add)} 個"

# ---- 一顆一顆加，每加一顆記一個點 ----
current = info['x_test'][pos].numpy().copy()
steps = [current.copy()]
for c in range(N_CAT):
    for _ in range(int(add[c])):
        current[c] += 1
        steps.append(current.copy())

traj_z = flow_common.encode(model, torch.from_numpy(np.stack(steps)))
traj_score = flow_common.knn_score(tree, traj_z)

print(f"目標 patch id={info['target']}（第 {flow_common.PCT} 百分位），"
      f"POI {int(info['data'].n_poi[info['target']])} 個")
print("加入：" + "、".join(f"{CAT_ZH[c]}×{int(v)}" for c, v in enumerate(add) if v > 0)
      + f"（共 {int(add.sum())} 顆）")
print(f"分數：{traj_score[0]:.4f} → {traj_score[-1]:.4f}")

# ---- 畫圖 ----
fig, ax = plt.subplots(figsize=(7, 7), layout='constrained')
ax.scatter(z_test[:, 0], z_test[:, 1], c=score, s=DOT, cmap=flow_common.CMAP,
           linewidths=0, alpha=0.7, rasterized=True)

ax.plot(traj_z[:, 0], traj_z[:, 1], color='#111111', linewidth=1.2, zorder=3)
ax.scatter(traj_z[1:-1, 0], traj_z[1:-1, 1], s=70, c='#ffffff',
           edgecolors='#111111', linewidths=1.0, zorder=4)
ax.scatter(traj_z[0, 0], traj_z[0, 1], s=1500, facecolors='none',
           edgecolors='#d62728', linewidths=1.5, zorder=5)
ax.scatter(traj_z[0, 0], traj_z[0, 1], s=500, marker='*', c='#d62728',
           edgecolors='#111111', linewidths=0.8, zorder=6)
ax.scatter(traj_z[-1, 0], traj_z[-1, 1], s=100, marker='D', c='#ffdd57',
           edgecolors='#111111', linewidths=0.8, zorder=6)

center, half = traj_z[0], ZOOM / 2
ax.set_xlim(center[0] - half, center[0] + half)
ax.set_ylim(center[1] - half, center[1] + half)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])
ax.grid(alpha=0.15)

out = os.path.join(out_dir, 'draw_latent_trajectory.png')
fig.savefig(out, dpi=150)
print(f"已存 {out}")
