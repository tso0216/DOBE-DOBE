"""畫出 test set 的 latent 分布（點依分數上色），並標出目標 patch。
版型比照 experiment/2.2.1_case_outlier 的左半張圖。
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow_common  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

ZOOM = 2.0   # 視野邊長（latent 單位），以目標 patch 為中心
DOT = 60     # 背景點大小

out_dir = os.path.dirname(os.path.abspath(__file__))

info = flow_common.load()
z, score, pos = info['z_test'], info['score'], info['pos']

fig, ax = plt.subplots(figsize=(7, 7), layout='constrained')
ax.scatter(z[:, 0], z[:, 1], c=score, s=DOT, cmap=flow_common.CMAP,
           linewidths=0, alpha=0.7, rasterized=True)

ax.scatter(z[pos, 0], z[pos, 1], s=1500, facecolors='none', edgecolors='#d62728',
           linewidths=1.5, zorder=5)
ax.scatter(z[pos, 0], z[pos, 1], s=500, marker='*', c='#d62728',
           edgecolors='#111111', linewidths=0.8, zorder=6)

center, half = z[pos], ZOOM / 2
ax.set_xlim(center[0] - half, center[0] + half)
ax.set_ylim(center[1] - half, center[1] + half)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])
ax.grid(alpha=0.15)

out = os.path.join(out_dir, 'draw_latent.png')
fig.savefig(out, dpi=150)

print(f"test set {len(z)} 個 patch，分數中位數 {float(score.mean()):.4f}")
print(f"目標 patch id={info['target']}（第 {flow_common.PCT} 百分位），"
      f"POI {int(info['data'].n_poi[info['target']])} 個，分數 {score[pos]:.4f}，"
      f"z=({z[pos, 0]:+.3f}, {z[pos, 1]:+.3f})")
print(f"已存 {out}")
