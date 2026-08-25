"""繪製 get_data.py 輸出的 POI 類別佔比長條圖"""
import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

out_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(out_dir, 'data.csv'), index_col='category')

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(df.index[::-1], df['ratio'][::-1])
ax.set_xlabel('percentage (%)')
ax.set_title('POI category percentage')

for bar, ratio, count in zip(bars, df['ratio'][::-1], df['count'][::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f'{ratio:.1f}% ({count})', va='center', fontsize=9)

ax.set_xlim(0, df['ratio'].max() * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '0.1_poi_raido.png'), dpi=150)
