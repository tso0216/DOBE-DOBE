import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.dirname(script_dir)
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv('data/tky_clean.csv')

counts = df['category'].value_counts()
ratios = counts / counts.sum() * 100

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(counts.index[::-1], ratios[::-1])
ax.set_xlabel('percentage (%)')
ax.set_title(f'POI category percentage ')

for bar, ratio, count in zip(bars, ratios[::-1], counts[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f'{ratio:.1f}% ({count})', va='center', fontsize=9)

ax.set_xlim(0, ratios.max() * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '0_poi_ratio.png'), dpi=150)
