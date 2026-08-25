"""繪製 get_data.py 輸出的 POI 經緯度散點圖（依類別著色）"""
import math
import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

out_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(out_dir, 'data.csv'))

lon_range = df['lon'].max() - df['lon'].min()
lat_range = df['lat'].max() - df['lat'].min()
mean_lat_rad = math.radians(df['lat'].mean())
width = 9 * lon_range * math.cos(mean_lat_rad) / lat_range
fig, ax = plt.subplots(figsize=(max(width, 6), 9))
categories = df['category'].value_counts().index

for cat in categories:
    sub = df[df['category'] == cat]
    ax.scatter(sub['lon'], sub['lat'], s=1, alpha=0.3, label=cat)

ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_title('POI spatial distribution')
ax.set_aspect('equal')
ax.legend(markerscale=3, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '0.2_spatial.png'), dpi=150, bbox_inches='tight')
