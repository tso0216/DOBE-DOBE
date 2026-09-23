"""把 get_data.py 的統計輸出畫成數張獨立的圖，對應論文 3.1 節。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import CATEGORIES, MIN_POI, N_CAT  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'

out_dir = os.path.dirname(os.path.abspath(__file__))
summary = pd.read_csv(os.path.join(out_dir, 'summary.csv'), index_col='metric')['value']
cat = pd.read_csv(os.path.join(out_dir, 'category.csv'), index_col='category').reindex(CATEGORIES)
checkin = pd.read_csv(os.path.join(out_dir, 'checkin.csv'))['checkin_count'].values
cell = pd.read_csv(os.path.join(out_dir, 'cell.csv'))['poi_per_cell'].values
patch = pd.read_csv(os.path.join(out_dir, 'patch.csv'))


def num(k):
    """k：summary.csv 的 metric 名稱。回傳：該欄位的數值。"""
    return float(summary[k])


def save(fig, name):
    """fig：matplotlib figure。name：輸出檔名（相對本資料夾）。回傳：None。"""
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    print(name)


# ---- 3.1.1 各處理階段的資料量 ----
fig, ax = plt.subplots(figsize=(7.5, 4.5))
stages = ['Raw\ncheck-ins', 'Unique\nPOIs', 'Cleaned\nPOIs', 'Occupied\ncells',
          'Kept\npatches', 'POIs in\npatches']
vals = [num('raw_checkins'), num('raw_unique_poi'), num('clean_poi'),
        num('occupied_cells'), num('kept_patches'), num('poi_in_patches')]
bars = ax.bar(stages, vals,
              color=['#9e9e9e', '#9e9e9e', '#4363d8', '#f58231', '#e6194b', '#3cb44b'])
ax.set_yscale('log')
ax.set_ylim(500, vals[0] * 4)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.2, f'{int(v):,}', ha='center', fontsize=9)
ax.set_ylabel('Count (log scale)')
ax.set_title('Data volume at each preprocessing stage')
save(fig, '0.4.1_pipeline.png')

# ---- 3.1.2 每個 POI 的打卡次數：長尾 ----
fig, ax = plt.subplots(figsize=(6.5, 4.5))
s = np.sort(checkin)[::-1]
ax.loglog(np.arange(1, len(s) + 1), s, color='#4363d8', linewidth=1.6)
ax.axhline(checkin.mean(), color='#e6194b', linestyle='--', linewidth=1.1,
           label=f'mean = {checkin.mean():.1f}')
ax.axhline(np.median(checkin), color='#3cb44b', linestyle='--', linewidth=1.1,
           label=f'median = {np.median(checkin):.0f}')
ax.set_xlabel('POI rank (by check-in count)')
ax.set_ylabel('Check-in count')
ax.set_title(f'Check-ins per POI: heavy tail\n'
             f'(std = {checkin.std(ddof=1):.1f}, max = {s[0]:,.0f})')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')
save(fig, '0.4.2_checkin_ranksize.png')

# ---- 3.1.3 類別佔比：POI 數 vs 打卡數 ----
fig, ax = plt.subplots(figsize=(8, 5))
pos = np.arange(len(cat))[::-1]
h = 0.4
ax.barh(pos + h / 2, cat['poi_ratio'], height=h, color='#4363d8', label='Share of POIs')
ax.barh(pos - h / 2, cat['checkin_ratio'], height=h, color='#f58231', label='Share of check-ins')
for p, v in zip(pos, cat['poi_ratio']):
    ax.text(v + 0.6, p + h / 2, f'{v:.1f}%', va='center', fontsize=8)
for p, v in zip(pos, cat['checkin_ratio']):
    ax.text(v + 0.6, p - h / 2, f'{v:.1f}%', va='center', fontsize=8)
ax.set_yticks(pos)
ax.set_yticklabels(CATEGORIES, fontsize=9)
ax.set_xlim(0, 56)
ax.set_xlabel('Percentage (%)')
ax.set_title('Category share: POI count vs check-in count')
ax.legend(fontsize=9, loc='lower right')
save(fig, '0.4.3_category_share.png')

# ---- 3.1.4 100 m 格點的 POI 數與 MIN_POI 門檻 ----
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.arange(0.5, cell.max() + 1.5, 1)
ax.hist(cell, bins=bins, color='#bdbdbd', label=f'Discarded (POI < {MIN_POI})')
ax.hist(cell[cell >= MIN_POI], bins=bins, color='#e6194b', label='Kept as patch')
ax.axvline(MIN_POI, color='black', linestyle='--', linewidth=1.2)
ax.text(MIN_POI + 1.5, ax.get_ylim()[1] * 0.35, f'MIN_POI = {MIN_POI}', fontsize=9)
ax.set_yscale('log')
ax.set_xlabel('POIs per 100 m cell')
ax.set_ylabel('Number of cells (log scale)')
ax.set_title(f"POIs per occupied 100 m cell\n"
             f"({int(num('occupied_cells')):,} occupied -> "
             f"{int(num('kept_patches')):,} kept, {num('kept_cell_ratio'):.1%})")
ax.legend(fontsize=9)
save(fig, '0.4.4_cell_poi.png')

# ---- 3.1.5 每個 patch 的 POI 數 ----
fig, ax = plt.subplots(figsize=(7, 4.5))
n_poi = patch['n_poi'].values
ax.hist(n_poi, bins=np.arange(9.5, n_poi.max() + 1.5, 2), color='#3cb44b', edgecolor='white')
ax.axvline(n_poi.mean(), color='#e6194b', linestyle='--',
           label=f'mean = {n_poi.mean():.1f} $\\pm$ {n_poi.std(ddof=1):.1f}')
ax.axvline(np.median(n_poi), color='#4363d8', linestyle='--',
           label=f'median = {np.median(n_poi):.0f}')
ax.set_xlabel('POIs per patch')
ax.set_ylabel('Number of patches')
ax.set_title(f'POIs per patch (n = {len(n_poi):,})')
ax.legend(fontsize=9)
save(fig, '0.4.5_patch_poi.png')

# ---- 3.1.6 每個 patch 出現幾種類別 ----
fig, ax = plt.subplots(figsize=(7, 4.5))
n_cat = patch['n_cat'].values
ax.hist(n_cat, bins=np.arange(0.5, N_CAT + 1.5, 1), color='#911eb4', edgecolor='white')
ax.axvline(n_cat.mean(), color='#e6194b', linestyle='--',
           label=f'mean = {n_cat.mean():.2f} $\\pm$ {n_cat.std(ddof=1):.2f}')
ax.set_xticks(range(1, N_CAT + 1))
ax.set_xlabel('Number of distinct categories in a patch')
ax.set_ylabel('Number of patches')
ax.set_title(f'Functional mixing: categories per patch (max possible = {N_CAT})')
ax.legend(fontsize=9)
save(fig, '0.4.6_categories_per_patch.png')

# ---- 3.1.7 每個 patch 的 Shannon entropy ----
fig, ax = plt.subplots(figsize=(7, 4.5))
ent = patch['entropy'].values
ax.hist(ent, bins=40, color='#f58231', edgecolor='white')
ax.axvline(ent.mean(), color='#e6194b', linestyle='--',
           label=f'mean = {ent.mean():.2f} $\\pm$ {ent.std(ddof=1):.2f}')
ax.axvline(np.log(N_CAT), color='black', linestyle=':',
           label=f'$\\ln {N_CAT}$ = {np.log(N_CAT):.2f} (uniform)')
ax.set_xlabel('Shannon entropy of the category counts')
ax.set_ylabel('Number of patches')
ax.set_title('Functional mixing: entropy per patch')
ax.legend(fontsize=9)
save(fig, '0.4.7_entropy_per_patch.png')
