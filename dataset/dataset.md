# 第三章　資料集

> 本檔是第三章的撰寫大綱。每一張圖各配一個獨立的程式碼區塊，圖號為「圖 3.x.y」，
> 可直接貼進 [main.ipynb](main.ipynb) 執行。所有程式碼只讀 `../data/` 底下的原始資料，
> 路徑一律相對，輸出的圖存到 `dataset/fig/`。

---

## 3.0　共用前置（先執行一次）

```python
# === 3.0 共用設定：路徑、常數、載入資料、投影 ===
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

NB_DIR = os.getcwd()                                       # dataset/
DATA = os.path.join(NB_DIR, '..', 'data', 'tky_clean.csv')  # 清理後的 POI 清單
RAW = os.path.join(NB_DIR, '..', 'data', 'tky_raw.txt')     # 原始打卡紀錄
FIG_DIR = os.path.join(NB_DIR, 'fig')
os.makedirs(FIG_DIR, exist_ok=True)

CRS = 'EPSG:6677'      # 日本平面直角座標系第 9 系，單位公尺
CELL = 100             # 格點邊長 s (m)
HALF = 50              # patch 視窗半寬 r (m)
MIN_POI = 10           # patch 保留門檻 tau
CATEGORIES = [
    'Dining and Drinking', 'Retail', 'Nightlife Spot',
    'Community and Government', 'Travel and Transportation',
    'Business and Professional Services', 'Landmarks and Outdoors',
    'Arts and Entertainment', 'Health and Medicine', 'Sports and Recreation',
]
COLORS = ['#e6194b', '#3cb44b', '#911eb4', '#4363d8', '#f58231',
          '#46f0f0', '#008080', '#f032e6', '#9a6324', '#808000']
K = len(CATEGORIES)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

df = pd.read_csv(DATA)
_tr = Transformer.from_crs('EPSG:4326', CRS, always_xy=True)
X, Y = _tr.transform(df['lon'].values, df['lat'].values)   # 投影後的公尺座標
CAT_ID = df['category'].map({c: i for i, c in enumerate(CATEGORIES)}).values


def save(fig, name):
    """fig：matplotlib figure。name：輸出檔名（存到 dataset/fig/）。回傳：None。"""
    fig.savefig(os.path.join(FIG_DIR, name))
    plt.show()
    print(os.path.join('fig', name))
```

---

## 3.1　Overview

**目的：交代資料從哪裡來、原始提供了哪些欄位、經過什麼處理後我們手上有什麼。**

- **3.1.1　資料來源**
  - Foursquare 東京（TKY）打卡紀錄，原始檔 `data/tky_raw.txt`。
  - 涵蓋 2012-04-03 ～ 2013-02-16，共 318 天。
  - 規模：573,703 筆打卡、2,293 位使用者、61,858 個唯一 POI。
  - 空間範圍：緯度 35.5102–35.8672、經度 139.4709–139.9126，投影後約 40.0 km × 39.6 km。

- **3.1.2　原始資料提供的欄位**
  | 欄位 | 內容 | 本研究使用 |
  |---|---|---|
  | `user_id` | 打卡使用者編號 | 否 |
  | `poi_id` | 地點唯一碼 | 是（去重的鍵） |
  | `category_id` / `category_name` | Foursquare 細類（371 類） | 是（映射成大類） |
  | `lat` / `lon` | WGS84 座標 | 是 |
  | `timezone offset` / `utc_time` | 時區與時間戳 | 否（本研究不做時序） |

- **3.1.3　前處理與我們實際使用的資料**
  1. 依 `poi_id` 去重，保留最晚一筆打卡的類別與座標，另計該地點的累計打卡次數。
  2. 用 Foursquare 類別階層表把細類 `A > B > C` 取第一層，得到 10 個大類。
  3. 剔除非固定地點：移動載具（Train、Taxi、Plane、Boat or Ferry、Moving Target、Bus Line）
     與線狀、面狀地物（Road、Bridge、River、Hiking Trail、Neighborhood、City）。
  4. 產出 `data/tky_clean.csv`：59,321 個 POI（剔除 2,537 個），欄位為
     `poi_id, category_id, category_name, category, lat, lon, checkin_count`。
  - 因此本研究可用的訊號只有三項：**經緯度、10 大類類別、累計打卡次數**。

- **3.1.4　各階段資料量（圖 3.1.1）**
  | 階段 | 數量 |
  |---|---|
  | 原始打卡 | 573,703 |
  | 唯一 POI | 61,858 |
  | 清理後 POI | 59,321 |
  | 有 POI 的 100 m 格點 | 19,158 |
  | 通過門檻、成為樣本的 patch | 1,233 |

### 圖 3.1.1　各處理階段的資料量

```python
# === 圖 3.1.1 各處理階段的資料量（log 縱軸） ===
n_raw, poi_seen = 0, set()
with open(RAW, encoding='utf-8') as f:
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) == 8:
            n_raw += 1
            poi_seen.add(p[1])

cell_id = np.floor(np.column_stack([X, Y]) / CELL).astype(np.int64)
_, cell_cnt = np.unique(cell_id, axis=0, return_counts=True)

stages = ['Raw\ncheck-ins', 'Unique\nPOIs', 'Cleaned\nPOIs',
          'Occupied\ncells', 'Kept\npatches']
vals = [n_raw, len(poi_seen), len(df), len(cell_cnt), int((cell_cnt >= MIN_POI).sum())]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
bars = ax.bar(stages, vals, color=['#9e9e9e', '#9e9e9e', '#4363d8', '#f58231', '#e6194b'])
ax.set_yscale('log')
ax.set_ylim(500, vals[0] * 4)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.2, f'{v:,}', ha='center', fontsize=9)
ax.set_ylabel('Count (log scale)')
ax.set_title('Data volume at each preprocessing stage')
save(fig, '3.1.1_pipeline.png')
```

---

## 3.2　Observation & Statistics

**目的：用統計與圖說明資料的分布長什麼樣子——類別怎麼分、空間怎麼分、每個單位裡有多少東西。**

- **3.2.1　類別佔比（圖 3.2.1）**
  - 10 大類極度不均：Dining and Drinking 佔 41.9%、Retail 21.3%，兩類合計逾六成；
    最少的 Sports and Recreation 僅 0.99%。
  - 同時比較「POI 數佔比」與「打卡數佔比」：Travel and Transportation 只佔 6.4% 的 POI
    卻吃下 46.7% 的打卡（平均每個 POI 68.5 次），顯示打卡量反映的是使用者取樣偏誤，
    而非該地點在都市中的機能份量。

- **3.2.2　空間分布（圖 3.2.2）**
  - 全體 POI 依類別著色的散點圖，看得出主要幹道與車站沿線的聚集，以及西側大片低密度區。

- **3.2.3　空間密度（圖 3.2.3）**
  - 投影到 EPSG:6677 後以 250 m 網格計數，log 色階。
  - 密度極端不均：最密的 1 km 見方區域有 1,402 個 POI，外圍多數格點為個位數或空白。
  - 呈現多中心結構（新宿、澀谷、池袋等次中心），不是單一核心向外遞減。

- **3.2.4　每個 POI 的打卡次數分布（圖 3.2.4）**
  - 長尾：mean 9.32、std 96.69、median 2、p90 12、p99 122、max 12,372。
  - std 遠大於 mean，說明此欄位不適合直接當作特徵。

- **3.2.5　每個 100 m 格點的 POI 數（圖 3.2.5）**
  - 19,158 個有 POI 的格點：mean 3.10、std 5.22、median 1、p90 7、max 85。
  - 超過一半的格點只有 1 個 POI，直接把每格當樣本會得到大量近乎空白的輸入，
    這是 3.3 設定 `MIN_POI = 10` 門檻的依據。

- **3.2.6　每個 patch 的內容量與機能混合度（圖 3.2.6）**
  - 通過門檻的 1,233 個 patch：POI 數 mean 18.98、std 10.49、median 15、max 85。
  - 類別數 mean 4.58、std 1.40、max 9；Shannon entropy mean 1.151、std 0.310，
    相對於均勻上界 ln 10 = 2.303 偏低。
  - 解讀：多數 patch 由單一機能主導，但仍混有 4–5 種類別，這正是本研究要學的結構。

- **3.2.7　敘述統計總表（表 3.2.1）**
  | 統計量 | n | mean | std | min | median | p90 | max |
  |---|---|---|---|---|---|---|---|
  | 每位使用者打卡數 | 2,293 | 250.20 | 221.57 | 100 | 173 | 475.8 | 2,991 |
  | 每個 POI 打卡數 | 59,321 | 9.32 | 96.69 | 1 | 2 | 12 | 12,372 |
  | 每個格點的 POI 數 | 19,158 | 3.10 | 5.22 | 1 | 1 | 7 | 85 |
  | 每個格點的 POI 數 | 1,233 | 18.98 | 10.49 | 10 | 15 | 32 | 85 |
  | 每個 patch 的類別數 | 1,233 | 4.58 | 1.40 | 1 | 4 | 6 | 9 |
  | 每個 patch 的 entropy | 1,233 | 1.151 | 0.310 | 0 | 1.171 | 1.529 | 1.888 |

### 圖 3.2.1　POI 類別佔比（POI 數 vs 打卡數）

```python
# === 圖 3.2.1 類別佔比 ===
g = df.groupby('category')
poi_ratio = (g.size() / len(df) * 100).reindex(CATEGORIES)
ck_ratio = (g['checkin_count'].sum() / df['checkin_count'].sum() * 100).reindex(CATEGORIES)

fig, ax = plt.subplots(figsize=(8, 5))
pos = np.arange(K)[::-1]
h = 0.4
ax.barh(pos + h / 2, poi_ratio, height=h, color='#4363d8', label='Share of POIs')
ax.barh(pos - h / 2, ck_ratio, height=h, color='#f58231', label='Share of check-ins')
for p, v in zip(pos, poi_ratio):
    ax.text(v + 0.6, p + h / 2, f'{v:.1f}%', va='center', fontsize=8)
for p, v in zip(pos, ck_ratio):
    ax.text(v + 0.6, p - h / 2, f'{v:.1f}%', va='center', fontsize=8)
ax.set_yticks(pos)
ax.set_yticklabels(CATEGORIES, fontsize=9)
ax.set_xlim(0, 56)
ax.set_xlabel('Percentage (%)')
ax.set_title('POI category share: number of POIs vs number of check-ins')
ax.legend(fontsize=9, loc='lower right')
save(fig, '3.2.1_category_share.png')
```

### 圖 3.2.2　POI 空間分布（依類別著色）

```python
# === 圖 3.2.2 空間分布散點 ===
fig, ax = plt.subplots(figsize=(7.5, 7))
for i, (cat, color) in enumerate(zip(CATEGORIES, COLORS)):
    m = CAT_ID == i
    ax.scatter(df['lon'].values[m], df['lat'].values[m], s=1.2, alpha=0.5,
               color=color, label=cat, rasterized=True)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_aspect(1 / np.cos(np.deg2rad(df['lat'].mean())))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3,
          fontsize=8, markerscale=8, frameon=False)
ax.set_title(f'Spatial distribution of {len(df):,} POIs')
save(fig, '3.2.2_spatial_scatter.png')
```

### 圖 3.2.3　POI 密度熱圖（250 m 網格）

```python
# === 圖 3.2.3 密度熱圖 ===
from matplotlib.colors import LogNorm

D = 250
gx = np.floor(X / D).astype(np.int64)
gy = np.floor(Y / D).astype(np.int64)
img = np.zeros((gy.max() - gy.min() + 1, gx.max() - gx.min() + 1))
np.add.at(img, (gy - gy.min(), gx - gx.min()), 1)
img[img == 0] = np.nan

fig, ax = plt.subplots(figsize=(7.5, 7.5))
extent = [gx.min() * D / 1000, (gx.max() + 1) * D / 1000,
          gy.min() * D / 1000, (gy.max() + 1) * D / 1000]
im = ax.imshow(img, origin='lower', extent=extent, cmap='magma',
               norm=LogNorm(vmin=1, vmax=np.nanmax(img)), interpolation='nearest')
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
             label=f'POIs per {D} m cell (log scale)')
ax.set_xlabel(f'{CRS} X (km)')
ax.set_ylabel(f'{CRS} Y (km)')
ax.set_aspect('equal')
ax.set_title(f'POI density over Tokyo (max {int(np.nanmax(img))} per {D} m cell; '
             f'white = no POI)')
save(fig, '3.2.3_density_map.png')
```

### 圖 3.2.4　每個 POI 的打卡次數分布

```python
# === 圖 3.2.4 打卡次數分布（rank-size，log-log） ===
ck = df['checkin_count'].values
s = np.sort(ck)[::-1]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(np.arange(1, len(s) + 1), s, color='#4363d8', linewidth=1.6)
ax.axhline(ck.mean(), color='#e6194b', linestyle='--', linewidth=1.1,
           label=f'mean = {ck.mean():.2f}')
ax.axhline(np.median(ck), color='#3cb44b', linestyle='--', linewidth=1.1,
           label=f'median = {np.median(ck):.0f}')
ax.set_xlabel('POI rank (by check-in count)')
ax.set_ylabel('Check-in count')
ax.set_title(f'Check-ins per POI: heavy tail '
             f'(std = {ck.std(ddof=1):.2f}, max = {s[0]:,})')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')
save(fig, '3.2.4_checkin_dist.png')
```

### 圖 3.2.5　每個 100 m 格點的 POI 數分布

```python
# === 圖 3.2.5 每格 POI 數 ===
cell_id = np.floor(np.column_stack([X, Y]) / CELL).astype(np.int64)
_, cnt = np.unique(cell_id, axis=0, return_counts=True)

fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.arange(0.5, cnt.max() + 1.5, 1)
ax.hist(cnt, bins=bins, color='#bdbdbd', label=f'POI < {MIN_POI} (discarded)')
ax.hist(cnt[cnt >= MIN_POI], bins=bins, color='#e6194b',
        label=f'POI $\\geq$ {MIN_POI} (kept)')
ax.axvline(MIN_POI, color='black', linestyle='--', linewidth=1.2)
ax.set_yscale('log')
ax.set_xlabel(f'POIs per {CELL} m cell')
ax.set_ylabel('Number of cells (log scale)')
ax.set_title(f'POIs per occupied {CELL} m cell '
             f'(n = {len(cnt):,}, mean = {cnt.mean():.2f} $\\pm$ {cnt.std(ddof=1):.2f}, '
             f'median = {np.median(cnt):.0f})')
ax.legend(fontsize=9)
save(fig, '3.2.5_cell_poi.png')
```

### 圖 3.2.6　每個 patch 的 POI 數、類別數與 entropy

```python
# === 圖 3.2.6 patch 層級的三個分布 ===
cell_id = np.floor(np.column_stack([X, Y]) / CELL).astype(np.int64)
cells, inv, cnt = np.unique(cell_id, axis=0, return_inverse=True, return_counts=True)
kept = cnt >= MIN_POI

M = np.zeros((len(cells), K))
np.add.at(M, (inv, CAT_ID), 1)
M = M[kept]                                   # 每個 patch 的 10 維類別計數
n_poi = M.sum(axis=1)
n_cat = (M > 0).sum(axis=1)
p = M / n_poi[:, None]
ent = -(p * np.log(p, where=p > 0, out=np.zeros_like(p))).sum(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, v, color, xlabel in zip(
        axes, [n_poi, n_cat, ent], ['#3cb44b', '#911eb4', '#f58231'],
        ['POIs per patch', 'Distinct categories per patch', 'Shannon entropy per patch']):
    bins = np.arange(0.5, K + 1.5, 1) if xlabel.startswith('Distinct') else 30
    ax.hist(v, bins=bins, color=color, edgecolor='white')
    ax.axvline(v.mean(), color='#e6194b', linestyle='--',
               label=f'mean = {v.mean():.2f} $\\pm$ {v.std(ddof=1):.2f}')
    ax.axvline(np.median(v), color='#4363d8', linestyle=':',
               label=f'median = {np.median(v):.2f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of patches')
    ax.legend(fontsize=8)
axes[2].axvline(np.log(K), color='black', linestyle=':',
                label=f'$\\ln {K}$ = {np.log(K):.2f}')
axes[2].legend(fontsize=8)
fig.suptitle(f'Patch-level statistics (n = {len(M):,})', y=1.02)
save(fig, '3.2.6_patch_stats.png')
```

---

## 3.3　Format Definition

**目的：定義 patch——模型實際吃進去的那個單位——怎麼框出來、用什麼格式表示。**

- **3.3.1　patch 的定義**
  - 座標一律先投影到 EPSG:6677（單位公尺）再計算。
  - patch 的中心取自邊長 $s=100$ m 的規則格點：格索引 $g\in\mathbb{Z}^2$，格心 $u_g=(g+\tfrac{1}{2})s$。
  - 以 Chebyshev（$L_\infty$）距離、半寬 $r=50$ m 框出正方形視窗：
    $$\mathcal{N}(g)=\{\,p\in P:\ \lVert(x_p,y_p)-u_g\rVert_\infty\le r\,\}$$
  - 保留門檻 $\tau=10$：視窗內 POI 數不足 $\tau$ 的中心直接丟棄，
    $G^{*}=\{g:|\mathcal{N}(g)|\ge\tau\}$，最終得到 $|G^{*}|=1{,}233$ 個 patch。
  - 由於 $2r=s$，視窗恰好等於格子本身，patch 之間**不重疊、無重複計數**，
    共涵蓋 23,403 個 POI（全體的 39.5%）。

- **3.3.2　patch 的表示法**
  - 保留幾何的稀疏點列表（存於 `data/patch/patches.npz`）：
    $$S_g=\{(\boldsymbol{\delta}_i,c_i)\ :\ p_i\in\mathcal{N}(g)\},\qquad \boldsymbol{\delta}_i=(x_i,y_i)-u_g\in[-r,r]^2$$
  - 存相對位移而非固定尺寸矩陣，是為了保留後續做旋轉等幾何增強的空間。
  - 模型輸入則丟掉幾何、只留類別計數向量：
    $$\mathbf{x}_g\in\mathbb{N}^{K},\qquad \mathbf{x}_g[k]=|\{p\in\mathcal{N}(g):c_p=k\}|,\qquad \sum_{k}\mathbf{x}_g[k]=|\mathcal{N}(g)|$$
  - 全體樣本堆成資料矩陣 $X\in\mathbb{N}^{1233\times 10}$。
  - 由 $\mathbf{x}_g$ 導出的兩個描述量（3.2.6 使用）：
    $$\hat{p}_k=\frac{\mathbf{x}_g[k]}{\sum_j \mathbf{x}_g[j]},\qquad H_g=-\sum_k \hat{p}_k\log\hat{p}_k,\qquad D_g=\max_k \hat{p}_k$$

### 圖 3.3.1　一個 patch：Chebyshev 視窗與相對格心的座標

```python
# === 圖 3.3.1 單一 patch 的視窗與相對座標 ===
from matplotlib.patches import Rectangle

cell_id = np.floor(np.column_stack([X, Y]) / CELL).astype(np.int64)
cells, inv, cnt = np.unique(cell_id, axis=0, return_inverse=True, return_counts=True)
kept = cnt >= MIN_POI

# 範例格依規則挑：八個鄰格都有 POI、類別數不低於中位數、POI 數最接近中位數，
# 確保挑到的是典型格而非特例
occupied = {tuple(c) for c in cells}
nb8 = np.array([sum(((c[0] + i, c[1] + j) in occupied)
                    for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
                for c in cells])
onehot = np.zeros((len(cells), K), dtype=bool)
onehot[inv, CAT_ID] = True
n_cat = onehot.sum(axis=1)
ok = kept & (nb8 == 8) & (n_cat >= np.median(n_cat[kept]))
cand = np.flatnonzero(ok if ok.any() else kept)
pick = cand[np.argmin(np.abs(cnt[cand] - np.median(cnt[kept])))]
center = (cells[pick] + 0.5) * CELL                    # 格心 u_g

inside = (np.abs(X - center[0]) <= HALF) & (np.abs(Y - center[1]) <= HALF)
dx, dy = X[inside] - center[0], Y[inside] - center[1]  # 相對位移 delta

fig, ax = plt.subplots(figsize=(6, 6))
for i, (cat, color) in enumerate(zip(CATEGORIES, COLORS)):
    mm = CAT_ID[inside] == i
    ax.scatter(dx[mm], dy[mm], s=110, alpha=0.9, color=color, zorder=3, label=cat)
ax.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False,
                       edgecolor='#77021c', linewidth=3, zorder=2))
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
ax.scatter([0], [0], marker='+', s=220, color='#77021c', linewidth=2.5, zorder=4)
ax.set_xlim(-HALF * 1.2, HALF * 1.2)
ax.set_ylim(-HALF * 1.2, HALF * 1.2)
ax.set_aspect('equal')
ax.set_xlabel('$\\Delta x$ from the cell centre (m)')
ax.set_ylabel('$\\Delta y$ from the cell centre (m)')
ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
ax.set_title(f'One patch: Chebyshev radius $r$ = {HALF} m '
             f'({inside.sum()} POIs, threshold $\\tau$ = {MIN_POI})')
save(fig, '3.3.1_patch_window.png')
```

### 圖 3.3.2　丟掉幾何，聚合成 10 維類別計數向量

```python
# === 圖 3.3.2 類別計數向量（需先跑圖 3.3.1 決定 inside） ===
vec = np.bincount(CAT_ID[inside], minlength=K)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
pos = np.arange(K)[::-1]
ax.barh(pos, vec, color=COLORS, height=0.72)
for p, n in zip(pos, vec):
    ax.text(n + 0.12, p, str(int(n)), va='center', fontsize=10)
ax.set_yticks(pos)
ax.set_yticklabels(CATEGORIES, fontsize=9)
ax.set_xlim(0, max(vec.max() * 1.25, 1))
ax.set_xlabel('POI count')
ax.set_title(f'Drop the geometry, keep the {K}-dim count vector\n'
             f'$\\mathbf{{x}}_g$ = {vec.tolist()}')
save(fig, '3.3.2_count_vector.png')
```

---

## 圖表索引

| 編號 | 內容 |
|---|---|
| 圖 3.1.1 | 各處理階段的資料量 |
| 圖 3.2.1 | POI 類別佔比（POI 數 vs 打卡數） |
| 圖 3.2.2 | POI 空間分布散點 |
| 圖 3.2.3 | POI 密度熱圖（250 m 網格） |
| 圖 3.2.4 | 每個 POI 的打卡次數分布 |
| 圖 3.2.5 | 每個 100 m 格點的 POI 數分布 |
| 圖 3.2.6 | 每個 patch 的 POI 數、類別數、entropy |
| 表 3.2.1 | 敘述統計總表 |
| 圖 3.3.1 | 一個 patch 的 Chebyshev 視窗與相對座標 |
| 圖 3.3.2 | 10 維類別計數向量 |
