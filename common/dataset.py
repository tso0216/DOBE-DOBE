import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# fsq（Foursquare 打卡紀錄清乾淨後的唯一地點清單，類別是 FSQ 的最大類）固定參數
CSV = os.path.join(ROOT, "data", "tky_clean.csv")
CAT_COL = "category"
CATEGORIES = [
    "Dining and Drinking",
    "Retail",
    "Nightlife Spot",
    "Community and Government",
    "Travel and Transportation",
    "Business and Professional Services",
    "Landmarks and Outdoors",
    "Arts and Entertainment",
    "Health and Medicine",
    "Sports and Recreation",
]
CAT_ZH = [
    "餐飲", "零售", "夜生活", "社區/政府", "交通",
    "商業服務", "地標/戶外", "藝文娛樂", "醫療", "運動休閒",
]

HALF_WIDTH = 50.0
CELL = 15
CENTER_STEP = 100      # patch 中心的格點間距(公尺)
MIN_POI = 10            # 圓內少於這個數量的中心直接丟掉

GRID = int(HALF_WIDTH * 2 / CELL)
CRS = "EPSG:6677"       # 日本平面直角座標系第9系，涵蓋東京都，單位公尺

PALETTE = [
    "#e6194b", "#3cb44b", "#911eb4", "#4363d8", "#f58231",
    "#46f0f0", "#008080", "#f032e6", "#9a6324", "#808000",
]

N_CAT = len(CATEGORIES)
CAT_COLORS = PALETTE[:N_CAT]

PATCHES = os.path.join(ROOT, "data", "patch", "patches.npz")
FEATURES = os.path.join(ROOT, "data", "patch", "features.npz")
FEATURES_CSV = os.path.join(ROOT, "data", "patch", "features.csv")

# 記錄進 result.log 的 dataset 區塊，方便回頭比較不同次訓練用的參數
PATCH_PARAMS = {
    "csv": CSV,
    "cat_col": CAT_COL,
    "categories": CATEGORIES,
    "half_width": HALF_WIDTH,
    "cell": CELL,
    "center_step": CENTER_STEP,
    "min_poi": MIN_POI,
    "crs": CRS,
}


def result(version, name):
    """某個模型版本的輸出：model/<version>/result/<name>。"""
    d = os.path.join(ROOT, "model", version, "result")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)
