"""讀取原始 POI 資料的經緯度與類別，供 draw.py 畫空間分佈散點圖"""
import os
import pandas as pd

CSV = os.path.join(os.path.dirname(__file__), "../../data/tky_clean.csv")

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(CSV)[["lon", "lat", "category"]]
df.to_csv(os.path.join(out_dir, "data.csv"), index=False)
print(f"共 {len(df)} 筆 POI")
