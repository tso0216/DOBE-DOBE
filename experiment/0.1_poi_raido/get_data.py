"""計算 POI 各類別的數量與佔比"""
import os
import pandas as pd

CSV = os.path.join(os.path.dirname(__file__), "../../data/tky_clean.csv")

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(CSV)
counts = df['category'].value_counts()
ratio = (counts / counts.sum() * 100).rename('ratio')

pd.concat([counts.rename('count'), ratio], axis=1).to_csv(
    os.path.join(out_dir, 'data.csv'), index_label='category')
print(f"共 {len(df)} 筆 POI，{len(counts)} 個類別")
