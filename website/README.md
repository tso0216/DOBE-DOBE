# DOBE Latent & Geographic Workbench

lab.py 的網頁版。模型固定用 v3（ddae_fsce_tfidf 的 fold3_mae 參數），
所有相依檔案都在本資料夾內，不會去讀專案其他位置：

- `weights/fold3_mae.pt`：模型參數（從 `experiment/model/ddae_fsce_tfidf/` 複製）
- `data/patches.npz`：patch 資料（從 `data/patch/` 複製）
- `model_def.py`：AE 結構（與 `model/v3_ddae_tfidf/model.py` 相同）

## 執行

```bash
pip install -r requirements.txt
python app.py
```

開 http://127.0.0.1:5001

## 操作

- 左圖（地理）：藍點 = test patch，點一下選取；紅點 = train，鎖定只顯示不可選；紅框 = robust 距離離群
- 中圖（latent）：點一下換 patch；★ = 未加料原點，紅點 = 目前位置，紅線 = 加料軌跡
- 右圖（POI）：點一下加一顆目前類別的 POI；下方按鈕可 +50 隨機、Undo（一批）、Reset
- 頂部：S 分數（k=8 近鄰平均 latent 距離）與 Δ、robust 距離 / 離群門檻
