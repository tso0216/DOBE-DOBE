# 實驗設計與參數設定

## 模型選擇

本研究採用 **v3_ddae_tfidf** 作為目標模型：一個以 Poisson 分布作為解碼目標的
Denoising Autoencoder（DAE），並以 TF-IDF 加權的 FSCE（fuzzy-simplicial-set
cross-entropy）相似度圖作為正則化項，搭配 PCGrad 梯度手術來平衡重建與圖正則化
兩個目標之間的衝突。

## 資料前處理（Patch 建構）

- **資料來源**：東京 Foursquare 打卡紀錄，每筆資料含 POI 座標與其所屬類別，共
  $C=10$ 類（如 Dining and Drinking、Retail、Nightlife Spot、Travel and
  Transportation、Health and Medicine、Sports and Recreation 等）。
- **座標投影**：將原始 WGS84 座標投影至日本平面直角座標系第 9 系
  （EPSG:6677，單位為公尺）。
- **格點中心生成**：以間距 $s=100$ m 的規則格網切分投影平面，僅保留「格內至少
  含一個 POI」的格點作為候選 patch 中心。
- **鄰域收集**：以 KD-tree 對每個候選中心搜尋 Chebyshev 距離（$L_\infty$，正方
  形視窗）半徑 $r=50$ m 內的 POI。
- **密度篩選**：鄰域內 POI 數少於 $n_{\min}=10$ 的候選中心予以丟棄，避免產生統
  計上不具代表性的 patch。
- **特徵聚合**：每個保留下來的 patch 統計各類別出現次數，得到計數向量
  $x_i\in\mathbb{Z}_{\ge0}^{10}$，此步驟捨棄空間形狀資訊，只保留機能組成。

## 特徵與圖建構

- **TF-IDF 轉換**：對類別計數矩陣做平滑化 TF-IDF 轉換，降低高重疊類別（如
  Dining）的權重，並提升具區辨性類別（如 Sports and Recreation、Health and
  Medicine）的權重。
- **FSCE 相似度圖**：在 TF-IDF 表徵上，以 $k=10$ 近鄰、cosine 距離建圖，並用
  UMAP 的 fuzzy simplicial set 求出邊權重 $w_{ij}\in[0,1]$ 與低維核函數參數
  $a,b$（由 spread $=1.0$、min\_dist $=0.1$ 推得）；此圖每個訓練 fold 只用該
  fold 的訓練 patch 重新建構。
- **僅供視覺化的分群**：在 TF-IDF 空間上做 $K=8$ 的 K-Means 分群，僅用於潛在
  空間視覺化上色，不參與任何 loss 計算。

## 模型架構

- **Encoder / Decoder**：皆為 4 層 Linear–LayerNorm–GELU 區塊，隱藏維度
  $h=64$，連接 $C=10$ 維輸入空間與 $d=2$ 維潛在空間；解碼器輸出
  $\log\hat\lambda$，將每個 patch 的類別計數視為 Poisson 分布建模。
- **Denoising 破壞**：Encoder 輸入以機率 $p=0.3$ 做 thinning 破壞（每個計數單
  位獨立以機率 $p$ 被丟棄，再除以 $1/(1-p)$ 做無偏縮放），而重建目標維持為未
  破壞的原始計數向量。

## 訓練與最佳化

- **最佳化器**：Adam，weight decay $1\times10^{-6}$，batch size $B=256$，訓練
  $T=2000$ 個 epoch，隨機種子固定為 $0$。
- **學習率排程**：cosine annealing，從 $\eta_{\max}=1\times10^{-2}$ 降到
  $\eta_{\min}=1\times10^{-3}$。
- **損失函數**：Poisson 負對數似然重建項，加上 FSCE 正則化項，權重為
  $\lambda_t=\lambda_{\max}\cdot\min(1, t/T_{\text{warmup}})$，其中
  $\lambda_{\max}=0.25$、$T_{\text{warmup}}=200$ epoch；FSCE 邊以
  $B_e=256$ 的 mini-batch 抽樣。
- **梯度手術**：當重建梯度與 FSCE 梯度方向衝突（內積為負）時，套用 PCGrad
  把 FSCE 梯度中與重建梯度衝突的分量投影掉，優先保留重建方向。

## 評估流程

- **資料切分**：採巢狀切分——先切出獨立測試集（$\text{test\_frac}=20\%$），
  不參與任何 fold 的訓練或驗證；剩餘資料依 $N_{\text{splits}}=5$ 做 k-fold
  交叉驗證（train/val 輪替）。
- **模型選擇**：每個 fold 內獨立追蹤四份 checkpoint，分別對應驗證集 MAE、
  MSE、WAPE、Poisson deviance 最低的 epoch（評估時輸入為未破壞的原始資料）。
- **評估指標公式**：四個指標皆先在單一 patch 的 $C=10$ 個類別上計算，再對整批
  patch 取平均，其中 $x_c$ 為真實類別計數，$\hat\lambda_c=\exp(\log\hat\lambda_c)$
  為解碼器輸出的 Poisson 率參數（$x_c\ln(x_c/\hat\lambda_c)$ 在 $x_c=0$ 時定義
  為 $0$）：
  - $\text{MAE}=\frac{1}{C}\sum_{c=1}^{C}|x_c-\hat\lambda_c|$
  - $\text{MSE}=\frac{1}{C}\sum_{c=1}^{C}(x_c-\hat\lambda_c)^2$
  - $\text{WAPE}=\frac{\sum_{c=1}^{C}|x_c-\hat\lambda_c|}{\sum_{c=1}^{C}x_c}$
  - $\text{Deviance}=\frac{2}{C}\sum_{c=1}^{C}(x_c\ln(x_c/\hat\lambda_c)-(x_c-\hat\lambda_c))$
- **報告的測試**：重建品質透過單一 patch 的真實計數 vs. 重建計數長條圖檢視，
  學到的表徵則透過潛在座標 $z_i$ 的 2D 散點圖檢視；量化的測試集結果留待後續
  章節報告。

> **提醒**：以上 thinning 機率 $p=0.3$ 是 `cfg.py` 中 denoising 的設計值。請
> 確認這是你要報告的設定，因為目前 `result/result.log` 中最近一次訓練實際上
> 是以 `NOISE_P=0` 執行的。
