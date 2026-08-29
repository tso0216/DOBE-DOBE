V3（v3_ddae_tfidf）Method 流程（章節重整版：Offline / Online）

# 0. 符號定義

- $\mathcal{C}=\{1,\dots,C\}$：POI 類別集合，$C=10$（N_CAT）

- $i$：patch 索引；$x_i\in\mathbb{Z}_{\ge0}^{C}$：patch $i$ 的類別計數向量，$x_{i,c}$ 為第 $c$ 類 POI 在 patch 內出現次數

- $z_i\in\mathbb{R}^{d}$：patch $i$ 的潛在座標，$d=2$（LATENT_DIM）

- $\hat\lambda_i\in\mathbb{R}_{>0}^{C}$：解碼器輸出的 Poisson 率參數（重建目標）


# 1. Offline 流程

Offline 流程涵蓋從原始打卡資料到訓練出模型、並在固定的 test 集上完成 rebuild／latent 兩個評估測試為止，全部只依賴訓練時就已固定的資料集，不處理新進資料。

## 1.1 資料建構（Patch Construction）

- 資料來源：Foursquare 東京打卡紀錄（tky_clean.csv），每筆資料為一筆 POI 的經緯度與所屬類別，類別已收斂為 $C=10$ 類。

- 座標投影：將原始經緯度（WGS84）投影至日本平面直角座標系第9系（EPSG:6677），單位為公尺，以利後續以「公尺」為單位計算距離。

- 格點中心生成：以邊長 $s=100\text{m}$ 將投影平面切成規則格網，僅保留「格內至少含一個 POI」的格點作為候選 patch 中心，避免在完全空曠處產生無意義的 patch：
$$p_i = \left( \left\lfloor \frac{x}{s} \right\rfloor + \frac{1}{2} \right) s$$

- 鄰域收集：以 Chebyshev 距離（$L_\infty$，即正方形視窗而非圓形）、半徑 $r=50\text{m}$，透過 KD-tree 搜尋每個候選中心 $p_i$ 周圍的 POI 集合：
$$N_i = \{\, q : \lVert q - p_i \rVert_\infty < r \,\}$$

- 密度篩選：僅保留 $|N_i| \ge n_{\min}$ 的中心作為正式 patch，過濾掉 POI 過於稀疏、統計上不具代表性的區域。

- 輸出格式：每個 patch 儲存其內部 POI 的類別 id 與相對於中心的位移 $(dx, dy)$（公尺），以 CSR 稀疏格式（offsets 陣列）壓縮儲存，避免存成大量空白的固定尺寸網格。


## 1.2 特徵聚合（Category-Count Aggregation）

- 對每個 patch，統計各類別出現次數，得到計數向量 $x_i$：
$$x_{i,c}=\sum_{q\in N_i}\mathbb{1}[\text{cat}(q)=c],\quad c=1,\dots,C$$

- 這一步捨棄空間位置，只留類別分布。模型本質是對「patch 的機能組成」建模，不是對空間形狀建模。


## 1.3 TF-IDF 轉換（用於分群與圖建構，不是模型輸入）

- 對全體 patch 的計數矩陣 $X\in\mathbb{Z}^{N\times C}$ 做 TF-IDF（sklearn.TfidfTransformer，smooth_idf=True, L2 normalize）：
$$\text{idf}_c=\ln\frac{1+N}{1+\text{df}_c}+1,\qquad \tilde x_{i,c}=\frac{x_{i,c}\cdot\text{idf}_c}{\lVert x_{i,\cdot}\odot\text{idf}\rVert_2}$$

- 用途 (a)：對 $\tilde x_i$ 做 K-Means 分群，得到分群標籤 $c_i$，僅用於訓練過程中潛在空間視覺化上色，不參與任何 loss：
$$\{c_i\}_{i=1}^{N} = \text{KMeans}_K\big(\{\tilde x_i\}_{i=1}^{N}\big)$$

- 用途 (b)：作為建構下一步 FSCE 圖的距離依據。


## 1.4 高維相似度圖建構（FSCE Graph，UMAP 式模糊單純複形）

- 在 TF-IDF 空間以 cosine 距離找 $k=10$（N_NEIGHBORS）近鄰，用 UMAP 的 fuzzy_simplicial_set 建圖，得到邊集合 $(i,j)$ 與邊權重 $w_{ij}\in[0,1]$（高維相似度）。

- 同時求得核函數參數 $(a,b)$（find_ab_params(spread=1.0, min_dist=0.1)），供 1.7 節損失中低維相似度核使用。

- 這張圖只在訓練集內的 patch 上建，訓練時每個 mini-batch 隨機抽 EDGE_BATCH 條正邊 $(i,j,w_{ij})$，另外均勻抽等量負邊（隨機配對、權重設為 0）作為對照。


## 1.5 破壞過程（Corruption / Denoising 輸入）

- 對計數向量做「thinning」破壞（NOISE_P=0.3）：每個計數單位獨立以機率 $\text{keep}=1-p$ 保留，再除以 $\text{keep}$ 做無偏縮放：
$$\tilde x_{i,c}=\frac{1}{1-p}\sum_{u=1}^{x_{i,c}} \text{Bernoulli}(1-p)_u,\qquad \mathbb{E}[\tilde x_{i,c}]=x_{i,c}$$

- 另有 mask 模式：整個特徵以機率 $p$ 全歸零再除以 keep，程式碼裡可切換但預設用 thinning。

- 編碼器吃的是 $\tilde x_i$（破壞後），reconstruction 目標是原始 $x_i$，這是模型「denoising」名稱的來源。


## 1.6 模型架構（Denoising Autoencoder + Poisson 解碼）

- Encoder $f_\theta:\mathbb{R}^C\to\mathbb{R}^d$：4 層 Linear→LayerNorm→GELU（隱藏維度 $h=64$）+ 最終線性層到 $d=2$。

- Decoder $g_\theta:\mathbb{R}^d\to\mathbb{R}^C$：對稱結構，輸出 $\log\hat\lambda_i=g_\theta(z_i)$（輸出的是 log 率參數，不是直接的計數重建）：
$$z_i=f_\theta(\tilde x_i),\qquad \log\hat\lambda_i=g_\theta(z_i)$$


## 1.7 損失函數

- (a) 重建損失：Poisson 負對數似然（把計數當 Poisson 分布建模，捨去與參數無關的 $\log(x_{i,c}!)$ 常數項）：
$$\mathcal{L}_{\text{recon}}=\frac1C\sum_{c=1}^{C}\left(\hat\lambda_{i,c}-x_{i,c}\log\hat\lambda_{i,c}\right)$$

- (b) FSCE 損失（UMAP 交叉熵形式，把高維邊權重 $w_{ij}$ 當標籤，低維用 Student-t 型核 $q$ 逼近）：
$$q_{ij}=\left(1+a\,\lVert z_i-z_j\rVert_2^{2b}\right)^{-1}$$
$$\mathcal{L}_{\text{fsce}}=-\big[w_{ij}\log q_{ij}+(1-w_{ij})\log(1-q_{ij})\big]$$

- (c) 總損失與 warm-up 權重：
$$\lambda_t=\lambda_{\max}\cdot\min\left(1,\frac{t}{T_{\text{warmup}}}\right),\qquad \mathcal{L}=\mathcal{L}_{\text{recon}}+\lambda_t\,\mathcal{L}_{\text{fsce}}$$
（$\lambda_{\max}=0.25$，$T_{\text{warmup}}=200$ epoch）


## 1.8 多目標梯度手術（PCGrad）

- 對 $\mathcal{L}_{\text{recon}}$ 與 $\lambda_t\mathcal{L}_{\text{fsce}}$ 分別對參數 $\theta$ 取梯度 $g_r,g_f$。

- 若兩者方向衝突（內積為負），把 $g_f$ 投影掉與 $g_r$ 衝突的分量，只保留正交分量；$g_r$ 本身不變（這裡是非對稱版本，優先保護重建梯度）：
$$\text{若 } g_r\cdot g_f<0:\quad g_f' = g_f-\frac{g_r\cdot g_f}{\lVert g_r\rVert^2}g_r,\qquad g_\theta=g_r+g_f'$$
$$\text{否則:}\quad g_\theta=g_r+g_f$$


## 1.9 訓練流程

- 資料切分：make_split，隨機切分（非空間分群），依比例切成 train/val/test：
$$\text{val} : \text{test} = \text{val\_frac} : \text{test\_frac},\qquad \text{train} = 1-\text{val\_frac}-\text{test\_frac}$$

- 最佳化：Adam，weight decay $\gamma$，訓練 $T$ 個 epoch，batch size $B$，學習率採 cosine annealing，從 $\eta_{\max}$ 降到 $\eta_{\min}$：
$$\eta_t=\eta_{\min}+\frac12(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)$$

- 模型選擇：每 epoch 用驗證集算 METRIC（預設 MAE，見 1.10 節），取驗證誤差最低的 epoch 存 checkpoint（評估時輸入未加破壞噪聲，即用原始 $x_i$ 直接編碼）。

- 最終在從未參與訓練或選模的 test 集上報一次數值，作為論文的主要指標。


## 1.10 推論與評估（對應 CLAUDE.md 要求的 rebuild + latent plot 兩個測試）

- Reconstruction 測試（analyze/rebuild.py）：挑單一 patch，畫真實計數 vs. 重建 $\hat\lambda$ 的類別長條圖，並報該 patch 誤差在全體中的百分位。

- Latent 測試（analyze/latent_plot_plain.py）：把全體 patch 的 $z_i$ 投影到 2D 平面畫散點圖。

- 誤差指標可選（METRICS 字典）：
$$\text{MAE}=\frac1C\sum_c|x_c-\hat\lambda_c|,\quad \text{MSE}=\frac1C\sum_c(x_c-\hat\lambda_c)^2,\quad \text{WAPE}=\frac{\sum_c|x_c-\hat\lambda_c|}{\sum_c x_c}$$


# 2. Online 流程

Online 流程建立在 offline 訓練完成、encoder $f_\theta$ 已凍結的前提上：使用者可任意框選一塊查詢區域並即時取得異常分數，也可即時編輯（新增 POI）觀察分數變化，全程不需重新訓練模型、也不限於訓練時用過的規則網格 patch。

## 2.1 使用者查詢定義（User Query）

- 使用者在地圖上框選一塊查詢區域 $A\subset\mathbb{R}^2$（不必是 1.1 節的規則格點 patch，可為任意框選範圍）。

- 落在 $A$ 內的 POI 集合與其類別計數向量，定義方式與 1.2 節相同（只是中心/範圍改由使用者即時指定）：
$$N_A=\{\,q : q\in A\,\},\qquad x_{A,c}=\sum_{q\in N_A}\mathbb{1}[\text{cat}(q)=c],\quad c=1,\dots,C$$

- $x_A\in\mathbb{Z}_{\ge0}^{C}$ 即對應圖例中的 input vector（例如 $[8,12,1,2,8,1,\dots]$）。


## 2.2 Encode 與初始分數

- 推論時不做 1.5 節的破壞（thinning/mask），直接用原始計數向量餵給 encoder：
$$z_A=f_\theta(x_A)$$

- 分數定義（leave-one-out kNN 平均 latent 距離）：
$$S_A=\frac{1}{k_S}\sum_{j\in\mathcal N_{k_S}(A)}\lVert z_A-z_j\rVert_2,\qquad k_S=8$$
其中 $\{z_j\}_{j=1}^{N}$ 為參考集合 $R$（訓練時算好的 test 集 latent 座標），$\mathcal N_{k_S}(A)$ 為 $R$ 中離 $z_A$ 最近的 $k_S$ 個；若 $A$ 本身就是 $R$ 內既有的 patch（如 2.4 節案例研究直接取用 test 集 patch），則排除自己再取 $k_S$ 個（leave-one-out）。$S_A$ 大＝離群、小＝典型；$S$ 未正規化、依 latent 尺度而定。


## 2.3 使用者編輯（新增 POI）與分數更新

- 使用者於查詢區域內新增 POI，以類別計數增量向量表示：
$$\delta\in\mathbb{Z}_{\ge0}^{C},\qquad \delta_c=\text{使用者新增的第 }c\text{ 類 POI 數量}$$
對應圖例中的增量向量（例如 $[+3,+0,+4,+0,+0,+1,\dots]$）。

- 更新後的輸入向量與新 latent：
$$x_A'=x_A+\delta,\qquad z_A'=f_\theta(x_A')$$
（圖例中 $[8,12,1,2,8,1,\dots]+[+3,+0,+4,+0,+0,+1,\dots]=[11,12,5,2,8,2,\dots]$）

- 新分數與分數變化量：
$$S_A'=\frac{1}{k_S}\sum_{j\in\mathcal N_{k_S}(A')}\lVert z_A'-z_j\rVert_2,\qquad \Delta S=S_A'-S_A$$

- 軌跡視覺化：把 $\delta$ 拆成 $m=\sum_c\delta_c$ 個單位增量 $e_1,\dots,e_m$（每個 $e_u$ 為某一類別 $+1$），依序累加得到中介向量與其 latent 座標
$$x_A^{(0)}=x_A,\quad x_A^{(t)}=x_A^{(t-1)}+e_t,\quad z_A^{(t)}=f_\theta\big(x_A^{(t)}\big),\quad t=1,\dots,m,\quad z_A^{(m)}=z_A'$$
對應圖中從星形（$z_A$）到菱形（$z_A'$）的路徑，僅供視覺化，路徑本身不影響 $S_A'$ 的計算（$S_A'$ 只由最終 $x_A'$ 決定）。


## 2.4 案例研究算法（outlier / common 反事實搜尋，對應 2.2.1 / 2.2.2 案例）

此為 2.3 節「使用者手動選 $\delta$」的自動化版本：不由使用者手動指定新增哪些 POI，而是窮舉搜尋在給定預算下最能改變分數的 $\delta$。

1. 目標 patch 選法：test 集中 $S_i$ 最大者為 outlier、最小者為 common（沿用 2.2 節分數定義，$A$ 取 test 集內既有 patch，故用 leave-one-out kNN）。

2. 窮舉搜尋：對每個預算 $b=1..B_{\max}=10$，列舉所有滿足 $\sum_c\delta_c=b$ 的類別重複組合 $\delta\in\mathbb{Z}_{\ge0}^{C}$，數量 $\binom{C+b-1}{b}$，$b\le10$ 合計 184,755 種，全部枚舉、無近似、無取樣；每個候選 $\delta$ 依 2.3 節算出 $x_A'=x_A+\delta\to z_A'\to S_A'=:S^{\text{re}}$。

3. 目標方向：outlier 取 $\arg\min_\delta S^{\text{re}}$（拉回典型）、common 取 $\arg\max_\delta S^{\text{re}}$（推成離群）。


# 3. 超參數總表

## 3.1 資料建構

| 符號 | 意義 | 預設值 |
|---|---|---|
| $s$ | patch 格點間距（CENTER_STEP） | 100 m |
| $r$ | patch 鄰域半徑（HALF_WIDTH） | 50 m |
| $n_{\min}$ | patch 最少 POI 數（MIN_POI） | 10 |
| $C$ | POI 類別數（N_CAT） | 10 |

## 3.2 模型架構

| 符號 | 意義 | 預設值 |
|---|---|---|
| $d$ | 潛在空間維度（LATENT_DIM） | 2 |
| $h$ | encoder/decoder 隱藏層維度（HIDDEN） | 64 |

## 3.3 訓練與最佳化

| 符號 | 意義 | 預設值 |
|---|---|---|
| $B$ | mini-batch 大小（BATCH） | 256 |
| $T$ | 訓練總 epoch 數（EPOCHS） | 2000 |
| $\eta_{\max}$ | 初始學習率（LR） | $1\times10^{-2}$ |
| $\eta_{\min}$ | cosine annealing 最終學習率（LR_MIN） | $1\times10^{-3}$ |
| $\gamma$ | Adam weight decay 係數（WEIGHT_DECAY） | $1\times10^{-6}$ |
| seed | 隨機種子（SEED） | 0 |

## 3.4 FSCE 圖建構與損失權重

| 符號 | 意義 | 預設值 |
|---|---|---|
| $k$ | FSCE 圖 kNN 近鄰數（N_NEIGHBORS） | 10 |
| $K$ | K-Means 分群數（N_CLUSTERS，僅視覺化用） | 8 |
| $B_e$ | FSCE 邊 mini-batch 大小（EDGE_BATCH） | 256 |
| $\lambda_{\max}$ | FSCE 損失最大權重（LAMBDA_FSCE） | 0.25 |
| $T_{\text{warmup}}$ | FSCE 權重 warm-up epoch 數（WARMUP_EPOCHS） | 200 |
| spread, min\_dist | UMAP 核參數 $(a,b)$ 的輸入常數 | 1.0, 0.1 |

## 3.5 破壞噪聲

| 符號 | 意義 | 預設值 |
|---|---|---|
| $p$ | denoising 破壞機率（NOISE_P） | 0.3 |

## 3.6 資料切分

| 符號 | 意義 | 預設值 |
|---|---|---|
| val\_frac | 驗證集比例 | 0.15 |
| test\_frac | 測試集比例 | 0.2 |

## 3.7 模式選擇（非數值超參數，程式碼中可切換）

| 名稱 | 說明 | 預設值 |
|---|---|---|
| GRAPH\_MODE | FSCE 圖用 TF-IDF cosine 或 log1p count euclidean | tfidf |
| NOISE\_MODE | 破壞方式：thinning 或 mask | thinning |
| METRIC | 驗證/測試誤差指標：mae、mse、wape | mae |
| FSCE | 是否啟用 FSCE 正則項 | True |
| PCGRAD | 是否啟用 PCGrad 梯度手術 | True |


# 4. 建議的流程圖箱體順序（可直接畫成 figure）

原始打卡資料 → 座標投影+格點中心 → Patch鄰域收集(半徑50m) → 類別計數向量 $x_i$ → 〔分支A: TF-IDF→KNN圖→FSCE邊 $(w_{ij},a,b)$〕〔分支B: thinning破壞→$\tilde x_i$〕 → Encoder → $z_i$ →（Decoder→$\hat\lambda_i$，同時 $z_i,z_j$ 進 FSCE loss）→ $\mathcal{L}_{\text{recon}}+\lambda_t\mathcal{L}_{\text{fsce}}$ → PCGrad → 更新 $\theta$

（以上為 Offline 訓練迴圈；Online 案例分析則是在此迴圈結束、$\theta$ 凍結後，另外以 $z_i$ 計算 $S_i$ 並執行反事實搜尋。）
