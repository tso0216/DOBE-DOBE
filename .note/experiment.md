 
資料建構 
1.資料來源：Foursquare 東京打卡紀錄，每筆資料為一筆 POI 的經緯度與所屬類別，類別使用Foursquare官方的類別層級中的最頂層 $$C=10$$ 。
2.座標投影：將原始經緯度（WGS84）投影至日本平面直角座標系第9系（EPSG:6677），單位為m。
3.網格：以邊長$$s$$將投影平面切成規則格網，僅保留  $$|N_i| \ge $$n_{\min}$$的網格作為正式 patch，過濾掉 POI 過於稀疏、統計上不具代表性的區域。
特徵聚合
1.對每個 patch，統計各類別出現次數，得到計數向量 $$x_i$$：$$x_{i,c}=\sum_{q\in N_i}\mathbb{1}[\text{cat}(q)=c],\quad c=1,\dots,C$$

TF-IDF transform
1.對全體 patch 的計數矩陣 $$X\in\mathbb{Z}^{N\times C}$$ 做 TF-IDF（sklearn.TfidfTransformer，smooth_idf=True, L2 normalize）：
$$\text{idf}_c=\ln\frac{1+N}{1+\text{df}_c}+1,\qquad \tilde x_{i,c}=\frac{x_{i,c}\cdot\text{idf}_c}{\lVert x_{i,\cdot}\odot\text{idf}\rVert_2}$$

FSCE Graph
1.在 TF-IDF 空間以 cosine 距離找 $$k=10$$近鄰，用 UMAP 的 fuzzy_simplicial_set 建圖，得到邊集合 $$(i,j)$$ 與邊權重 $$w_{ij}\in[0,1]$$，同時求得核函數參數 $$(a,b)$$
2.這張圖只在訓練集內的 patch 上建，訓練時每個 mini-batch 隨機抽 EDGE_BATCH 條正邊 $$(i,j,w_{ij})$$，另外均勻抽等量負邊（隨機配對、權重設為 0）作為對照

Denoise
1.對計數向量做 thinning 破壞：每個計數單位獨立以機率 $$\text{keep}=1-p$$ 保留，再除以 $$\text{keep}$$ 做無偏縮放：
$$\tilde x_{i,c}=\frac{1}{1-p}\sum_{u=1}^{x_{i,c}} \text{Bernoulli}(1-p)_u,\qquad \mathbb{E}[\tilde x_{i,c}]=x_{i,c}$$
2.encoder吃的是 $$\tilde x_i$$，reconstruction 目標是原始 $$x_i$$

模型架構
1.Encoder $$f_\theta:\mathbb{R}^C\to\mathbb{R}^d$$：4 層 Linear→LayerNorm→GELU（hidden = $$h$$）+ 最終線性層到 $$d$$。
2.Decoder $$g_\theta:\mathbb{R}^d\to\mathbb{R}^C$$：對稱結構，輸出 $$\log\hat\lambda_i=g_\theta(z_i)$$（輸出的是 log 率參數，不是直接的計數重建）：
$$z_i=f_\theta(\tilde x_i),\qquad \log\hat\lambda_i=g_\theta(z_i)$$

損失函數
(a) 重建損失：Poisson 負對數似然：
$$\mathcal{L}_{\text{recon}}=\frac1C\sum_{c=1}^{C}\left(\hat\lambda_{i,c}-x_{i,c}\log\hat\lambda_{i,c}\right)$$
(b) FSCE 損失（UMAP 交叉熵形式，把高維邊權重 $$w_{ij}$$ 當標籤，低維用 Student-t 型核 $$q$$ 逼近）：
$$q_{ij}=\left(1+a\,\lVert z_i-z_j\rVert_2^{2b}\right)^{-1}$$
$$\mathcal{L}_{\text{fsce}}=-\big[w_{ij}\log q_{ij}+(1-w_{ij})\log(1-q_{ij})\big]$$
(c) 總損失與 warm-up：
$$\lambda_t=\lambda_{\max}\cdot\min\left(1,\frac{t}{T_{\text{warmup}}}\right),\qquad \mathcal{L}=\mathcal{L}_{\text{recon}}+\lambda_t\,\mathcal{L}_{\text{fsce}}$$


多目標梯度手術(PCGrad)
1.對 $$\mathcal{L}_{\text{recon}}$$ 與 $$\lambda_t\mathcal{L}_{\text{fsce}}$$ 分別對參數 $$\theta$$ 取梯度 $$g_r,g_f$$。
2.若兩者方向衝突（內積為負），把 $$g_f$$ 投影掉與 $$g_r$$ 衝突的分量，只保留正交分量；$$g_r$$ 本身不變（優先保護重建梯度）：
$$\text{若 } g_r\cdot g_f<0:\quad g_f' = g_f-\frac{g_r\cdot g_f}{\lVert g_r\rVert^2}g_r,\qquad g_\theta=g_r+g_f'$$
$$\text{否則:}\quad g_\theta=g_r+g_f$$

訓練流程
1. data split：make_split，隨機切分網格，train/val/test $$\approx 60\%/20\%/20\%$$。
2. optimizer：Adam，weight decay $$\gamma$$，訓練 $$T$$ 個 epoch，batch size $$B$$，學習率採 cosine annealing，從 $$\eta_{\max}$$ 降到 $$\eta_{\min}$$：
$$\eta_t=\eta_{\min}+\frac12(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)$$
4. 最終在從未參與訓練或選模的 test 集，作為論文的主要指標


hyperparameter
資料建構與模型架構
符號
意義
預設值
$$s$$
patch 網格間距
100 m
$$n_{\min}$$
patch 最少 POI 數
10
$$C$$
POI 類別數
10
$$d$$
latent dimension
2
$$h$$
encoder/decoder 隱藏層維度 
64


訓練與最佳化
符號
意義
預設值
$$B$$
mini-batch size
256
$$T$$
epoch
2000
$$\eta_{\max}$$
初始學習率 (LR)
$$1 \times 10^{-2}$$
$$\eta_{\min}$$
最終學習率 (LR_MIN)
$$1 \times 10^{-3}$$
$$\gamma$$
Adam weight decay 係數
$$1 \times 10^{-6}$$
seed
seed
0


FSCE 與圖建構
符號
意義
預設值
$$k$$
kNN 近鄰數
10
$$K$$
K-Means 分群數
8
$$B_e$$
FSCE edge mini-batch size
256
$$\lambda_{\max}$$
FSCE 損失最大權重
0.25
$$T_{\text{warmup}}$$
Warm-up epoch 數
200
$$a, b$$
UMAP 核參數 (spread, min_dist)
1.0, 0.1


雜訊、切分與模式設定
參數
意義
預設值
$$p$$
denoing probability
0.3
NOISE_MODE
破壞方式
thinning


