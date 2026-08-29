訓練用的是 $q_{ij}=(1+a\lVert z_i-z_j\rVert^{2b})^{-1}$，這裡的分數卻是原始歐氏距離。兩者單調相關但不同單位——你在 .note/s.md 的 M1 是用 $q$ 定義的，實作是歐氏距離，要挑一個並在 method 交代理由。

設計：每個 test patch × 每個類別 $c$ × 加入量 $n\in{1,2,3}$，記錄 $\Delta_{i,c,n}=S^{\text{fix}}(x_i+n e_c)-S^{\text{fix}}(x_i)$。
UNIFORM 對照組：同樣加 $n$ 個但平均分給 10 類（每類 $+n/10$），用來把「總量效應」和「類別效應」拆開。
淨效應（逐 patch 配對相減，不是組平均相減）：$\Delta^{\text{net}}{i,c,n}=\Delta{i,c,n}-\Delta_{i,\text{unif},n}$。
統計呈現：平均 + 95% CI（$1.96\hat\sigma/\sqrt n$）、距離縮短的 patch 比例（對照 50% 線）、以及 1–99 百分位裁切後的差距分布直方圖。沒有做多重比較校正（10 類 × 3 個量 = 30 次比較），建議在 method 主動說明或補 Bonferroni/BH。
順便把 1.3.1 一起寫進去：它跟 1.3.2 只差在鄰居定義（空間 kNN vs latent kNN）。把「鄰居定義」抽成方法裡的一個參數，正好對上你 s.md 裡「地理鄰接是外生訊號、不循環」那個論點，這是很強的一段，別漏掉。

$\arg\min_\delta S^{\text{re}}$

分數定義（leave-one-out kNN 平均 latent 距離）：
$$S_i=\frac{1}{k_S}\sum_{j\in\mathcal N_{k_S}(i)}\lVert z_i-z_j\rVert_2,\qquad k_S=8$$
$\mathcal N_{k_S}(i)$ 是 test 集內排除自己的最近 $k_S$ 個。$S_i$ 大＝離群、小＝典型。
2.2.1 / 2.2.2（案例研究那兩個）的算法
1.目標 patch 選法：test 集中 $S_i$ 最大者為 outlier、最小者為 common
2. 窮舉搜尋：對每個預算 $b=1..B_{\max}=10$，列舉所有總和恰為 $b$ 的類別重複組合，數量 $\binom{C+b-1}{b}$，$b\le10$ 合計 184,755 種，全部枚舉、無近似、無取樣。
3. 目標方向：outlier 取 $\arg\min_\delta S^{\text{re}}$（拉回典型）、common 取 $\arg\max_\delta S^{\text{re}}$（推成離群）
