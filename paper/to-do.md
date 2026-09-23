# Camera-ready 修改清單（SpatialConnect 2026, Submission 5）

來源標記：R1 / R2 / R3 = Reviewer 1 / 2 / 3

---

## A. 明確錯誤，必須修正

- [ ] **Figure 2 的 ΔS 解讀與內文相反**（R1-#5）
  - 內文：S_A 越高 = 越偏離 peer（outlier），ΔS > 0 = 往 outlier 移動。
  - 圖 2 寫「If ΔS > 0, then POI addition is advantageous」→ 方向相反，需統一。
  - 圖中右下文字改為：*If ΔS < 0, the addition moves the area toward its functional peers (more typical); if ΔS > 0, it moves the area away from them (more atypical).*
  - 順便修正圖中錯字 `Calcuate` → `Calculate`（兩處）。
- [ ] **Figure 2 缺少進入 stage 2 的箭頭**（R1-#2、#9b）
  - 已確認：Category Count Vector 只有連到 2. Denoising，**3. FSCE Graph Construction 的 TF-IDF Transformation 沒有任何流入箭頭**，要從 Category Count Vector 拉一條箭頭進去。
- [x] **Table 2 刻意漏掉 DAE**（R1-#4、R3-#3）
  - Table 2 把 Full model 標為最佳，但 Table 3 的純 DAE 在 MAE/MSE/WAPE 都更好。
  - 做法：Table 2 加入 DAE 那列，粗體改標 DAE，並在文中直接說明 reconstruction 上的取捨。
- [x] **Online 階段拿 test set 當 reference database**（R1-#6）
  - Method 3.2「leave-one-out kNN average latent distance against the test set」需要改成用 train（或 train+val）作為 reference，test set 只用於評估。
  - 已改：論文 Method 文字、`experiment/2.2.{1,2}_*/get_data.py`（reference = train+val）。
  - 已重跑並更新 `paper/figure/fig7.png`（由 `experiment/2.2.3_case_combined/draw.py` 產生）與 5.4 內文分數：outlier 0.91→0.17、common 0.08→2.59。
- [x] **內文交叉引用錯誤**（自行檢查發現）
  - L316：「Section 5.3」的 log(1+x) baseline，Section 5.3 實際上沒有這個比較 → 補上實驗或刪除這句。
  - L376：「TF-IDF preprocessing module discussed in Section 2」→ 應為 Section 3。
  - `[cite: 1]` placeholder（R1-#9a）：目前 tex 中已找不到，編譯後再確認 PDF。

## B. 論述與 claim 降調（三位 reviewer 共同重點）

- [x] **明確定義「typicality ≠ suitability」的假設**（R1-#1、R2-#1）
  - 在 Introduction 或 Method 明確寫出：S_A 衡量的是「相對於功能相似 peer 的典型程度」，並把「典型 ≈ 結構平衡」寫成本文的 **working assumption**。
  - 承認反例：機場、醫療區、大學、娛樂區等特化區域可能是合理的 outlier；常見的配置也可能是 oversupply。
- [x] **Case study 改定位為「評分機制的示範」而非「驗證」**（R1-#2、R2-#1、R3-#2）
  - 5.4 中「This proves」「directly validates the core premise」等字眼改為 illustrate / demonstrate the behavior of the score。
  - 說明兩個案例是用同一個 score 最佳化出來的，所以只證明 objective 可控，不代表加入的 POI 在現實中合適。
- [x] **全文用字降調**
  - Abstract、Introduction 最後一段、5.3（「This proves…」「guaranteeing…」）、Conclusion 中的 prove / validate / robustly quantifies 等改成較保守的說法。
  - 標題與 abstract 中的 "suitability" 考慮加上限定語（例如 peer-relative / typicality-based suitability）。
- [x] **新增 Limitations & Future Work 段落**（R1-#7、R2-#3、R3-#2）
  - 缺乏外部 ground truth（歷史結果、需求指標、專家評估）。
  - Foursquare 2012–2013 資料老舊，且有人口、平台、取樣偏差；check-in venue ≠ 實際設施。
  - 說明為何不使用 user / temporal 資訊（本文只關注空間組成），並承認因此看不到實際需求、使用量、容量、可及性、商業可行性。
  - 提出可行的驗證路徑：例如用較新的 POI 資料看實際新增/關店、需求代理指標、規劃師評估。
- [x] **補充實際應用情境與利害關係人**（R2-#3）
  - 描述預期的規劃情境（誰用、怎麼用、在決策流程哪一步），呼應 workshop 的 community-oriented 主題。

## C. 補實驗 / 補分析（視時間取捨，優先度由高到低）

- [ ] **直接 baseline：不經學習的 peer-distance score**（R2-#2、R1-#3）
  - 在 TF-IDF 空間、log(1+x) 空間直接算相同的 kNN 平均距離 score，比較與 learned latent 的差異（同時也補上 L316 承諾的 log(1+x) 比較）。
  - 可加：LOF、Isolation Forest、PCA/UMAP-based scoring。
  - 至少比較這些 score 的排名一致性（Spearman / Kendall）與 case study 上的行為。
- [ ] **強化 FSCE 的量化證據**（R1-#4、R3-#3）
  - Figure 5 neighbor preservation 約 0.16，需加上：AE / DAE / PCA / 無 FSCE 的 baseline、多個 seed 的 variance、train vs held-out 的比較。
  - 可額外報 trustworthiness / continuity 等 topology 指標。
- [ ] **Spatial block cross-validation**（R1-#6）
  - 以空間區塊切分 train/test（或切出 Tokyo 另一區域）以避免空間自相關造成的資料洩漏；時間不夠至少在 Limitations 中提及。
- [ ] **與 Parametric UMAP 的關係**（R2-#2）
  - 在 Related Work 或 Method 說明本方法與 Parametric UMAP（同樣使用 fuzzy simplicial set 目標、可加 reconstruction）的差異（Poisson reconstruction、thinning denoising、PCGrad、TF-IDF graph）。

## D. 圖表與排版

- [ ] **Figure 3(b) 空間分布圖難以判讀**（R1-#8）
  - 點重疊嚴重；改為密度圖/hexbin，或疊上道路與車站圖層；或分類別 small multiples。
  - 若無法加入道路/車站，刪除或弱化「沿主要道路與車站聚集」的描述，或補簡單量化（例如距車站距離的分布）。
- [ ] 資料集老舊的說明（R1-#3、#7）：在 Dataset 節說明選用 TKY 的原因（公開 benchmark、可重現），並列入 Limitations。
- [ ] Preamble 重複 `\usepackage{makecell}`，清掉一個。
- [ ] Hyperparameters 中的 K-Means K=8 只用於視覺化，說明其用途，避免讀者誤會是模型的一部分。
