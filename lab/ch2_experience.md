# 此為論文第二章節大綱：實驗

## 0. 資料觀察
資料的簡單統計跟數據看有幾個 POI 比例等等，還有輸入之間相不相似
## 1. 量化
### 1-1 重建比對: 我們會看看
1-1-1 
baseline降到二維的不同方法，使用PCA，AE，VAE，跟我們的方法
1-1-2
看看加入不同POI量個別偏移多少
### 1-2 ablation study
看看我們加跟沒加架構的差異
AE
AE+entropy
AE+entorpy+tfidf
DAE
DAE+entropy
DAE+entorpy+tfidf
1-2-2
看看加入不同POI量個別偏移多少
1-2-3 TF-IDF 前處理對分群品質的影響

## 2. 質化
### 2-1 丟不同模型重建圖跟我們每個點都平移後比較
### 2-2 驗證 entropy那個
需要一張圖，我們需要看 entropy在 25% 50% 100% 的epochs progress中重建的2維長怎樣，同時也要這些平移
