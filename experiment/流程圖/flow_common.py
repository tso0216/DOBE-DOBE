"""流程圖那幾張 latent 圖共用的設定，跟 experiment/2.2.1_case_outlier 對齊：
同一份 checkpoint、同一種 split、同一種 S 分數算法。

latent 一律用 CKPT 現場重算，不吃 result/latents 的預存檔（那份是另一次訓練的
輸出，座標系對不上）。建 KDTree 評分時會把目標自己排除，否則位移之後它的原始
位置會被當成鄰居，把分數壓低。
"""
import itertools
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import N_CAT, PATCHES, make_split  # noqa: E402
from model.v3_ddae_tfidf.dataset import Patches  # noqa: E402
from model.v3_ddae_tfidf.model import AE  # noqa: E402

CKPT = os.path.join(ROOT, "experiment", "model", "ddae_fsce_tfidf", "fold3_mae.pt")
K = 8            # 取 latent 上最近的 k 個 patch 當鄰居
B_MAX = 10       # 加入的 POI 數上限（窮舉組合數隨 B_MAX 爆炸，10 上下才跑得動）
PCT = 0          # 目標點取分數分布的第幾百分位（0=分數最低／latent 最密集，100=最離群）
BATCH = 16384    # 編碼候選時的批次大小
SEED = 0
CMAP = "viridis"


def load_model(path=CKPT):
    """path：checkpoint 路徑。回傳：AE，已載入權重並設為 eval。"""
    sd = torch.load(path, map_location="cpu")
    hidden = sd["encoder.0.weight"].shape[0]
    latent_dim = sd["decoder.0.weight"].shape[1]
    model = AE(latent_dim, hidden)
    model.load_state_dict(sd)
    model.eval()
    return model


def encode(model, x):
    """model：AE。x：(B, N_CAT) count tensor。回傳：(B, latent_dim) numpy latent。"""
    with torch.no_grad():
        out = [model.encoder(x[i:i + BATCH]) for i in range(0, len(x), BATCH)]
    return torch.cat(out).numpy().astype(np.float64)


def loo_score(z_all, k=K):
    """z_all：(N, d) latent。k：鄰居數。回傳：(N,) 每個 patch 到自己以外最近 k 個
    patch 的平均距離（leave-one-out）。"""
    dist, idx = cKDTree(z_all).query(z_all, k=k + 1)
    mask = idx != np.arange(len(z_all))[:, None]   # 有重複座標時 self 未必排第一欄
    nb_dist = np.array([row[m][:k] for row, m in zip(dist, mask)])
    return nb_dist.mean(axis=1)


def knn_score(tree, z, k=K):
    """tree：建在參考點集上的 cKDTree。z：(B, d) 要評分的座標。k：鄰居數。
    回傳：(B,) 每個座標到參考點集中最近 k 個點的平均距離。"""
    dist, _ = tree.query(z, k=k)
    return dist.mean(axis=1)


def all_additions(budget=B_MAX, n_cat=N_CAT):
    """budget：加入的 POI 數上限。回傳：(M, n_cat) 增量矩陣與 (M,) 加入總數，
    涵蓋所有總和為 1~budget 的加法組合（重複組合，不含全 0）。"""
    rows, sizes = [], []
    for b in range(1, budget + 1):
        for combo in itertools.combinations_with_replacement(range(n_cat), b):
            rows.append(np.bincount(combo, minlength=n_cat))
            sizes.append(b)
    return np.array(rows, dtype=np.float32), np.array(sizes)


def load():
    """回傳：dict，含 model、data、eval_idx（test 集的 patch 編號）、x_test（count）、
    z_test、score（每個 test patch 的 S 分數）、pos（目標點在 test 集內的位置）、
    target（它的 patch 編號）、tree（扣掉目標自己的參考點集）。

    目標點不取最極端的那個，而是把分數由低到高排序後取第 PCT 百分位。"""
    data = Patches(PATCHES)
    _, _, test_idx = make_split(data.lat, data.lon, seed=SEED)
    eval_idx = test_idx.numpy()
    x_test = data.agg(test_idx)

    model = load_model()
    z_test = encode(model, x_test)
    score = loo_score(z_test)

    order = np.argsort(score)
    rank = np.clip(round(len(order) * PCT / 100) - 1, 0, len(order) - 1)
    pos = int(order[rank])
    keep = np.arange(len(eval_idx)) != pos

    return dict(model=model, data=data, eval_idx=eval_idx, x_test=x_test,
                z_test=z_test, score=score, pos=pos,
                target=int(eval_idx[pos]), tree=cKDTree(z_test[keep]))
