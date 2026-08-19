"""雙分支 encoder：count 分支（MLP）＋ 幾何分支（GAT），兩支各出 HIDDEN 維，
concat 之後用同一個 head 壓成 z。loss 跟 v2_ddae_fsce 完全一樣，沒有加新的項。

count 分支的每個 hidden block 後面、以及兩支 concat 後的融合層後面，都
殘差疊加一組 NeuronMoE（見 ae.py 同名 class，跟 v2_ddae_moe 是同一份實作）：
輸出向量的每一個神經元各自有自己的 router（cosine gating）跟自己的一組
expert，互不共用權重。GAT 分支本身沒有 MoE（節點是 N_CAT 個類別、不是
64 維 hidden 向量，接法不一樣，這裡先不動）。

幾何用 patch 內 POI 的座標距離，壓成 N_CAT x N_CAT 對稱矩陣：
    D[i,j] = 每個 i 類的點到最近的 j 類點的距離，對 i 類取平均
    e[i,j] = log( D[i,j] / (0.5 / sqrt(n_j / AREA)) )
分母是隨機均勻散布時的期望最近鄰距離，所以 e=0 是隨機、負是吸引、正是排斥；
除掉它，n_j 的數量資訊才不會從幾何這邊再漏一次跟 count 分支搶 latent 維度。

GAT 的節點固定是那 N_CAT 個類別，節點特徵只有類別 embedding 加上 log1p(count)，
patch 之間的差異全部由邊特徵 e 承載——所以 e 必須進 attention 的計分式
（GATv2），不能只當標量權重。

破壞在 POI 層級擲銅板，count 與距離矩陣都用活下來的點算：
  "thinning"  每個 POI 以 1-p 的機率被保留
  "mask"      每一類以 p 的機率整批被丟掉
count 除以 1-p 做尺度補償；距離矩陣是比值、不需要補償。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT, HALF_WIDTH  # noqa: E402,F401

HIDDEN = 64
GAT_HEADS = 4
GAT_LAYERS = 2
EXPERT_HIDDEN = 128
N_EXPERTS = 4
COSINE_TEMP = 0.07  # cosine gating 的 softmax 溫度，越小分佈越尖

AREA = (2.0 * HALF_WIDTH) ** 2   # patch 是邊長 2*HALF_WIDTH 的方形窗


class Patches:
    def __init__(self, path):
        d = np.load(path)
        self.cat = torch.from_numpy(d["cat"].astype(np.int64))
        self.dx = torch.from_numpy(d["dx"].astype(np.float32))
        self.dy = torch.from_numpy(d["dy"].astype(np.float32))
        self.offsets = torch.from_numpy(d["offsets"])
        self.n_poi = d["n_poi"]
        self.lat = d["center_lat"]
        self.lon = d["center_lon"]
        self.n = len(self.n_poi)

    def _pad(self, idx):
        starts = self.offsets[idx]
        lens = self.offsets[idx + 1] - starts
        m = int(lens.max())
        ar = torch.arange(m)
        valid = ar[None, :] < lens[:, None]
        pos = (starts[:, None] + ar[None, :]).clamp(max=len(self.cat) - 1)
        dxy = torch.stack([self.dx[pos], self.dy[pos]], dim=-1)
        return dxy, self.cat[pos], valid

    def agg(self, idx):
        _, cat, valid = self._pad(idx)
        return (F.one_hot(cat, N_CAT) * valid[..., None]).sum(dim=1).float()

    def batch(self, idx, p=0.0, mode="thinning", generator=None):
        """回傳 (x, x_in, e, emask)：x 乾淨 count (B,N_CAT)、x_in 加噪 count
        (B,N_CAT)、e 距離矩陣 (B,N_CAT,N_CAT)、emask 哪些格算得出來。"""
        dxy, cat, valid = self._pad(idx)
        onehot = F.one_hot(cat, N_CAT)
        x = (onehot * valid[..., None]).sum(dim=1).float()

        if p <= 0:
            keep, scale = valid, 1.0
        elif mode == "thinning":
            coin = torch.rand(valid.shape, generator=generator) < (1.0 - p)
            keep, scale = valid & coin, 1.0 - p
        elif mode == "mask":
            coin = torch.rand((len(idx), N_CAT), generator=generator) < (1.0 - p)
            keep, scale = valid & coin.gather(1, cat), 1.0 - p
        else:
            raise ValueError(f"未知的 mode：{mode}")

        x_keep = (onehot * keep[..., None]).sum(dim=1).float()
        e, emask = dist_matrix(dxy, cat, keep, x_keep)
        return x, x_keep / scale, e, emask


def dist_matrix(dxy, cat, keep, counts):
    """回傳 (e, emask)，形狀都是 (B,N_CAT,N_CAT)。"""
    b, m, _ = dxy.shape
    d = torch.cdist(dxy, dxy)
    d = d.masked_fill(torch.eye(m, dtype=torch.bool)[None], torch.inf)  # 不算自己
    d = d.masked_fill(~keep[:, None, :], torch.inf)                     # 終點得還活著

    # dnn[b,u,j]：第 u 個點到最近的 j 類點的距離，patch 裡沒有 j 類就是 inf
    dnn = torch.stack(
        [d.masked_fill(~(keep & (cat == j))[:, None, :], torch.inf).min(dim=2).values
         for j in range(N_CAT)], dim=2)

    fin = torch.isfinite(dnn)
    src = (F.one_hot(cat, N_CAT) * keep[..., None]).float()
    num = torch.einsum("bmi,bmj->bij", src, torch.where(fin, dnn, 0.0))
    den = torch.einsum("bmi,bmj->bij", src, fin.float())

    exp_d = 0.5 / (counts.clamp(min=1e-8) / AREA).sqrt()
    r = (num / den.clamp(min=1.0)) / exp_d[:, None, :]
    r = 0.5 * (r + r.transpose(1, 2))

    emask = (den > 0)
    emask = emask & emask.transpose(1, 2)   # 兩個方向都要算得出來
    e = torch.where(emask, r, torch.ones_like(r)).clamp(min=1e-3).log()
    return e, emask


class GATBranch(nn.Module):
    """節點固定是 N_CAT 個類別、全連接，所以直接用 dense 張量算 attention。
    GATv2：a 在 LeakyReLU 之後，attention 才會隨 patch 改變——節點特徵幾乎
    是固定的 embedding，用 v1 的 static attention 這條分支等於廢掉。
    """

    def __init__(self, hidden=HIDDEN, heads=GAT_HEADS, layers=GAT_LAYERS):
        super().__init__()
        self.heads, self.dh = heads, hidden // heads
        self.emb = nn.Embedding(N_CAT, hidden)
        self.cnt = nn.Linear(1, hidden)
        self.w = nn.ModuleList(nn.Linear(hidden, hidden, bias=False)
                               for _ in range(layers))
        self.we = nn.ModuleList(nn.Linear(1, hidden, bias=False)
                                for _ in range(layers))
        self.a = nn.ParameterList(
            nn.Parameter(nn.init.xavier_uniform_(torch.empty(heads, self.dh)))
            for _ in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))

    def forward(self, x, e, emask):
        b, n = x.shape
        cnt = x.clamp(min=0.0)
        share = cnt / cnt.sum(dim=1, keepdim=True).clamp(min=1e-6)
        h = self.emb.weight[None] + self.cnt(torch.log1p(cnt)[..., None])

        nmask = cnt > 0
        eye = torch.eye(n, dtype=torch.bool, device=x.device)[None]
        m = (emask | (eye & nmask[:, None, :])) & nmask[:, None, :]  # 自環避免整列 -inf
        e = e[..., None]

        for w, we, a, norm in zip(self.w, self.we, self.a, self.norm):
            wh = w(h).view(b, n, self.heads, self.dh)
            s = F.leaky_relu(wh[:, :, None] + wh[:, None, :]
                             + we(e).view(b, n, n, self.heads, self.dh), 0.2)
            score = (s * a).sum(dim=-1).masked_fill(~m[..., None], -torch.inf)
            alpha = torch.nan_to_num(score.softmax(dim=2))
            out = torch.einsum("bijh,bjhd->bihd", alpha, wh).reshape(b, n, -1)
            h = F.gelu(norm(out * nmask[..., None]))

        return (h * share[..., None]).sum(dim=1)   # count=0 的節點 share=0，自然排除


class NeuronMoE(nn.Module):
    """輸出的每一個神經元各自有自己的 router 跟自己的一組 expert（互不共用
    權重）：expert 是 in_dim -> hidden -> 1 的 2 層 MLP，router 是輸入方向跟
    這個神經元專屬的 N 個 expert 方向做 cosine 相似度、softmax 出來的稠密
    混合（不做 top-k 截斷）。所有神經元、所有 expert 的權重疊成大張量，
    用 einsum 一次算完，不逐一 for-loop。
    """

    def __init__(self, in_dim, out_dim, n_experts=N_EXPERTS, hidden=EXPERT_HIDDEN,
                cos_temp=COSINE_TEMP):
        super().__init__()
        self.temp = cos_temp
        self.proto = nn.Parameter(torch.randn(out_dim, n_experts, in_dim) * in_dim ** -0.5)
        self.w1 = nn.Parameter(
            (torch.rand(out_dim, n_experts, in_dim, hidden) * 2 - 1) * in_dim ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(out_dim, n_experts, hidden))
        self.w2 = nn.Parameter(
            (torch.rand(out_dim, n_experts, hidden) * 2 - 1) * hidden ** -0.5)
        self.b2 = nn.Parameter(torch.zeros(out_dim, n_experts))

    def forward(self, x):
        """x 是 (B,in_dim)，回傳 (B,out_dim)：每個神經元用自己的 router 權重
        混合自己的 N 個 expert 算出來的值。
        """
        sim = torch.einsum("bi,oni->bon", F.normalize(x, dim=-1),
                           F.normalize(self.proto, dim=-1))       # (B,out_dim,N)
        gates = F.softmax(sim / self.temp, dim=-1)
        h = F.gelu(torch.einsum("bi,onih->bonh", x, self.w1) + self.b1)
        y = torch.einsum("bonh,onh->bon", h, self.w2) + self.b2   # (B,out_dim,N)
        return (gates * y).sum(dim=-1)


class AE(nn.Module):
    """count 分支是 4 個 64 維 hidden block，每個 block（Linear+LayerNorm+GELU）
    後面殘差疊加一組 NeuronMoE；GAT 分支不動。兩支 concat 後的融合層
    （head）後面也殘差疊加一組 NeuronMoE。decoder 一模一樣沒動。"""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(N_CAT, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU()),
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU()),
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU()),
            nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU()),
        ])
        self.moe_hidden = nn.ModuleList(NeuronMoE(HIDDEN, HIDDEN) for _ in self.blocks)
        self.gat = GATBranch(HIDDEN)
        # 兩支先各自 LayerNorm 再 concat，count 那支的量級才不會蓋過幾何那支
        self.norm_cnt = nn.LayerNorm(HIDDEN)
        self.norm_geo = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN * 2, latent_dim)  # latent 前不接 norm/激活
        self.moe_head = NeuronMoE(HIDDEN * 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, N_CAT),   # 輸出是 log λ，不接 norm
        )

    def encode_cnt(self, x):
        h = x
        for block, moe in zip(self.blocks, self.moe_hidden):
            h = block(h)
            h = h + moe(h)
        return h

    def encode(self, x, e, emask):
        h = torch.cat([self.norm_cnt(self.encode_cnt(x)),
                       self.norm_geo(self.gat(x, e, emask))], dim=1)
        return self.head(h) + self.moe_head(h)

    def forward(self, x, e, emask):
        z = self.encode(x, e, emask)
        return z, self.decoder(z)

    def branch_ratio(self):
        """head 權重 count 半 / 幾何半的範數比，>>1 代表幾何被無視。"""
        w = self.head.weight.detach()
        return (w[:, :HIDDEN].norm() / w[:, HIDDEN:].norm().clamp(min=1e-12)).item()


def poisson_nll(log_lam, x):
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)


def build_fsce_graph(x, n_neighbors=15, metric="euclidean"):
    """回傳 (edge_i, edge_j, edge_w, a, b)：UMAP fuzzy simplicial set 的邊兩端、
    模糊隸屬度，以及低維核 1/(1+a·d^(2b)) 的形狀參數。"""
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(x)
    knn_dists, knn_idx = knn.kneighbors(x)
    graph, _, _ = fuzzy_simplicial_set(
        x, n_neighbors=n_neighbors, random_state=0, metric=metric,
        knn_indices=knn_idx, knn_dists=knn_dists,
    )
    graph = graph.tocoo()
    edge_i = torch.from_numpy(graph.row).long()
    edge_j = torch.from_numpy(graph.col).long()
    edge_w = torch.from_numpy(graph.data).float()
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    return edge_i, edge_j, edge_w, a, b


def fsce_loss(z_i, z_j, w, a, b, eps=1e-4):
    # d2 用 eps 墊底：b<1 時 d2^b 在 0 處梯度無限大，兩點重合會讓 backward 炸 NaN
    d2 = (z_i - z_j).pow(2).sum(dim=1).clamp(min=eps)
    q = (1.0 + a * d2.pow(b)).reciprocal().clamp(eps, 1 - eps)
    return -(w * q.log() + (1 - w) * (1 - q).log())
