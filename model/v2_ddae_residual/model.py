import os
import sys

import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from cfg import HIDDEN
from config.dataset import N_CAT


class ResidualLinear(nn.Module):
    """pre-norm 的 residual block：輸出 = shortcut(x) + act(linear(norm(x)))。
    LayerNorm 只作用在 residual branch 的入口，主幹（shortcut）維持一條沒有
    任何 normalize 的通路，梯度可以直接流回前面的層。in_dim/out_dim 相同時
    shortcut 是 identity；不同時 shortcut 是另一個不帶 bias 的 Linear，把 x
    投影到 out_dim 維後再相加（ResNet 的 projection shortcut）。

    in_dim/out_dim：輸入/輸出維度。activate：是否對 linear 的輸出套 GELU
    （latent、log_lam 這種輸出層要傳 False，維持沒有激活函數）。prenorm：
    是否在 residual branch 入口放 LayerNorm(in_dim)。encoder 第一層（輸入是
    raw count，normalize 掉會抹掉整包 patch 的總量資訊）跟 decoder 第一層
    （輸入是 2 維 latent，LayerNorm 會把它壓成 ±1）都要傳 False。
    """

    def __init__(self, in_dim, out_dim, activate=True, prenorm=True):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim) if prenorm else nn.Identity()
        self.linear = nn.Linear(in_dim, out_dim)
        if in_dim == out_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.zeros_(self.shortcut.weight)
        self.act = nn.GELU() if activate else nn.Identity()

    def forward(self, x):
        """x 是 (B,in_dim)，回傳 (B,out_dim)。"""
        return self.shortcut(x) + self.act(self.linear(self.norm(x)))


class AE(nn.Module):
    """跟 v2_ddae_base 的 AE 差異只有每層都套了 ResidualLinear（見
    ResidualLinear docstring）；層數、HIDDEN 維度、latent 前不接激活都沒變。
    denoising 完全發生在資料端（見 dataset.corrupt()），模型本身不需要知道。
    """

    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualLinear(N_CAT, HIDDEN, prenorm=False),   # 輸入是 raw count，normalize 會抹掉 patch 的總量
            ResidualLinear(HIDDEN, HIDDEN),
            ResidualLinear(HIDDEN, HIDDEN),
            ResidualLinear(HIDDEN, HIDDEN),
            ResidualLinear(HIDDEN, latent_dim, activate=False),   # latent 前不接激活，離群值才不會被壓回來
        )
        self.decoder = nn.Sequential(
            ResidualLinear(latent_dim, HIDDEN, prenorm=False),   # 輸入是 2 維 latent，LayerNorm 會把它壓成 ±1
            ResidualLinear(HIDDEN, HIDDEN),
            ResidualLinear(HIDDEN, HIDDEN),
            ResidualLinear(HIDDEN, HIDDEN),
            ResidualLinear(HIDDEN, N_CAT, activate=False),   # 輸出是 log λ
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def poisson_nll(log_lam, x):
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)


def build_fsce_graph(x, n_neighbors=15, metric="euclidean"):
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
    d2 = (z_i - z_j).pow(2).sum(dim=1).clamp(min=eps)
    q = (1.0 + a * d2.pow(b)).reciprocal().clamp(eps, 1 - eps)
    return -(w * q.log() + (1 - w) * (1 - q).log())


if __name__ == '__main__':
    model = AE(latent_dim=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
