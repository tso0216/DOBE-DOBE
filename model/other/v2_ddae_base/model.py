import os
import sys

import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from model.other.v2_ddae_base.cfg import HIDDEN
from common.dataset import N_CAT


class AE(nn.Module):
    def __init__(self, latent_dim=2, hidden=HIDDEN):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_CAT, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),   # latent 前不接 norm/激活
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, N_CAT),   # 輸出是 log λ，不接 norm
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def poisson_nll(log_lam, x):
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def wape(log_lam, x):
    """log_lam：(B,N_CAT) 重建的 log λ。x：(B,N_CAT) 乾淨 count。回傳 (B,) 各 patch 的 WAPE。"""
    lam = torch.exp(log_lam)
    return (x - lam).abs().sum(dim=1) / x.sum(dim=1).clamp_min(1e-8)


def mae(log_lam, x):
    """log_lam：(B,N_CAT) 重建的 log λ。x：(B,N_CAT) 乾淨 count。回傳 (B,) 各 patch 的 MAE。"""
    lam = torch.exp(log_lam)
    return (x - lam).abs().mean(dim=1)


def mse(log_lam, x):
    """log_lam：(B,N_CAT) 重建的 log λ。x：(B,N_CAT) 乾淨 count。回傳 (B,) 各 patch 的 MSE。"""
    lam = torch.exp(log_lam)
    return (x - lam).pow(2).mean(dim=1)


METRICS = {"wape": wape, "mae": mae, "mse": mse}   # cfg.METRIC 選哪個就用哪個當評估指標


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
