import os
import sys

import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from cfg import HIDDEN
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
        """x：(B,N_CAT) 的 count 向量。回傳 (B,latent_dim) 的 latent z。"""
        return self.encoder(x)

    def forward(self, x):
        """x：(B,N_CAT) 的 count 向量。回傳 (z, log_lam)：latent 與重建的 log λ。"""
        z = self.encoder(x)
        return z, self.decoder(z)


def poisson_nll(log_lam, x):
    """log_lam：(B,N_CAT) 重建的 log λ。x：(B,N_CAT) 乾淨 count。回傳 (B,) 的 NLL。"""
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    """log_lam：(B,N_CAT) 重建的 log λ。x：(B,N_CAT) 乾淨 count。回傳 (B,) 的 deviance。"""
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
