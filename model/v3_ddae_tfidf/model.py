import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from model.v3_ddae_tfidf.cfg import HIDDEN, N_NEIGHBORS
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
            nn.Linear(hidden, latent_dim),
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
            nn.Linear(hidden, N_CAT),
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
    lam = torch.exp(log_lam)
    return (x - lam).abs().sum(dim=1) / x.sum(dim=1).clamp_min(1e-8)


def mae(log_lam, x):
    lam = torch.exp(log_lam)
    return (x - lam).abs().mean(dim=1)


def mse(log_lam, x):
    lam = torch.exp(log_lam)
    return (x - lam).pow(2).mean(dim=1)


METRICS = {"wape": wape, "mae": mae, "mse": mse}


def compute_tfidf_features(x_all_np):
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    x_tfidf = tfidf.fit_transform(x_all_np).toarray()
    return x_tfidf, tfidf.idf_


def build_tfidf_fsce_graph(x_tfidf_tr, n_neighbors=N_NEIGHBORS):
    """Builds a TF-IDF weighted Cosine FSCE graph:
    - High-overlap categories (Dining) are down-weighted by IDF.
    - Distinctive categories (Station, Community, Nightlife, Health) are emphasized.
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(x_tfidf_tr)
    knn_dists, knn_idx = knn.kneighbors(x_tfidf_tr)
    graph, _, _ = fuzzy_simplicial_set(
        x_tfidf_tr, n_neighbors=n_neighbors, random_state=0, metric="cosine",
        knn_indices=knn_idx, knn_dists=knn_dists,
    )
    graph = graph.tocoo()

    edge_i = torch.from_numpy(graph.row).long()
    edge_j = torch.from_numpy(graph.col).long()
    edge_w = torch.from_numpy(graph.data).float()
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    return edge_i, edge_j, edge_w, a, b


def build_fsce_graph(x, n_neighbors=N_NEIGHBORS, metric="euclidean"):
    """建 plain（非 TF-IDF 加權）FSCE graph：x 是 log1p count 特徵。"""
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


def pcgrad_step(model, recon, weighted_fsce, eps=1e-12):
    mp = list(model.parameters())
    g_r = torch.autograd.grad(recon, mp, retain_graph=True, allow_unused=True)
    g_r = [torch.zeros_like(p) if g is None else g for p, g in zip(mp, g_r)]

    g_f = torch.autograd.grad(weighted_fsce, mp, retain_graph=True, allow_unused=True)
    g_f = [torch.zeros_like(p) if g is None else g for p, g in zip(mp, g_f)]

    flat_r = torch.cat([g.reshape(-1) for g in g_r])
    flat_f = torch.cat([g.reshape(-1) for g in g_f])
    dot = (flat_r * flat_f).sum()
    coef = dot / flat_r.pow(2).sum().clamp_min(eps) if dot < 0 else None

    for p, gr, gf in zip(mp, g_r, g_f):
        p.grad = gr + (gf - coef * gr if coef is not None else gf)
