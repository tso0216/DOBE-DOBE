import os
import sys

import torch
import torch.nn as nn

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
        )
        self.to_mu = nn.Linear(hidden, latent_dim)        # latent 前不接 norm/激活
        self.to_logvar = nn.Linear(hidden, latent_dim)     # latent 前不接 norm/激活
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

    def reparameterize(self, mu, logvar):
        """mu、logvar：(B,latent_dim) latent 分布的均值與 log 變異數。
        回傳取樣後的 (B,latent_dim) latent z。
        """
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, x):
        """x：(B,N_CAT) 的 count 向量。回傳取樣後的 (B,latent_dim) latent z。"""
        h = self.encoder(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        return self.reparameterize(mu, logvar)

    def forward(self, x):
        """x：(B,N_CAT) 的 count 向量。
        回傳 (z, log_lam, mu, logvar)：取樣的 latent、重建的 log λ、
        latent 分布的均值與 log 變異數。
        """
        h = self.encoder(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, self.decoder(z), mu, logvar


def poisson_nll(log_lam, x):
    """log_lam：(B,N_CAT) 重建的 log λ。x：(B,N_CAT) 乾淨 count。回傳 (B,) 的 NLL。"""
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


def kl_divergence(mu, logvar):
    """mu、logvar：(B,latent_dim) latent 分布的均值與 log 變異數。
    回傳 (B,) 對標準常態 N(0,I) 的 KL 散度。
    """
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)


if __name__ == '__main__':
    model = AE(latent_dim=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
