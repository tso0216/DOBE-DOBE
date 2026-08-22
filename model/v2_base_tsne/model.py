import os
import sys

import numpy as np
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


def _pairwise_sq(x):
    s = (x ** 2).sum(axis=1)
    d2 = s[:, None] + s[None, :] - 2.0 * (x @ x.T)
    np.fill_diagonal(d2, 0.0)
    return np.maximum(d2, 0.0)


def build_tsne_p(x, perplexity=30.0, norm="row", tol=1e-5, max_iter=100):
    """x：(N,D) 高維特徵（外面已 log1p）。perplexity：每點的有效鄰居數。
    norm："row" 回傳條件機率 P(j|i)（每列和 1），"joint" 回傳對稱聯合機率（總和 1）。
    回傳 (N,N) 的 torch.float 矩陣，對角為 0。
    """
    n = len(x)
    d2 = _pairwise_sq(np.asarray(x, dtype=np.float64))
    log_u = np.log(perplexity)
    p_cond = np.zeros((n, n))
    off = ~np.eye(n, dtype=bool)

    for i in range(n):
        di = d2[i][off[i]]                      # 排掉自己那一格
        beta, lo, hi = 1.0, -np.inf, np.inf
        for _ in range(max_iter):
            p = np.exp(-di * beta)
            ssum = p.sum()
            if ssum < 1e-12:                    # beta 太大導致全部下溢，退回去
                h = 0.0
            else:
                h = np.log(ssum) + beta * (di * p).sum() / ssum
                p = p / ssum
            diff = h - log_u                    # h 是 Shannon entropy，目標是 log(perplexity)
            if abs(diff) < tol:
                break
            if diff > 0:                        # entropy 太大 → 鄰域太寬 → 加大 beta
                lo = beta
                beta = beta * 2.0 if hi == np.inf else (beta + hi) / 2.0
            else:
                hi = beta
                beta = beta / 2.0 if lo == -np.inf else (beta + lo) / 2.0
        p_cond[i][off[i]] = p / max(p.sum(), 1e-12)   # 保險：下溢分支可能沒正規化過

    if norm == "row":
        # 每個點自己一列、和為 1，梯度貢獻是 O(1)；joint 模式下每點只佔 1/N 的質量，
        # 吸引力被稀釋成 1/N，KL 要磨很久才降得下來
        return torch.from_numpy(p_cond).float()
    p = p_cond + p_cond.T
    p /= max(p.sum(), 1e-12)
    return torch.from_numpy(p).float()


def renorm_p(p, norm="row", eps=1e-12):
    """p：(B,B) 從完整 P 取出的子矩陣。norm："row" 逐列重新正規化，"joint" 除以總和。
    回傳同形狀、重新正規化後的機率矩陣。
    """
    if norm == "row":
        return p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    return p / p.sum().clamp_min(eps)


def tsne_loss(z, p, norm="row", scale=True, log_s=None, gamma=1.0, eps=1e-12):
    """z：(B,latent) 這批的 latent。p：(B,B) 這批重新正規化後的機率矩陣。
    norm："row" 算逐列 KL 再取平均，"joint" 算整體 KL。
    scale：True 時先把 z 等向縮放成單位尺度，讓 t-SNE 只決定相對佈局、不決定 latent 尺度。
    log_s：純量 tensor，Q 專用的可學 log 尺度；t-SNE 靠撐大尺度降 KL，交給這個參數去撐，
           decoder 看到的 z 就不會被推著漂。
    gamma：排斥項的權重，1.0 是標準 KL，<1 把「推開非鄰居」那一半打折。
    回傳純量 KL(P‖Q)，Q 用自由度 1 的 Student-t。
    """
    if scale:
        # 排斥項 -log ΣQ 無界，會一直把點往外推；除掉尺度後 decoder 看到的 latent 分佈才穩
        z = z / z.std(dim=0).mean().clamp_min(eps)
        if log_s is not None:
            z = z * log_s.exp()
    d2 = torch.cdist(z, z).pow(2)
    num = (1.0 + d2).reciprocal()
    num = num - torch.diag_embed(torch.diagonal(num))   # 對角歸零，不用 in-place 免得斷梯度
    # KL 拆成吸引項與排斥項：-Σ P log Q = -Σ P log num + log Σ num（因為 ΣP=1），
    # 後面那個 log Σ num 就是把所有非鄰居往外推的來源，用 gamma 控制它的力道
    ent = (p * p.clamp_min(eps).log()).sum(dim=1)      # Σ P log P，常數項，留著讓數值可比
    attract = -(p * num.clamp_min(eps).log()).sum(dim=1)
    if norm == "row":
        repulse = num.sum(dim=1).clamp_min(eps).log()
        return (ent + attract + gamma * repulse).mean()
    repulse = num.sum().clamp_min(eps).log()
    return ent.sum() + attract.sum() + gamma * repulse


if __name__ == '__main__':
    model = AE(latent_dim=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
