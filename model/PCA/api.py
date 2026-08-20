import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfg import LATENT_DIM


def fit(data, train_idx, latent_dim=LATENT_DIM):
    """data：Patches。train_idx：只拿來 fit 的 patch 編號 tensor。
    latent_dim：壓縮後的維度。回傳已 fit 好的 sklearn PCA。
    """
    x = data.agg(train_idx).numpy()
    return PCA(n_components=latent_dim).fit(x)


def encode(model, x):
    """model：已 fit 的 PCA。x：(B,N_CAT) 的 count 向量。回傳 (B,latent_dim) 的 z。"""
    return torch.from_numpy(model.transform(x.numpy())).float()


def rebuild(model, x):
    """model：已 fit 的 PCA。x：(B,N_CAT) 的 count 向量。
    回傳 (B,N_CAT) 的重建 count（線性反投影，clip 到非負）。
    """
    recon = model.inverse_transform(model.transform(x.numpy()))
    return torch.from_numpy(np.clip(recon, 0, None)).float()


def test_mse(model, data, idx, batch=256):
    """model：已 fit 的 PCA。data：Patches。idx：要評估的 patch 編號 tensor（例如
    test split）。batch：每批大小。回傳這批 idx 的 (mse, mae, mape) 三個 float，
    mape 只算真實 count > 0 的格子。
    """
    diffs, counts = [], []
    for i in range(0, len(idx), batch):
        x = data.agg(idx[i:i + batch])
        diffs.append(rebuild(model, x) - x)
        counts.append(x)
    diff = torch.cat(diffs)
    x = torch.cat(counts)
    mse = diff.pow(2).mean().item()
    mae = diff.abs().mean().item()
    mask = x > 0
    mape = (diff[mask].abs() / x[mask]).mean().item() * 100
    return mse, mae, mape
