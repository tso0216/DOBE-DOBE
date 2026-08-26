import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.other.v2_ddae_base.cfg import CKPT, LATENT_DIM, METRIC
from model import AE, METRICS

metric_fn = METRICS[METRIC]


def load_model(ckpt=CKPT, device="cpu"):
    model = AE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def encode(model, x):
    """model：AE。x：(B,N_CAT) 的 count 向量。回傳 (B,latent_dim) 的 latent z。"""
    with torch.no_grad():
        return model.encode(x)


def rebuild(model, x):
    """model：AE。x：(B,N_CAT) 的 count 向量。回傳 (B,N_CAT) 的重建 log λ。"""
    with torch.no_grad():
        _, log_lam = model(x)
    return log_lam


def evaluate(model, data, idx, batch=256):
    """model：AE。data：Patches。idx：要評估的 patch 編號 tensor。batch：每批大小。
    回傳這批 idx 的平均 cfg.METRIC 指標。
    """
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            x = data.agg(idx[i:i + batch])
            _, log_lam = model(x)
            out.append(metric_fn(log_lam, x))
    return torch.cat(out).mean().item()


def test_mse(model, data, idx, batch=256):
    model.eval()
    diffs, counts = [], []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            x = data.agg(idx[i:i + batch])
            _, log_lam = model(x)
            diffs.append(torch.exp(log_lam) - x)
            counts.append(x)
    diff = torch.cat(diffs)
    x = torch.cat(counts)
    mse = diff.pow(2).mean().item()
    mae = diff.abs().mean().item()
    mask = x > 0
    mape = (diff[mask].abs() / x[mask]).mean().item() * 100
    return mse, mae, mape
