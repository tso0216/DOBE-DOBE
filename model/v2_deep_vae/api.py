import os
import sys

import torch

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from model.v2_deep_vae.cfg import ANALYZE_FOLD, ANALYZE_METRIC, LATENT_DIM, ckpt_path
from model.v2_deep_vae.model import AE, METRICS


def load_model(fold=ANALYZE_FOLD, metric=ANALYZE_METRIC, device="cpu"):
    """fold：fold 編號（1 起算）。metric：要載入哪份 checkpoint（mae/mse/wape/deviance）。
    device：載入到哪個裝置。回傳已載入權重、eval() 模式的 AE。
    """
    model = AE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(ckpt_path(fold, metric), map_location=device))
    model.eval()
    return model


def encode(model, x):
    """model：AE。x：(B,N_CAT) 的 count 向量。回傳取樣後的 (B,latent_dim) latent z。"""
    with torch.no_grad():
        return model.encode(x)


def rebuild(model, x):
    """model：AE。x：(B,N_CAT) 的 count 向量。回傳 (B,N_CAT) 的重建 log λ。"""
    with torch.no_grad():
        _, log_lam, _, _ = model(x)
    return log_lam


def evaluate(model, data, idx, metric=ANALYZE_METRIC, batch=256):
    """model：AE。data：Patches。idx：要評估的 patch 編號 tensor。metric：指標名。batch：每批大小。
    回傳這批 idx 的平均指標值。
    """
    metric_fn = METRICS[metric]
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            x = data.agg(idx[i:i + batch])
            _, log_lam, _, _ = model(x)
            out.append(metric_fn(log_lam, x))
    return torch.cat(out).mean().item()
