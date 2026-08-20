import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfg import CKPT, LATENT_DIM
from model import AE, poisson_deviance


def load_model(ckpt=CKPT, device="cpu"):
    """ckpt：checkpoint 檔案路徑。device：載入到哪個裝置。
    回傳已載入權重、eval() 模式的 AE。
    """
    model = AE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
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


def evaluate(model, data, idx, batch=256):
    """model：AE。data：Patches。idx：要評估的 patch 編號 tensor。batch：每批大小。
    回傳這批 idx 的平均 poisson deviance。
    """
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            x = data.agg(idx[i:i + batch])
            _, log_lam, _, _ = model(x)
            out.append(poisson_deviance(log_lam, x))
    return torch.cat(out).mean().item()
