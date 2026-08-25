"""計算 AE+entropy 在不同訓練進度快照下，「Travel and Transportation」加入 POI 前後的 2 維 latent 座標"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import CATEGORIES, PATCHES  # noqa: E402
from model.v3_ddae_tfidf.cfg import LATENT_DIM, SEED, device  # noqa: E402
from model.v3_ddae_tfidf.dataset import Patches  # noqa: E402
from model.v3_ddae_tfidf.model import AE  # noqa: E402

CKPT_DIR = os.path.join(HERE, "..", "model", "entropy_progress_ckpt")
SNAPSHOT_PERCENTS = [10, 50, 100]
SHIFT_CATEGORY = 'Travel and Transportation'
SHIFT_AMOUNT = 5


def ckpt_path(percent):
    return os.path.join(CKPT_DIR, f"epoch_pct{percent}_seed{SEED}.pt")


def encode_before_after(percent, x_before, x_after):
    """percent：訓練進度，對應 ckpt_path 存的權重檔。
    x_before/x_after：(N, N_CAT) 平移前後的 count tensor，一一對應。
    回傳 (z_before, z_after)，各為 (N, 2) numpy array。
    """
    model = AE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(ckpt_path(percent), map_location=device))
    model.eval()
    with torch.no_grad():
        z_before = model.encode(x_before.to(device)).cpu().numpy()
        z_after = model.encode(x_after.to(device)).cpu().numpy()
    return z_before, z_after


def main():
    missing = [p for p in SNAPSHOT_PERCENTS if not os.path.exists(ckpt_path(p))]
    if missing:
        raise FileNotFoundError(f"找不到進度 {missing} 的權重，請確認 {CKPT_DIR} 底下的快照完整")

    data = Patches(PATCHES)
    cat_idx = CATEGORIES.index(SHIFT_CATEGORY)
    x_before = data.agg(torch.arange(data.n))
    x_after = x_before.clone()
    x_after[:, cat_idx] += SHIFT_AMOUNT

    out = os.path.join(HERE, "data.csv")
    with open(out, "w") as f:
        f.write("percent,sample_id,z1_before,z2_before,z1_after,z2_after\n")
        for percent in sorted(SNAPSHOT_PERCENTS):
            z_before, z_after = encode_before_after(percent, x_before, x_after)
            for sample_id, ((zb1, zb2), (za1, za2)) in enumerate(zip(z_before, z_after)):
                f.write(f"{percent},{sample_id},{zb1},{zb2},{za1},{za2}\n")
            print(f"訓練進度 {percent}% 已編碼完成")
    print(f"已存 {out}")


main()
