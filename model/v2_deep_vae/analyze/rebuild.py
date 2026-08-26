"""重建單一 patch：載入這個版本的 checkpoint，畫「真實 count vs 重建 λ」
長條圖，用 --n 指定 patch 編號（0 起算）。
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from model.v2_deep_vae.cfg import ANALYZE_FOLD, ANALYZE_METRIC, LATENT_DIM, VERSION, ckpt_path, latents_path  # noqa: E402
from model.v2_deep_vae.dataset import Patches  # noqa: E402
from model.v2_deep_vae.model import AE, METRICS, poisson_nll  # noqa: E402
from common.dataset import CAT_COLORS, CAT_ZH, N_CAT, PATCHES, result  # noqa: E402

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="patch 編號（0 起算）")
    ap.add_argument("--fold", type=int, default=ANALYZE_FOLD, help="fold 編號（1 起算）")
    ap.add_argument("--metric", default=ANALYZE_METRIC,
                    help="要看哪份 checkpoint（mae/mse/wape/deviance）")
    args = ap.parse_args()
    n, fold, metric = args.n, args.fold, args.metric
    metric_fn = METRICS[metric]

    data = Patches(PATCHES)
    assert 0 <= n < data.n, f"--n 要在 0~{data.n - 1}"

    model = AE(LATENT_DIM)
    model.load_state_dict(torch.load(ckpt_path(fold, metric), map_location="cpu"))
    model.eval()

    idx = torch.tensor([n])
    x = data.agg(idx)
    with torch.no_grad():
        z, log_lam, _, _ = model(x)
    cnt = x[0].numpy()
    lam = torch.exp(log_lam)[0].numpy()
    mv = metric_fn(log_lam, x).item()
    nll = poisson_nll(log_lam, x).item()

    err = np.load(latents_path(fold, metric))["err"]
    pct = (err < mv).mean() * 100

    print(f"fold {fold}／ckpt[{metric}]")
    print(f"patch {n}：POI {int(data.n_poi[n])}  "
          f"({float(data.lat[n]):.5f}, {float(data.lon[n]):.5f})")
    print(f"latent z = ({z[0, 0]:+.3f}, {z[0, 1]:+.3f})")
    print(f"{metric.upper()} = {mv:.6f}（全體中位數 {np.median(err):.6f}，"
          f"此 patch 排在第 {pct:.1f} 百分位）")
    print(f"NLL = {nll:.6f}（省略 log(y!) 常數項，可能為負）")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    idx_c = np.arange(N_CAT)
    w = 0.38
    ax.bar(idx_c - w / 2, cnt, width=w, color=CAT_COLORS, alpha=0.55,
           label="真實 count")
    ax.bar(idx_c + w / 2, np.round(lam), width=w, color=CAT_COLORS, alpha=0.95,
           hatch="//", edgecolor="white", label="重建 λ")
    ax.set_xticks(idx_c)
    ax.set_xticklabels(CAT_ZH, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("個數", fontsize=9)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15, linewidth=0.5, axis="y")
    for s in ax.spines.values():
        s.set_alpha(0.3)
    ax.set_title(f"{VERSION} fold{fold} ckpt[{metric}] patch {n}"
                 f"（{metric.upper()} {mv:.4f}，POI {int(data.n_poi[n])} 個）", fontsize=11)

    fig.tight_layout()
    out = result(VERSION, f"rebuild_fold{fold}_{metric}.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"已存 {out}")


main()
