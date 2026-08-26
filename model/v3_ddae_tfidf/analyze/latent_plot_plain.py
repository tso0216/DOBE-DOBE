import argparse
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from model.v3_ddae_tfidf.cfg import (ANALYZE_FOLD, ANALYZE_METRIC, VERSION,  # noqa: E402
                                     latents_path)
from common.dataset import result  # noqa: E402

DOT = 4.0

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=ANALYZE_FOLD, help="fold 編號（1 起算）")
    ap.add_argument("--metric", default=ANALYZE_METRIC,
                    help="要看哪份 checkpoint（mae/mse/wape/deviance）")
    args = ap.parse_args()
    fold, metric = args.fold, args.metric

    z = np.load(latents_path(fold, metric))["z"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z[:, 0], z[:, 1], s=DOT, c="#3a6ea5", linewidths=0,
               alpha=0.6, rasterized=True)
    ax.set_title(f"{VERSION} fold{fold} ckpt[{metric}] latent space（原始座標）", fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    out = result(VERSION, f"latent_plot_plain_fold{fold}_{metric}.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"已存 {out}")


main()
