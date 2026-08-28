import argparse
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/.."))
from model.v3_ddae_tfidf.cfg import ANALYZE_FOLD, ANALYZE_METRIC, VERSION, latents_path  # noqa: E402

DOT = 4.0
HERE = os.path.dirname(os.path.abspath(__file__))

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def plot(z, title, out):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z[:, 0], z[:, 1], s=DOT, c="#3a6ea5", linewidths=0,
               alpha=0.6, rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"已存 {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=ANALYZE_FOLD, help="fold 編號（1 起算）")
    ap.add_argument("--metric", default=ANALYZE_METRIC,
                    help="要看哪份 checkpoint（mae/mse/wape/deviance）")
    args = ap.parse_args()
    fold, metric = args.fold, args.metric

    npz = np.load(latents_path(fold, metric))
    z, split = npz["z"], npz["split"]   # split: 0=train 1=val 2=test

    plot(z[split == 2], f"{VERSION} fold{fold} ckpt[{metric}] latent space（testset）",
         os.path.join(HERE, f"latent_plot_fold{fold}_{metric}_testset.png"))
    plot(z, f"{VERSION} fold{fold} ckpt[{metric}] latent space（all）",
         os.path.join(HERE, f"latent_plot_fold{fold}_{metric}_all.png"))


main()
