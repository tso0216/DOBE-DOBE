import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/.."))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from model.other.v2_deep_ae.cfg import VERSION  # noqa: E402
from common.dataset import result  # noqa: E402

LATENTS = result(VERSION, "latents.npz")
OUT = result(VERSION, "latent_plot_plain.png")

DOT = 4.0

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def main():
    z = np.load(LATENTS)["z"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z[:, 0], z[:, 1], s=DOT, c="#3a6ea5", linewidths=0,
               alpha=0.6, rasterized=True)
    ax.set_title(f"{VERSION} latent space（原始座標）", fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
