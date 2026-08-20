"""比較 PCA、Deep AE、Deep VAE、DDAE_BASE 在各自驗證集上的重建誤差
（Poisson deviance）分布。

PCA 沒有訓練好的 checkpoint，用同一份 patches 現場配一個 2 維 PCA 當基準；
其餘三個直接讀各自 model/<version>/result/latents.npz 裡訓練時存好的
err（split==1 為驗證集），還沒訓練完的版本會被跳過並印警告。
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import xlogy
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.dirname(script_dir)
root = os.path.dirname(out_dir)
os.makedirs(out_dir, exist_ok=True)

sys.path.insert(0, root)
from common.dataset import PATCHES, result  # noqa: E402

sys.path.insert(0, os.path.join(root, "model", "v2_deep_ae"))
from dataset import Patches, make_split  # noqa: E402

LATENT_DIM = 2
SEED = 0

NN_VERSIONS = [
    ("Deep AE", "v2_deep_ae"),
    ("Deep VAE", "v2_deep_vae"),
    ("DDAE_BASE", "v2_ddae_base"),
]

COLORS = ["#4363d8", "#3cb44b", "#f58231", "#e6194b"]


def poisson_deviance(x, log_lam):
    """x：(N,N_CAT) 乾淨 count。log_lam：(N,N_CAT) 重建的 log λ。
    回傳 (N,) 每個 patch 的 Poisson deviance。
    """
    lam = np.exp(log_lam)
    cell = 2 * (xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(axis=1)


def pca_val_err():
    """在 train split 上配一個 2 維 PCA，回傳驗證集每個 patch 的 deviance。

    回傳 (n_val,) 的 float 陣列。
    """
    data = Patches(PATCHES)
    x = data.agg(torch.arange(data.n)).numpy()
    train_idx, val_idx, _ = make_split(data.lat, data.lon, seed=SEED)
    train_idx, val_idx = train_idx.numpy(), val_idx.numpy()

    x_log = np.log1p(x)
    pca = PCA(n_components=LATENT_DIM, random_state=SEED).fit(x_log[train_idx])
    recon_log = pca.inverse_transform(pca.transform(x_log))
    lam = np.clip(np.expm1(recon_log), 1e-6, None)
    dev = poisson_deviance(x, np.log(lam))

    np.savez(os.path.join(out_dir, "1_pca_result.npz"),
              n_poi=data.n_poi, lat=data.lat, lon=data.lon,
              z=pca.transform(x_log), err=dev)
    return dev[val_idx]


def nn_val_err(version):
    """讀 model/<version>/result/latents.npz，回傳驗證集每個 patch 的 deviance。

    version：model/ 底下的版本目錄名。檔案不存在時回傳 None。
    """
    path = result(version, "latents.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as d:
        err, split = d["err"], d["split"]
    return err[split == 1]


def main():
    labels, groups = ["PCA"], [pca_val_err()]
    for name, version in NN_VERSIONS:
        err = nn_val_err(version)
        if err is None:
            print(f"[skip] {name}（{version}）還沒有 result/latents.npz，訓練完再跑一次")
            continue
        labels.append(name)
        groups.append(err)

    summary = {lab: float(np.mean(g)) for lab, g in zip(labels, groups)}
    np.savez(os.path.join(out_dir, "1_baseline_result.npz"), **summary)

    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot(groups, showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[i % len(COLORS)])
        body.set_alpha(0.55)
    ax.boxplot(groups, positions=range(1, len(groups) + 1), widths=0.15,
               showfliers=False, medianprops={"color": "black"})

    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Poisson deviance（驗證集，log scale）")
    ax.set_title("重建誤差比較")
    ax.grid(alpha=0.2, linewidth=0.5, axis="y")
    for i, g in enumerate(groups):
        ax.text(i + 1, np.max(g) * 1.05, f"mean {np.mean(g):.3f}\nn={len(g)}",
                ha="center", va="bottom", fontsize=8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.6)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_baseline_compare.png"), dpi=150)
    print(f"已存 {os.path.join(out_dir, '1_baseline_compare.png')}")
    for lab, v in summary.items():
        print(f"{lab:10s} mean_dev={v:.4f}")


main()
