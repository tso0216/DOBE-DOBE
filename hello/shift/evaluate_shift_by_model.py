"""比較 PCA、AE、VAE、ours 這 4 個 seed=0 模型：每個 POI 類別各加入不同數量後，
latent 平均偏移多少。偏移＝||encode(x+shift) - encode(x)||，對 test split 取平均。
輸出：hello/shift_by_model.png（每個 POI 類別一張長條子圖）、hello/shift_by_model.csv。
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.dataset import PATCHES, CATEGORIES, CAT_ZH, N_CAT, make_split, result  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "model", "v2_deep_ae"))
from dataset import Patches  # noqa: E402

SEED = 0
AMOUNTS = [1, 3, 5]
LATENT_DIM = 2

MODELS = ["PCA", "AE", "VAE", "ours"]
VERSIONS = {"AE": "v2_deep_ae", "VAE": "v2_deep_vae", "ours": "v2_ddae_base"}
COLORS = {"PCA": "#4363d8", "AE": "#3cb44b", "VAE": "#f58231", "ours": "#e6194b"}


def load_nn_encoders():
    """載入 AE / VAE / ours 三個 seed=0 checkpoint，回傳 {model名: encode函式}。
    encode 函式吃 (B,N_CAT) count tensor，回傳 (B,LATENT_DIM) numpy latent。
    """
    encoders = {}
    for name, version in VERSIONS.items():
        # cfg / model / api 是各版本共用的模組名，載完一版就要清掉快取，
        # 否則下一版 import 會拿到前一版的類別，跟 checkpoint 對不上。
        for mod in ("api", "cfg", "model"):
            sys.modules.pop(mod, None)
        version_dir = os.path.join(ROOT, "model", version)
        sys.path.insert(0, version_dir)
        import api  # noqa
        ckpt = result(version, f"model_weight/ae_seed{SEED}.pt")
        model = api.load_model(ckpt=ckpt)

        if name == "VAE":
            def encode(x, model=model):
                # 用分布均值而非取樣值，避免 reparameterize 的抽樣雜訊蓋過偏移訊號
                with torch.no_grad():
                    h = model.encoder(x)
                    return model.to_mu(h).numpy()
        else:
            def encode(x, model=model):
                with torch.no_grad():
                    return model.encode(x).numpy()

        encoders[name] = encode
        sys.path.remove(version_dir)
    return encoders


def load_pca_encoder(data, x):
    """在 seed=0 的 train split 上配 PCA，回傳 encode 函式（吃 tensor，回傳 numpy latent）。"""
    train_idx, _, _ = make_split(data.lat, data.lon, seed=SEED)
    x_log = np.log1p(x.numpy())
    pca = PCA(n_components=LATENT_DIM, random_state=SEED).fit(x_log[train_idx.numpy()])

    def encode(x_t):
        return pca.transform(np.log1p(x_t.numpy()))
    return encode


def main():
    data = Patches(PATCHES)
    _, _, test_idx = make_split(data.lat, data.lon, seed=SEED)
    x_test = data.agg(test_idx)

    encoders = load_nn_encoders()
    encoders["PCA"] = load_pca_encoder(data, data.agg(torch.arange(data.n)))

    z0 = {name: enc(x_test) for name, enc in encoders.items()}

    offsets = {name: np.zeros((N_CAT, len(AMOUNTS))) for name in MODELS}
    for c in range(N_CAT):
        for j, a in enumerate(AMOUNTS):
            x_shift = x_test.clone()
            x_shift[:, c] += a
            for name, enc in encoders.items():
                d = np.linalg.norm(enc(x_shift) - z0[name], axis=1).mean()
                offsets[name][c, j] = d

    out_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(out_dir, "shift_by_model.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "model", "amount", "avg_offset"])
        for c in range(N_CAT):
            for name in MODELS:
                for j, a in enumerate(AMOUNTS):
                    writer.writerow([CATEGORIES[c], name, a, offsets[name][c, j]])
    print(f"已存 {os.path.join(out_dir, 'shift_by_model.csv')}")

    n_groups, bar_w = len(AMOUNTS), 0.8 / len(MODELS)
    x_base = np.arange(n_groups)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
    for c, ax in enumerate(axes.flat):
        for i, name in enumerate(MODELS):
            ax.bar(x_base + i * bar_w, offsets[name][c], width=bar_w,
                   label=name, color=COLORS[name])
        ax.set_title(CAT_ZH[c], fontsize=10)
        ax.set_xticks(x_base + bar_w * (len(MODELS) - 1) / 2)
        ax.set_xticklabels(AMOUNTS)
        ax.grid(alpha=0.2, axis="y")
    for ax in axes[-1]:
        ax.set_xlabel("加入量")
    for ax in axes[:, 0]:
        ax.set_ylabel("latent 平均偏移")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"seed={SEED}：各 POI 類別加入不同數量後的 latent 偏移（test split）")
    fig.tight_layout()

    out = os.path.join(out_dir, "shift_by_model.png")
    fig.savefig(out, dpi=150)
    print(f"已存 {out}")


main()
