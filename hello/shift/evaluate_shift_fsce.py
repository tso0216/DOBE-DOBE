"""比較 AE、AE+FSCE、DAE、DAE+FSCE 這 4 個 seed=0 模型：每個 POI 類別各加入不同
數量後，latent 平均偏移多少。偏移＝||encode(x+shift) - encode(x)||，對 test split 取平均。
4 個 checkpoint 都放在 hello/shift/model_weight/：
  ae_nofsce_seed0.pt / dae_nofsce_seed0.pt 由 train_variants.py 訓練；
  ae_fsce_seed0.pt / dae_fsce_seed0.pt 是官方 model/<version>/result/model_weight/ae_seed0.pt 的副本
  （dae_fsce_seed0.pt 即先前分析裡的「ours」）。
輸出：hello/shift/shift_fsce.png（每個 POI 類別一張長條子圖）、hello/shift/shift_fsce.csv。
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from common.dataset import PATCHES, CATEGORIES, CAT_ZH, N_CAT, make_split  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "model", "v2_deep_ae"))
from dataset import Patches  # noqa: E402

SEED = 0
AMOUNTS = [1, 3, 5]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = ["AE", "AE+FSCE", "DAE", "DAE+FSCE"]
COLORS = {"AE": "#4363d8", "AE+FSCE": "#3cb44b", "DAE": "#f58231", "DAE+FSCE": "#e6194b"}

WEIGHT_DIR = os.path.join(OUT_DIR, "model_weight")

# (版本, checkpoint 路徑)——4 個 checkpoint 都是 hello/shift/model_weight/ 底下的檔案
CKPTS = {
    "AE": ("v2_deep_ae", os.path.join(WEIGHT_DIR, "ae_nofsce_seed0.pt")),
    "AE+FSCE": ("v2_deep_ae", os.path.join(WEIGHT_DIR, "ae_fsce_seed0.pt")),
    "DAE": ("v2_ddae_base", os.path.join(WEIGHT_DIR, "dae_nofsce_seed0.pt")),
    "DAE+FSCE": ("v2_ddae_base", os.path.join(WEIGHT_DIR, "dae_fsce_seed0.pt")),
}


def load_encoders():
    """依 CKPTS 載入 4 個模型，回傳 {model名: encode函式}。
    encode 函式吃 (B,N_CAT) count tensor，回傳 (B,latent_dim) numpy latent。
    """
    encoders = {}
    for name, (version, ckpt) in CKPTS.items():
        # cfg / model / api 是各版本共用的模組名，載完一版就要清掉快取，
        # 否則下一版 import 會拿到前一版的類別，跟 checkpoint 對不上。
        for mod in ("api", "cfg", "model"):
            sys.modules.pop(mod, None)
        version_dir = os.path.join(ROOT, "model", version)
        sys.path.insert(0, version_dir)
        import api  # noqa
        model = api.load_model(ckpt=ckpt)

        def encode(x, model=model):
            with torch.no_grad():
                return model.encode(x).numpy()

        encoders[name] = encode
        sys.path.remove(version_dir)
    return encoders


def main():
    data = Patches(PATCHES)
    _, _, test_idx = make_split(data.lat, data.lon, seed=SEED)
    x_test = data.agg(test_idx)

    encoders = load_encoders()
    z0 = {name: enc(x_test) for name, enc in encoders.items()}

    offsets = {name: np.zeros((N_CAT, len(AMOUNTS))) for name in MODELS}
    for c in range(N_CAT):
        for j, a in enumerate(AMOUNTS):
            x_shift = x_test.clone()
            x_shift[:, c] += a
            for name, enc in encoders.items():
                d = np.linalg.norm(enc(x_shift) - z0[name], axis=1).mean()
                offsets[name][c, j] = d

    with open(os.path.join(OUT_DIR, "shift_fsce.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "model", "amount", "avg_offset"])
        for c in range(N_CAT):
            for name in MODELS:
                for j, a in enumerate(AMOUNTS):
                    writer.writerow([CATEGORIES[c], name, a, offsets[name][c, j]])
    print(f"已存 {os.path.join(OUT_DIR, 'shift_fsce.csv')}")

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
    fig.suptitle(f"seed={SEED}：AE/DAE 有無 FSCE，各 POI 類別加入不同數量後的 latent 偏移（test split）")
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "shift_fsce.png")
    fig.savefig(out, dpi=150)
    print(f"已存 {out}")


main()
