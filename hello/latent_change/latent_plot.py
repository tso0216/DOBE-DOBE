"""畫 AE / AE+FSCE / DAE / DAE+FSCE 這 4 個 seed=0 模型的 latent space：
左欄是 test split 原始 latent，右欄是每筆 patch 的 category 類別加入 amount 個 POI 後的 latent。
輸出：與本檔同目錄的 latent_plot.png。
"""
import os
import sys

import matplotlib.pyplot as plt
import torch

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

category = 'Travel and Transportation'
amount = 5

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from common.dataset import PATCHES, CATEGORIES, make_split  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "model", "v2_deep_ae"))
from dataset import Patches  # noqa: E402

SEED = 0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.path.join(OUT_DIR, "model_weight")

MODELS = ["AE", "AE+FSCE", "DAE", "DAE+FSCE"]
COLORS = {"AE": "#4363d8", "AE+FSCE": "#3cb44b", "DAE": "#f58231", "DAE+FSCE": "#e6194b"}

# (版本, checkpoint 路徑)——4 個 checkpoint 都是本檔同目錄 model_weight/ 底下的檔案
CKPTS = {
    "AE": ("v2_deep_ae", os.path.join(WEIGHT_DIR, "ae_nofsce_seed0.pt")),
    "AE+FSCE": ("v2_deep_ae", os.path.join(WEIGHT_DIR, "ae_fsce_seed0.pt")),
    "DAE": ("v2_ddae_base", os.path.join(WEIGHT_DIR, "dae_nofsce_seed0.pt")),
    "DAE+FSCE": ("v2_ddae_base", os.path.join(WEIGHT_DIR, "dae_fsce_seed0.pt")),
}


def load_encoders():
    """依 CKPTS 載入 4 個模型，回傳 {model名: encode函式}。
    encode 函式吃 (B,N_CAT) count tensor，回傳 (B,2) numpy latent。
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
    c = CATEGORIES.index(category)

    data = Patches(PATCHES)
    _, _, test_idx = make_split(data.lat, data.lon, seed=SEED)
    x0 = data.agg(test_idx)
    x1 = x0.clone()
    x1[:, c] += amount

    encoders = load_encoders()

    fig, axes = plt.subplots(len(MODELS), 2, figsize=(8, 2.8 * len(MODELS)),
                              gridspec_kw={"wspace": 0.08, "hspace": 0.05})
    for row, name in enumerate(MODELS):
        z0 = encoders[name](x0)
        z1 = encoders[name](x1)
        ax_l, ax_r = axes[row]
        ax_l.scatter(z0[:, 0], z0[:, 1], s=6, alpha=0.5, color=COLORS[name])
        ax_r.scatter(z0[:, 0], z0[:, 1], s=6, alpha=0.15, color=COLORS[name])
        ax_r.scatter(z1[:, 0], z1[:, 1], s=6, alpha=0.5, color=COLORS[name])
        ax_r.quiver(z0[:, 0], z0[:, 1], z1[:, 0] - z0[:, 0], z1[:, 1] - z0[:, 1],
                    angles="xy", scale_units="xy", scale=1, color=COLORS[name],
                    alpha=0.25, width=0.003, headwidth=3, headlength=4)
        ax_l.set_ylabel(name, fontsize=12, fontweight="bold")
        ax_l.grid(alpha=0.2)
        ax_r.grid(alpha=0.2)
        ax_l.set_xticks([])
        ax_l.set_yticks([])
        ax_r.set_xticks([])
        ax_r.set_yticks([])
    axes[0, 0].set_title("latent before")
    axes[0, 1].set_title(f"latent after")
    fig.suptitle(f" latent space 位移")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = os.path.join(OUT_DIR, "latent_plot.png")
    fig.savefig(out, dpi=150)
    print(f"已存 {out}")


main()
