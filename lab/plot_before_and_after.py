import os
import sys

import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)   # 直接以模組名匯入，避開根目錄同名的 lab.py
from common.dataset import CAT_ZH, CATEGORIES, PATCHES, make_split  # noqa: E402
from run_mae import VAE, load_model  # noqa: E402
from model.v2_ddae_base.dataset import Patches  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

weight_dir = 'model_weight'
model = {
    'ae.pt': 'AE',
    'ae_fsce.pt': 'AE+fsce',
    'ae_tfidf.pt': 'AE+fsce+tfidf',
    'dae.pt': 'DAE',
    'dae_fsce.pt': 'DAE+fsce',
    'dae_tfidf.pt': 'DAE+fsce+tfidf (Ours)'
}
category = 'Travel and Transportation'
amount = 5
SEED = 0

COLOR_BEFORE = "#4363d8"
COLOR_AFTER = "#f58231"


def rebuild(m, x, batch=256):
    """m：AE 或 VAE。x：(B, N_CAT) count tensor。回傳 (B, N_CAT) 重建的期望 count（decoder 輸出的 log_lam 取 exp，VAE 用 mu 不取樣）。"""
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch]
            if isinstance(m, VAE):
                h = m.encoder(xb)
                outs.append(torch.exp(m.decoder(m.to_mu(h))))
            else:
                outs.append(torch.exp(m(xb)[1]))
    return torch.cat(outs)


def before_after(m, x, cat_idx, add):
    """m：AE 或 VAE。x：(B, N_CAT) count tensor。cat_idx：要加 POI 的類別索引。add：加入量。
    回傳 (before, after)，各為 (N_CAT,) 的平均重建 count；before 用原始 x，after 用第 cat_idx 類加了 add 個 POI 的 x。
    """
    x_shift = x.clone()
    x_shift[:, cat_idx] += add
    return rebuild(m, x).mean(0), rebuild(m, x_shift).mean(0)


def main():
    data = Patches(PATCHES)
    _, _, test_idx = make_split(data.lat, data.lon, seed=SEED)
    x_test = data.agg(test_idx)
    cat_idx = CATEGORIES.index(category)
    x_mean = x_test.mean(0)

    names = list(model.keys())
    ncols = 3
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows),
                             squeeze=False, sharey=True)

    xs = range(len(CAT_ZH))
    for ax, name in zip(axes.flat, names):
        m = load_model(os.path.join(ROOT, weight_dir, name))
        before, after = before_after(m, x_test, cat_idx, amount)
        ax.bar([i - 0.2 for i in xs], before.tolist(), width=0.4,
               label="before", color=COLOR_BEFORE)
        ax.bar([i + 0.2 for i in xs], after.tolist(), width=0.4,
               label="after", color=COLOR_AFTER)
        ax.scatter(list(xs), x_mean.tolist(), marker="_", s=200,
                   color="black", zorder=3, label="原始輸入")
        delta = (after[cat_idx] - before[cat_idx]).item()
        ax.set_title(f"{model[name]}（目標類別 Δ={delta:+.2f}）", fontsize=11)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(CAT_ZH, rotation=45, ha="right", fontsize=8)
        ax.get_xticklabels()[cat_idx].set_color("#d62728")
        ax.grid(alpha=0.2, axis="y")

    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("平均重建 count")
    axes.flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"各模型重建結果：在「{CAT_ZH[cat_idx]}」加入 {amount} 個 POI 前後對照"
                 f"（test set 平均，n={len(x_test)}）")
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "plot_before_and_after.png")
    fig.savefig(out, dpi=150)
    print(f"已存 {out}")


main()
