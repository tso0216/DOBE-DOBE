"""訓練 AE / DAE 各自關掉 FSCE 的版本（seed=0），存到 hello/shift/model_weight/，
不動 model/<version>/result 底下的官方 checkpoint 與 log。
AE+fsce、DAE+fsce 直接複用官方已訓練好的 model/<version>/result/model_weight/ae_seed0.pt。
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from common.dataset import PATCHES, make_split  # noqa: E402

SEED = 0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_version_modules(version, epochs):
    """version：model/ 底下的版本目錄名。epochs：這次訓練的 epoch 數。清掉
    cfg/model/dataset 的模組快取後重新 import，避免拿到別的版本殘留在 sys.modules
    裡的類別。回傳 (cfg, model模組, dataset模組)。
    """
    for mod in ("cfg", "model", "dataset"):
        sys.modules.pop(mod, None)
    version_dir = os.path.join(ROOT, "model", version)
    sys.path.insert(0, version_dir)
    os.environ["SEED"] = str(SEED)
    os.environ["EPOCHS"] = str(epochs)
    import cfg
    import model as model_mod
    import dataset as dataset_mod
    sys.path.remove(version_dir)
    return cfg, model_mod, dataset_mod


def train_one(version, denoise, out_path, epochs):
    """version：model/ 底下的版本目錄名。denoise：是否對輸入加噪（DAE 用）。
    out_path：訓練完的 state_dict 存檔路徑。epochs：訓練 epoch 數。FSCE 一律關閉。
    """
    cfg, model_mod, dataset_mod = load_version_modules(version, epochs)
    AE, poisson_nll, poisson_deviance = model_mod.AE, model_mod.poisson_nll, model_mod.poisson_deviance
    Patches = dataset_mod.Patches
    corrupt = dataset_mod.corrupt if denoise else None

    data = Patches(PATCHES)
    train_idx, val_idx, test_idx = make_split(data.lat, data.lon, seed=SEED)

    torch.manual_seed(SEED)
    model = AE(cfg.LATENT_DIM).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    best_dev, best_state = float("inf"), None

    for epoch in range(epochs):
        model.train()
        g = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        for i in range(0, len(perm), cfg.BATCH):
            batch = perm[i:i + cfg.BATCH]
            x = data.agg(batch)
            x_in = corrupt(x, cfg.NOISE_P, cfg.NOISE_MODE, generator=g) if denoise else x
            x, x_in = x.to(cfg.device), x_in.to(cfg.device)
            _, log_lam = model(x_in)
            loss = poisson_nll(log_lam, x).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                x = data.agg(val_idx).to(cfg.device)
                _, log_lam = model(x)
                dev = poisson_deviance(log_lam, x).mean().item()
            if dev < best_dev:
                best_dev = dev
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{version} denoise={denoise}] epoch {epoch + 1:4d} val_dev {dev:.5f}")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_path)
    print(f"[{version} denoise={denoise}] 已存 {out_path}（best val_dev {best_dev:.5f}）")


def main():
    ae_path = os.path.join(OUT_DIR, "model_weight", "ae_nofsce_seed0.pt")
    if not os.path.exists(ae_path):
        train_one("v2_deep_ae", denoise=False, out_path=ae_path, epochs=300)
    train_one("v2_ddae_base", denoise=True,
              out_path=os.path.join(OUT_DIR, "model_weight", "dae_nofsce_seed0.pt"), epochs=2000)


main()
