import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from cfg import (BATCH, CKPT, EPOCHS, LAMBDA_KL, LATENT_DIM, LR, METRIC, OUT,
                 SEED, VERSION, WEIGHT_DECAY, device, open_log)
from dataset import Patches
from model import AE, METRICS, kl_divergence, poisson_nll
from common.dataset import PATCHES, make_split

metric_fn = METRICS[METRIC]


def evaluate(model, data, idx):
    """model：AE。data：Patches。idx：patch 編號的 tensor。回傳這批 patch 的平均 cfg.METRIC 指標。"""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            x = data.agg(idx[i:i + BATCH]).to(device)
            _, log_lam, _, _ = model(x)
            out.append(metric_fn(log_lam, x))
    return torch.cat(out).mean().item()


def run(data, train_idx, val_idx, test_idx, log):
    torch.manual_seed(SEED)
    model = AE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_metric, best_epoch, best_state = float("inf"), -1, None

    for epoch in range(EPOCHS):
        try:
            model.train()
            g = torch.Generator().manual_seed(SEED + epoch)
            perm = train_idx[torch.randperm(len(train_idx), generator=g)]
            total, total_kl = 0.0, 0.0
            for i in range(0, len(perm), BATCH):
                batch = perm[i:i + BATCH]
                x = data.agg(batch).to(device)
                _, log_lam, mu, logvar = model(x)
                recon = poisson_nll(log_lam, x).mean()
                kld = kl_divergence(mu, logvar).mean()

                loss = recon + LAMBDA_KL * kld
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += recon.item() * len(batch)
                total_kl += kld.item() * len(batch)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_metric = evaluate(model, data, val_idx)   # 驗證用乾淨輸入
                if val_metric < best_metric:
                    best_metric, best_epoch = val_metric, epoch + 1
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                log(f"epoch {epoch + 1:4d} | train_nll {total / len(perm):.5f} | "
                    f"train_kl {total_kl / len(perm):.5f} | val_{METRIC} {val_metric:.5f}")
        except KeyboardInterrupt:
            log(f"\n[中斷] epoch {epoch + 1} 收到 Ctrl-C，用目前最佳模型存 checkpoint 跟 latent")
            break

    if best_state is not None:   # 用 val 最好的那版，不是最後一版
        model.load_state_dict(best_state)
        log(f"\n最佳 checkpoint：epoch {best_epoch}，val_{METRIC} {best_metric:.5f}")
    torch.save(model.state_dict(), CKPT)

    log(f"test_{METRIC} {evaluate(model, data, test_idx):.5f}"
        f"（{len(test_idx)} 個 patch，全程未參與訓練與選 checkpoint）")

    model.eval()
    zs, errs = [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x = data.agg(idx).to(device)
            z, log_lam, _, _ = model(x)
            zs.append(z.cpu())
            errs.append(metric_fn(log_lam, x).cpu())
    return torch.cat(zs).numpy(), torch.cat(errs).numpy()


def main():
    log = open_log(VERSION, {
        "EPOCHS": EPOCHS,
        "SEED": SEED,
        "METRIC": METRIC,
        "LAMBDA_KL": LAMBDA_KL,
    })

    data = Patches(PATCHES)
    train_idx, val_idx, test_idx = make_split(data.lat, data.lon, seed=SEED)
    log(f"split：train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}\n")

    z, err = run(data, train_idx, val_idx, test_idx, log)
    split = np.zeros(data.n, dtype=np.int8)   # 0=train 1=val 2=test
    split[val_idx.numpy()], split[test_idx.numpy()] = 1, 2
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err,
             split=split)
    log(f"done")


main()
