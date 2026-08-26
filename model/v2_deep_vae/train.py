import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# 直接跑這支檔案時 python 會把本目錄放進 sys.path，同目錄的 model.py 就會蓋掉 model/ package，
# 所以先把本目錄拿掉，只留 repo 根目錄
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, os.path.abspath(f"{_HERE}/../.."))
from model.v2_deep_vae.cfg import (BATCH, CKPT_METRICS, EPOCHS, LAMBDA_KL, LATENT_DIM, LR,
                 METRIC, N_SPLITS, SEED, TEST_FRAC, VERSION, WEIGHT_DECAY,
                 ckpt_path, device, latents_path, open_log)
from model.v2_deep_vae.dataset import Patches
from model.v2_deep_vae.model import AE, METRICS, kl_divergence, poisson_nll
from common.dataset import PATCHES, make_kfold_split


def evaluate(model, data, idx):
    """model：AE。data：Patches。idx：patch 編號的 tensor。回傳 dict，CKPT_METRICS 各指標在該索引上的平均值。"""
    model.eval()
    sums = {name: 0.0 for name in CKPT_METRICS}
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            x = data.agg(idx[i:i + BATCH]).to(device)
            _, log_lam, _, _ = model(x)
            for name in CKPT_METRICS:
                sums[name] += METRICS[name](log_lam, x).sum().item()
    return {name: s / len(idx) for name, s in sums.items()}


def run(data, train_idx, val_idx, test_idx, log, fold):
    """傳入：資料集、三組索引、log 函式、fold 編號（1 起算）。
    回傳：dict，key 是選 checkpoint 用的指標名，value 是該份 checkpoint 的 best_epoch、val 值與 test 值。"""
    torch.manual_seed(SEED + fold)
    model = AE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # 每個指標各自追蹤自己的最佳 epoch 與權重，彼此獨立
    best = {name: {"val": float("inf"), "epoch": -1, "state": None}
            for name in CKPT_METRICS}

    for epoch in range(EPOCHS):
        try:
            model.train()
            g = torch.Generator().manual_seed(SEED + fold * 100000 + epoch)
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
                val = evaluate(model, data, val_idx)   # 驗證用乾淨輸入
                for name in CKPT_METRICS:
                    if val[name] < best[name]["val"]:
                        best[name] = {
                            "val": val[name], "epoch": epoch + 1,
                            "state": {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()},
                        }
                log(f"[fold {fold}] epoch {epoch + 1:4d} | train_nll {total / len(perm):.5f} | "
                    f"train_kl {total_kl / len(perm):.5f} | val " +
                    " ".join(f"{n}={val[n]:.5f}" for n in CKPT_METRICS))
        except KeyboardInterrupt:
            log(f"\n[中斷] fold {fold} epoch {epoch + 1}")
            break

    results = {}
    for name in CKPT_METRICS:
        if best[name]["state"] is None:
            continue
        model.load_state_dict(best[name]["state"])
        torch.save(model.state_dict(), ckpt_path(fold, name))

        test = evaluate(model, data, test_idx)
        log(f"[fold {fold}] ckpt[{name}] epoch {best[name]['epoch']}，"
            f"val_{name} {best[name]['val']:.5f} → test " +
            " ".join(f"{n}={test[n]:.5f}" for n in CKPT_METRICS))

        model.eval()
        metric_fn = METRICS[name]
        zs, errs = [], []
        with torch.no_grad():
            for i in range(0, data.n, BATCH):
                idx = torch.arange(i, min(i + BATCH, data.n))
                x = data.agg(idx).to(device)
                z, log_lam, _, _ = model(x)
                zs.append(z.cpu())
                errs.append(metric_fn(log_lam, x).cpu())
        z, err = torch.cat(zs).numpy(), torch.cat(errs).numpy()

        split = np.zeros(data.n, dtype=np.int8)   # 0=train 1=val 2=test
        split[val_idx.numpy()], split[test_idx.numpy()] = 1, 2
        np.savez(latents_path(fold, name), n_poi=data.n_poi, lat=data.lat,
                 lon=data.lon, z=z, err=err, split=split)

        results[name] = {"epoch": best[name]["epoch"], "val": best[name]["val"],
                         "test": test}
    return results


def main():
    log = open_log(VERSION, {
        "EPOCHS": EPOCHS, "SEED": SEED, "METRIC": METRIC, "CKPT_METRICS": CKPT_METRICS,
        "LAMBDA_KL": LAMBDA_KL, "N_SPLITS": N_SPLITS, "TEST_FRAC": TEST_FRAC,
    })

    data = Patches(PATCHES)
    test_idx, folds = make_kfold_split(data.lat, data.lon, seed=SEED,
                                       test_frac=TEST_FRAC, n_splits=N_SPLITS)
    log(f"test 集：{len(test_idx)} 個 patch（{TEST_FRAC * 100:.0f}%，"
        f"獨立於全部 {N_SPLITS} 個 fold 的訓練與驗證）\n")

    fold_results = []
    for k, (train_idx, val_idx) in enumerate(folds):
        fold = k + 1
        log(f"\n===== fold {fold}/{N_SPLITS} =====")
        log(f"split：train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}")
        fold_results.append(run(data, train_idx, val_idx, test_idx, log, fold))

    log(f"\n===== {N_SPLITS}-fold test 結果（每個指標取自己那份 checkpoint）=====")
    log(f"{'fold':<6s}" + "".join(f"{n:>12s}" for n in CKPT_METRICS))
    for k, res in enumerate(fold_results):
        log(f"{k + 1:<6d}" + "".join(
            f"{res[n]['test'][n]:12.5f}" if n in res else f"{'-':>12s}"
            for n in CKPT_METRICS))
    log("-" * (6 + 12 * len(CKPT_METRICS)))
    for stat, fn in (("mean", np.mean), ("std", np.std)):
        log(f"{stat:<6s}" + "".join(
            f"{fn([r[n]['test'][n] for r in fold_results if n in r]):12.5f}"
            for n in CKPT_METRICS))
    log("\ndone")


main()
