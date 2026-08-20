import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from cfg import (BATCH, CKPT, EDGE_BATCH, EPOCHS, GRAPH_METRIC, LAMBDA_FSCE,
                 LATENT_DIM, LR, N_NEIGHBORS, OUT, SEED, USE_FSCE, VERSION,
                 WARMUP_EPOCHS, WEIGHT_DECAY, device, open_log)
from dataset import Patches, make_split
from model import AE, build_fsce_graph, fsce_loss, poisson_deviance, poisson_nll
from common.dataset import PATCHES


def evaluate(model, data, idx):
    """model：AE。data：Patches。idx：patch 編號的 tensor。回傳這批 patch 的平均 deviance。"""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            x = data.agg(idx[i:i + BATCH]).to(device)
            _, log_lam = model(x)
            out.append(poisson_deviance(log_lam, x))
    return torch.cat(out).mean().item()


def run(data, train_idx, val_idx, test_idx, log,
        edge_i=None, edge_j=None, edge_w=None, a=None, b=None):
    torch.manual_seed(SEED)
    model = AE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_edges = len(edge_i) if USE_FSCE else 0
    n_train = len(train_idx)
    best_dev, best_epoch, best_state = float("inf"), -1, None

    for epoch in range(EPOCHS):
        try:
            model.train()
            lam_t = LAMBDA_FSCE * min(1.0, (epoch + 1) / WARMUP_EPOCHS) if USE_FSCE else 0.0
            g = torch.Generator().manual_seed(SEED + epoch)
            perm = train_idx[torch.randperm(len(train_idx), generator=g)]
            total, total_fsce = 0.0, 0.0
            for i in range(0, len(perm), BATCH):
                batch = perm[i:i + BATCH]
                x = data.agg(batch).to(device)
                _, log_lam = model(x)
                recon = poisson_nll(log_lam, x).mean()

                if USE_FSCE:
                    eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
                    pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
                    ni = train_idx[torch.randint(0, n_train, (EDGE_BATCH,), generator=g)]
                    nj = train_idx[torch.randint(0, n_train, (EDGE_BATCH,), generator=g)]
                    xi = data.agg(torch.cat([pi, ni])).to(device)
                    xj = data.agg(torch.cat([pj, nj])).to(device)
                    w = torch.cat([pw, torch.zeros(EDGE_BATCH)]).to(device)
                    zi, zj = model.encode(xi), model.encode(xj)
                    fsce = fsce_loss(zi, zj, w, a, b).mean()
                    loss = recon + lam_t * fsce
                    total_fsce += fsce.item() * len(batch)
                else:
                    loss = recon

                opt.zero_grad()
                loss.backward()
                opt.step()
                total += recon.item() * len(batch)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                dev = evaluate(model, data, val_idx)   # 驗證用乾淨輸入
                if dev < best_dev:
                    best_dev, best_epoch = dev, epoch + 1
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                msg = f"epoch {epoch + 1:4d} | train_nll {total / len(perm):.5f} | "
                if USE_FSCE:
                    msg += f"train_fsce {total_fsce / len(perm):.5f} | "
                log(msg + f"val_dev {dev:.5f}")
        except KeyboardInterrupt:
            log(f"\n[中斷] epoch {epoch + 1} 收到 Ctrl-C，用目前最佳模型存 checkpoint 跟 latent")
            break

    if best_state is not None:   # 用 val 最好的那版，不是最後一版
        model.load_state_dict(best_state)
        log(f"\n最佳 checkpoint：epoch {best_epoch}，val_dev {best_dev:.5f}")
    torch.save(model.state_dict(), CKPT)

    log(f"test_dev {evaluate(model, data, test_idx):.5f}"
        f"（{len(test_idx)} 個 patch，全程未參與訓練與選 checkpoint）")

    model.eval()
    zs, errs = [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x = data.agg(idx).to(device)
            z, log_lam = model(x)
            zs.append(z.cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
    return torch.cat(zs).numpy(), torch.cat(errs).numpy()


def main():
    cfg_log = {
        "EPOCHS": EPOCHS,
        "SEED": SEED,
        "USE_FSCE": USE_FSCE,
    }
    if USE_FSCE:
        cfg_log["LAMBDA_FSCE"] = LAMBDA_FSCE
        cfg_log["WARMUP_EPOCHS"] = WARMUP_EPOCHS
    log = open_log(VERSION, cfg_log)

    data = Patches(PATCHES)
    train_idx, val_idx, test_idx = make_split(data.lat, data.lon, seed=SEED)
    log(f"split：train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}\n")

    if USE_FSCE:
        # fuzzy graph 只用 train 建，邊的 index 再 map 回全域，val/test 不進梯度
        x_tr = np.log1p(data.agg(train_idx).numpy())
        ei, ej, edge_w, a, b = build_fsce_graph(
            x_tr, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)
        edge_i, edge_j = train_idx[ei], train_idx[ej]
        z, err = run(data, train_idx, val_idx, test_idx, log,
                     edge_i, edge_j, edge_w, a, b)
    else:
        z, err = run(data, train_idx, val_idx, test_idx, log)
    split = np.zeros(data.n, dtype=np.int8)   # 0=train 1=val 2=test
    split[val_idx.numpy()], split[test_idx.numpy()] = 1, 2
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err,
             split=split)
    log(f"done")


main()
