"""重跑指定 fold 的訓練，把不同訓練進度的權重存進 experiment/model/entropy_progress_ckpt/

訓練流程（切分、graph、loss、optimizer、RNG 種子）與 model/v3_ddae_tfidf/train.py 的
run() 一致，差別只在：不挑 best checkpoint、不畫 snapshot 圖，改成在每個進度百分比存一份權重。
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from common.dataset import PATCHES, make_kfold_split  # noqa: E402
from model.v3_ddae_tfidf.cfg import (BATCH, EDGE_BATCH, EPOCHS, FSCE,  # noqa: E402
                                     GRAPH_MODE, LAMBDA_FSCE, LATENT_DIM, LR, LR_MIN,
                                     N_NEIGHBORS, N_SPLITS, NOISE_MODE, NOISE_P, PCGRAD,
                                     SEED, TEST_FRAC, WARMUP_EPOCHS, WEIGHT_DECAY, device)
from model.v3_ddae_tfidf.dataset import Patches, corrupt  # noqa: E402
from model.v3_ddae_tfidf.model import (AE, build_fsce_graph, build_tfidf_fsce_graph,  # noqa: E402
                                       compute_tfidf_features, fsce_loss, pcgrad_step,
                                       poisson_nll)

OUT_DIR = os.path.join(HERE, "model", "entropy_progress_ckpt")

FOLD = 3
SNAPSHOT_PERCENTS = list(range(5, 101, 5))


def ckpt_path(percent, fold):
    """percent：訓練進度百分比。fold：fold 編號（1 起算）。回傳該份進度快照的權重檔路徑。"""
    return os.path.join(OUT_DIR, f"epoch_pct{percent}_fold{fold}.pt")


def build_graph(data, x_tfidf_all, train_idx):
    """data：Patches。x_tfidf_all：全體 patch 的 TF-IDF 特徵。train_idx：該 fold 的訓練 index。
    回傳 (edge_i, edge_j, edge_w, a, b)，edge_i/edge_j 已映射回原始 patch 編號。"""
    if GRAPH_MODE == "tfidf":
        ei, ej, edge_w, a, b = build_tfidf_fsce_graph(
            x_tfidf_all[train_idx.numpy()], n_neighbors=N_NEIGHBORS)
    else:
        x_tr = np.log1p(data.agg(train_idx).numpy())
        ei, ej, edge_w, a, b = build_fsce_graph(
            x_tr, n_neighbors=N_NEIGHBORS)
    return train_idx[ei], train_idx[ej], edge_w, a, b


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    snapshot_at = {round(EPOCHS * p / 100): p for p in SNAPSHOT_PERCENTS}

    data = Patches(PATCHES)
    _, folds = make_kfold_split(data.lat, data.lon, seed=SEED,
                                test_frac=TEST_FRAC, n_splits=N_SPLITS)
    train_idx, val_idx = folds[FOLD - 1]
    print(f"fold {FOLD}／{N_SPLITS}：train {len(train_idx)} / val {len(val_idx)}")

    x_tfidf_all, _ = compute_tfidf_features(data.agg(torch.arange(data.n)).numpy())
    edge_i, edge_j, edge_w, a, b = build_graph(data, x_tfidf_all, train_idx)
    print(f"FSCE graph（{GRAPH_MODE}）：{len(edge_i)} 條邊，a={a:.4f} b={b:.4f}")

    torch.manual_seed(SEED + FOLD)
    model = AE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)
    n_edges = len(edge_i) if FSCE else 0
    n_train = len(train_idx)

    for epoch in range(EPOCHS):
        model.train()
        lam_t = LAMBDA_FSCE * min(1.0, (epoch + 1) / WARMUP_EPOCHS)
        g = torch.Generator().manual_seed(SEED + FOLD * 100000 + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        total, total_fsce = 0.0, 0.0

        for i in range(0, len(perm), BATCH):
            batch = perm[i:i + BATCH]
            x = data.agg(batch)
            x_in = corrupt(x, NOISE_P, NOISE_MODE, generator=g)
            x, x_in = x.to(device), x_in.to(device)
            _, log_lam = model(x_in)
            recon = poisson_nll(log_lam, x).mean()

            if FSCE:
                eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
                pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
                ni = train_idx[torch.randint(0, n_train, (EDGE_BATCH,), generator=g)]
                nj = train_idx[torch.randint(0, n_train, (EDGE_BATCH,), generator=g)]
                xi = corrupt(data.agg(torch.cat([pi, ni])), NOISE_P,
                             NOISE_MODE, generator=g).to(device)
                xj = corrupt(data.agg(torch.cat([pj, nj])), NOISE_P,
                             NOISE_MODE, generator=g).to(device)
                w = torch.cat([pw, torch.zeros(EDGE_BATCH)]).to(device)
                fsce = fsce_loss(model.encode(xi), model.encode(xj), w, a, b).mean()

                opt.zero_grad()
                if PCGRAD:
                    pcgrad_step(model, recon, lam_t * fsce)
                else:
                    (recon + lam_t * fsce).backward()
                total_fsce += fsce.item() * len(batch)
            else:
                opt.zero_grad()
                recon.backward()

            opt.step()
            total += recon.item() * len(batch)

        sched.step()

        percent = snapshot_at.get(epoch + 1)
        if percent is not None:
            out = ckpt_path(percent, FOLD)
            torch.save({k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()}, out)
            print(f"epoch {epoch + 1:4d}（{percent:3d}%）train_nll {total / len(perm):.5f} | "
                  f"train_fsce {total_fsce / len(perm):.5f} → 已存 {os.path.basename(out)}")

    print(f"完成，權重在 {OUT_DIR}")


main()
