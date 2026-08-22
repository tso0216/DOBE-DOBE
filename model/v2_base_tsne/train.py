import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from cfg import (BATCH, CKPT, DECAY_EPOCHS, EPOCHS, GAMMA, LAMBDA_TSNE,
                 LATENT_DIM, LR, LR_MIN, METRIC, NOISE_MODE, NOISE_P, OUT,
                 PCGRAD, PERPLEXITY, SEED, SELECT_AFTER, TSNE, TSNE_BATCH,
                 TSNE_LEARN_SCALE, TSNE_NORM, TSNE_SCALE, VERSION,
                 WARMUP_EPOCHS, WEIGHT_DECAY, device, open_log)
from dataset import Patches, corrupt
from model import AE, METRICS, build_tsne_p, poisson_nll, renorm_p, tsne_loss
from common.dataset import PATCHES, make_split

metric_fn = METRICS[METRIC]


def evaluate(model, data, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            x = data.agg(idx[i:i + BATCH]).to(device)
            _, log_lam = model(x)
            out.append(metric_fn(log_lam, x))
    return torch.cat(out).mean().item()


def eval_kl(model, data, train_idx, p_full, log_s):
    """model：AE。p_full：整個 train 的 P。log_s：可學尺度或 None。
    回傳全體 train 在 gamma=1 下的 KL(P‖Q)，只作診斷用、不進梯度。
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(data.agg(train_idx).to(device))
        v = tsne_loss(z, p_full, norm=TSNE_NORM, scale=TSNE_SCALE,
                      log_s=log_s, gamma=1.0).item()
    model.train()
    return v


def lambda_at(epoch):
    """epoch：0 起算的 epoch 編號。回傳這個 epoch 要用的 t-SNE 權重 lambda。"""
    e = epoch + 1
    if e <= WARMUP_EPOCHS:
        return LAMBDA_TSNE * e / WARMUP_EPOCHS
    if DECAY_EPOCHS <= 0:
        return LAMBDA_TSNE
    r = min(1.0, (e - WARMUP_EPOCHS) / DECAY_EPOCHS)   # 塑形完就把壓力收掉，讓 recon 自由收斂
    return LAMBDA_TSNE * 0.5 * (1.0 + math.cos(math.pi * r))


def pcgrad_step(model, log_s, recon, weighted_tsne, eps=1e-12):
    """model：AE。log_s：可學尺度或 None。recon、weighted_tsne：兩個純量 loss。
    把 t-SNE 梯度中與 recon 梯度反向的分量投影掉後寫進 .grad，回傳被投影的參數個數。
    """
    mp = list(model.parameters())
    g_r = torch.autograd.grad(recon, mp, retain_graph=True)
    tgt = mp + ([log_s] if log_s is not None else [])
    g_t = torch.autograd.grad(weighted_tsne, tgt, allow_unused=True)
    g_t = [torch.zeros_like(p) if g is None else g for p, g in zip(tgt, g_t)]

    # 投影要對「整個梯度向量」做，逐張量分開判斷會在某些層過度投影、某些層漏掉
    flat_r = torch.cat([g.reshape(-1) for g in g_r])
    flat_t = torch.cat([g.reshape(-1) for g in g_t[:len(mp)]])
    dot = (flat_r * flat_t).sum()
    coef = dot / flat_r.pow(2).sum().clamp_min(eps) if dot < 0 else None

    for p, gr, gt in zip(mp, g_r, g_t):
        p.grad = gr + (gt - coef * gr if coef is not None else gt)
    if log_s is not None:
        log_s.grad = g_t[-1]        # 尺度只服務 t-SNE，不必投影
    return 1 if coef is not None else 0


def run(data, train_idx, val_idx, test_idx, p_full, log):
    torch.manual_seed(SEED)
    model = AE(LATENT_DIM).to(device)
    # Q 專用的 log 尺度，只跟著 t-SNE loss 走，不進 checkpoint 也不影響 decoder
    log_s = (torch.zeros((), device=device, requires_grad=True)
             if TSNE_LEARN_SCALE else None)
    params = list(model.parameters()) + ([log_s] if log_s is not None else [])
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)
    n_train = len(train_idx)
    # TSNE_BATCH=0 或大於 train 數就用全部 train，KL 才是精確的（N 小，pairwise 撐得住）
    n_sub = n_train if TSNE_BATCH <= 0 else min(TSNE_BATCH, n_train)
    best_metric, best_epoch, best_state = float("inf"), -1, None

    for epoch in range(EPOCHS):
        try:
            model.train()
            lam_t = lambda_at(epoch)
            g = torch.Generator().manual_seed(SEED + epoch)
            perm = train_idx[torch.randperm(len(train_idx), generator=g)]
            total, total_tsne = 0.0, 0.0
            for i in range(0, len(perm), BATCH):
                batch = perm[i:i + BATCH]
                x = data.agg(batch)                                  # 乾淨目標（CPU）
                x_in = corrupt(x, NOISE_P, NOISE_MODE, generator=g)  # 加噪輸入
                x, x_in = x.to(device), x_in.to(device)
                _, log_lam = model(x_in)
                recon = poisson_nll(log_lam, x).mean()   # 目標永遠是乾淨的 x

                if TSNE:
                    # pos 是 p_full 的列號（即 train_idx 內的位置），只有 train 進梯度
                    pos = (torch.arange(n_train) if n_sub == n_train
                           else torch.randperm(n_train, generator=g)[:n_sub])
                    xs = corrupt(data.agg(train_idx[pos]), NOISE_P,
                                 NOISE_MODE, generator=g).to(device)
                    pos_d = pos.to(device)   # index tensor 要跟 p_full 同 device
                    p_sub = renorm_p(p_full[pos_d][:, pos_d], TSNE_NORM)   # 抽子集後要重新正規化
                    tsne = tsne_loss(model.encode(xs), p_sub, norm=TSNE_NORM,
                                     scale=TSNE_SCALE, log_s=log_s, gamma=GAMMA)
                    loss = recon + lam_t * tsne
                    total_tsne += tsne.item() * len(batch)
                else:
                    loss = recon

                opt.zero_grad()
                if TSNE and PCGRAD:
                    pcgrad_step(model, log_s, recon, lam_t * tsne)
                else:
                    loss.backward()
                opt.step()
                total += recon.item() * len(batch)

            sched.step()

            val_metric = evaluate(model, data, val_idx)   # 驗證不加噪
            if epoch + 1 >= SELECT_AFTER and val_metric < best_metric:
                best_metric, best_epoch = val_metric, epoch + 1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

            if (epoch + 1) % 50 == 0 or epoch == 0:
                test_metric = evaluate(model, data, test_idx)
                # train_tsne 是實際被優化的值（GAMMA≠1 時不是 KL，甚至可能是負的），
                # kl 這欄固定用 gamma=1 算，跨設定才能比
                kl = eval_kl(model, data, train_idx, p_full, log_s) if TSNE else 0.0
                log(f"epoch {epoch + 1:4d} train_nll {total / len(perm):.5f} | "
                    f"train_tsne {total_tsne / len(perm):.5f} | kl {kl:.4f} | "
                    f"q_scale {1.0 if log_s is None else log_s.exp().item():.2f} | "
                    f"val_{METRIC} {val_metric:.5f} | test_{METRIC} {test_metric:.5f}")
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
            x = data.agg(idx).to(device)   # 推論不加噪
            z, log_lam = model(x)
            zs.append(z.cpu())
            errs.append(metric_fn(log_lam, x).cpu())
    return torch.cat(zs).numpy(), torch.cat(errs).numpy()


def main():
    log = open_log(VERSION, {
        "EPOCHS": EPOCHS,
        "LR": LR,
        "LR_MIN": LR_MIN,
        "METRIC": METRIC,
        "SEED": SEED,
        "NOISE_P": NOISE_P,
        "NOISE_MODE": NOISE_MODE,
        "TSNE": TSNE,
        "PERPLEXITY": PERPLEXITY,
        "TSNE_NORM": TSNE_NORM,
        "TSNE_SCALE": TSNE_SCALE,
        "TSNE_LEARN_SCALE": TSNE_LEARN_SCALE,
        "TSNE_BATCH": TSNE_BATCH,
        "LAMBDA_TSNE": LAMBDA_TSNE,
        "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "DECAY_EPOCHS": DECAY_EPOCHS,
        "GAMMA": GAMMA,
        "PCGRAD": PCGRAD,
        "SELECT_AFTER": SELECT_AFTER,
    })

    data = Patches(PATCHES)
    train_idx, val_idx, test_idx = make_split(data.lat, data.lon, seed=SEED)
    log(f"split：train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}\n")

    if TSNE:
        # P 只用 train 建，列序就是 train_idx 的順序，val/test 不進梯度
        x_tr = np.log1p(data.agg(train_idx).numpy())
        p_full = build_tsne_p(x_tr, perplexity=PERPLEXITY,
                              norm=TSNE_NORM).to(device)
    else:
        p_full = None

    z, err = run(data, train_idx, val_idx, test_idx, p_full, log)
    split = np.zeros(data.n, dtype=np.int8)   # 0=train 1=val 2=test
    split[val_idx.numpy()], split[test_idx.numpy()] = 1, 2
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err,
             split=split)
    log(f"done")


main()
