import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
# 直接跑這支檔案時 python 會把本目錄放進 sys.path，同目錄的 model.py 就會蓋掉 model/ package，
# 所以先把本目錄拿掉，只留 repo 根目錄
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, os.path.abspath(f"{_HERE}/../.."))
from model.v3_ddae_tfidf.cfg import (BATCH, CKPT_METRICS, EDGE_BATCH, EPOCHS, FSCE,
                 GRAPH_MODE, HIDDEN, LAMBDA_FSCE, LATENT_DIM, LR, LR_MIN, METRIC,
                 N_CLUSTERS, N_NEIGHBORS, N_SPLITS, NOISE_MODE, NOISE_P, PCGRAD, SEED,
                 SNAPSHOT_PERCENTS, TEST_FRAC,
                 VERSION, WARMUP_EPOCHS, WEIGHT_DECAY, ckpt_path, device, latents_path, open_log)
from model.v3_ddae_tfidf.dataset import Patches, corrupt
from model.v3_ddae_tfidf.model import (AE, METRICS, build_fsce_graph, build_tfidf_fsce_graph,
                   compute_tfidf_features, fsce_loss, pcgrad_step, poisson_nll)
from common.dataset import CATEGORIES, PATCHES, make_kfold_split, result


def save_snapshot(z, labels, epoch, fold, tag=""):
    out_dir = result(VERSION, f"snapshots/fold{fold}")
    os.makedirs(out_dir, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    ids = sorted(set(labels.tolist()) - {-1})

    fig, ax = plt.subplots(figsize=(6, 6))
    for k, c in enumerate(ids):
        m = labels == c
        ax.scatter(z[m, 0], z[m, 1], s=4, color=cmap(k % 10), linewidths=0,
                   alpha=0.85, label=f"c{c} ({int(m.sum())})", rasterized=True)
    ax.set_title(f"{VERSION} fold{fold} epoch {epoch}{' ' + tag if tag else ''}", fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=6, markerscale=2, loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    name = f"epoch_{epoch:04d}{'_' + tag if tag else ''}.png"
    fig.savefig(os.path.join(out_dir, name), bbox_inches="tight", dpi=130)
    plt.close(fig)


def encode_all(model, data, metric, batch=BATCH):
    """傳入：model、資料集、算 err 用的指標名。回傳：(z, err)，全體 patch 的 latent 與逐 patch 誤差。"""
    model.eval()
    metric_fn = METRICS[metric]
    zs, errs = [], []
    with torch.no_grad():
        for i in range(0, data.n, batch):
            idx = torch.arange(i, min(i + batch, data.n))
            x = data.agg(idx).to(device)
            z, log_lam = model(x)
            zs.append(z.cpu())
            errs.append(metric_fn(log_lam, x).cpu())
    return torch.cat(zs).numpy(), torch.cat(errs).numpy()


def evaluate(model, data, idx):
    """傳入：model、資料集、要評估的 patch 索引。回傳：dict，CKPT_METRICS 各指標在該索引上的平均值。"""
    model.eval()
    sums = {name: 0.0 for name in CKPT_METRICS}
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            x = data.agg(idx[i:i + BATCH]).to(device)
            _, log_lam = model(x)
            for name in CKPT_METRICS:
                sums[name] += METRICS[name](log_lam, x).sum().item()
    return {name: s / len(idx) for name, s in sums.items()}


def run(data, train_idx, val_idx, test_idx, labels,
        edge_i, edge_j, edge_w, a, b, log, fold):
    """傳入：資料集、三組索引、KMeans 標籤、FSCE 邊、log 函式、fold 編號（1 起算）。
    回傳：dict，key 是選 checkpoint 用的指標名，value 是該份 checkpoint 的 best_epoch、val 值與 test 四指標。"""
    torch.manual_seed(SEED + fold)
    model = AE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)
    n_edges = len(edge_i) if FSCE else 0
    n_train = len(train_idx)
    # 四個指標各自追蹤自己的最佳 epoch 與權重，彼此獨立
    best = {name: {"val": float("inf"), "epoch": -1, "state": None}
            for name in CKPT_METRICS}
    snapshot_epochs = {round(EPOCHS * p / 100) for p in SNAPSHOT_PERCENTS}

    for epoch in range(EPOCHS):
        try:
            model.train()
            lam_t = LAMBDA_FSCE * min(1.0, (epoch + 1) / WARMUP_EPOCHS)
            g = torch.Generator().manual_seed(SEED + fold * 100000 + epoch)
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
                    zi, zj = model.encode(xi), model.encode(xj)
                    fsce = fsce_loss(zi, zj, w, a, b).mean()

                    opt.zero_grad()
                    if PCGRAD:
                        pcgrad_step(model, recon, lam_t * fsce)
                    else:
                        loss = recon + lam_t * fsce
                        loss.backward()
                    total_fsce += fsce.item() * len(batch)
                else:
                    opt.zero_grad()
                    recon.backward()

                opt.step()
                total += recon.item() * len(batch)

            sched.step()

            val = evaluate(model, data, val_idx)
            for name in CKPT_METRICS:
                if val[name] < best[name]["val"]:
                    best[name] = {
                        "val": val[name], "epoch": epoch + 1,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()},
                    }

            if (epoch + 1) % 50 == 0 or epoch == 0:
                test = evaluate(model, data, test_idx)
                log(f"[fold {fold}] epoch {epoch + 1:4d} train_nll {total / len(perm):.5f} | "
                    f"train_fsce {total_fsce / len(perm):.5f} | "
                    f"val " + " ".join(f"{n}={val[n]:.5f}" for n in CKPT_METRICS) +
                    f" | test_{METRIC} {test[METRIC]:.5f}")

            if (epoch + 1) in snapshot_epochs or epoch == 0:
                z_snap, _ = encode_all(model, data, METRIC)
                save_snapshot(z_snap, labels, epoch + 1, fold)
                model.train()

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

        z, err = encode_all(model, data, name)
        save_snapshot(z, labels, EPOCHS, fold, tag=name)
        split = np.zeros(data.n, dtype=np.int8)   # 0=train 1=val 2=test
        split[val_idx.numpy()], split[test_idx.numpy()] = 1, 2
        np.savez(latents_path(fold, name), n_poi=data.n_poi, lat=data.lat,
                 lon=data.lon, z=z, err=err, split=split)

        results[name] = {"epoch": best[name]["epoch"], "val": best[name]["val"],
                         "test": test}
    return results


def main():
    log = open_log(VERSION, {
        "EPOCHS": EPOCHS, "LR": LR, "LR_MIN": LR_MIN, "METRIC": METRIC,
        "CKPT_METRICS": CKPT_METRICS, "N_SPLITS": N_SPLITS, "TEST_FRAC": TEST_FRAC,
        "SEED": SEED, "NOISE_P": NOISE_P, "NOISE_MODE": NOISE_MODE,
        "FSCE": FSCE, "GRAPH_MODE": GRAPH_MODE, "LAMBDA_FSCE": LAMBDA_FSCE, "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "PCGRAD": PCGRAD, "N_CLUSTERS": N_CLUSTERS, "SNAPSHOT_PERCENTS": SNAPSHOT_PERCENTS,
        "HIDDEN": HIDDEN, "WEIGHT_DECAY": WEIGHT_DECAY, "EDGE_BATCH": EDGE_BATCH,
        "N_NEIGHBORS": N_NEIGHBORS,
    })

    data = Patches(PATCHES)
    test_idx, folds = make_kfold_split(data.lat, data.lon, seed=SEED,
                                       test_frac=TEST_FRAC, n_splits=N_SPLITS)
    log(f"test 集：{len(test_idx)} 個 patch（{TEST_FRAC * 100:.0f}%，"
        f"獨立於全部 {N_SPLITS} 個 fold 的訓練與驗證）\n")

    x_all_np = data.agg(torch.arange(data.n)).numpy()
    x_tfidf_all, idf_weights = compute_tfidf_features(x_all_np)
    log("TF-IDF POI 特徵轉換完成（IDF 獨特性加權計算完成）")
    for c in range(10):
        log(f"  {CATEGORIES[c]:35s} IDF = {idf_weights[c]:.4f}")

    labels = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(x_tfidf_all)
    log(f"\nTF-IDF 高維 POI 機能分群（K={N_CLUSTERS}）")

    fold_results = []
    for k, (train_idx, val_idx) in enumerate(folds):
        fold = k + 1
        log(f"\n===== fold {fold}/{N_SPLITS} =====")
        log(f"split：train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}")

        # fuzzy graph 每個 fold 只用該 fold 的 train 重建，val/test 不進梯度
        if GRAPH_MODE == "tfidf":
            x_tfidf_tr = x_tfidf_all[train_idx.numpy()]
            ei, ej, edge_w, a, b = build_tfidf_fsce_graph(x_tfidf_tr, n_neighbors=N_NEIGHBORS)
            log(f"TF-IDF Weighted FSCE graph：{len(ei)} 條邊，a={a:.4f} b={b:.4f}")
        else:
            x_tr = np.log1p(data.agg(train_idx).numpy())
            ei, ej, edge_w, a, b = build_fsce_graph(x_tr, n_neighbors=N_NEIGHBORS)
            log(f"Plain FSCE graph（log1p count, euclidean）：{len(ei)} 條邊，a={a:.4f} b={b:.4f}")
        edge_i, edge_j = train_idx[ei], train_idx[ej]

        fold_results.append(run(data, train_idx, val_idx, test_idx, labels,
                                edge_i, edge_j, edge_w, a, b, log, fold))

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
