import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from model.v3_ddae_tfidf.cfg import (BATCH, CKPT, EDGE_BATCH, EPOCHS, FSCE,
                 GRAPH_METRIC, GRAPH_MODE, HIDDEN, LAMBDA_FSCE, LATENT_DIM, LR, LR_MIN, METRIC,
                 N_CLUSTERS, N_NEIGHBORS, NOISE_MODE, NOISE_P, OUT, PCGRAD, SEED,
                 SNAPSHOT_PERCENTS,
                 VERSION, WARMUP_EPOCHS, WEIGHT_DECAY, device, open_log)
from model.v3_ddae_tfidf.dataset import Patches, corrupt
from model.v3_ddae_tfidf.model import (AE, METRICS, build_fsce_graph, build_tfidf_fsce_graph,
                   compute_tfidf_features, fsce_loss, pcgrad_step, poisson_nll)
from common.dataset import CATEGORIES, PATCHES, make_split, result

metric_fn = METRICS[METRIC]


def save_snapshot(z, labels, epoch, version):
    out_dir = result(version, "snapshots")
    os.makedirs(out_dir, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    ids = sorted(set(labels.tolist()) - {-1})

    fig, ax = plt.subplots(figsize=(6, 6))
    for k, c in enumerate(ids):
        m = labels == c
        ax.scatter(z[m, 0], z[m, 1], s=4, color=cmap(k % 10), linewidths=0,
                   alpha=0.85, label=f"c{c} ({int(m.sum())})", rasterized=True)
    ax.set_title(f"{version} epoch {epoch}", fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=6, markerscale=2, loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"epoch_{epoch:04d}.png"),
                bbox_inches="tight", dpi=130)
    plt.close(fig)


def encode_all(model, data, batch=BATCH):
    model.eval()
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
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            x = data.agg(idx[i:i + BATCH]).to(device)
            _, log_lam = model(x)
            out.append(metric_fn(log_lam, x))
    return torch.cat(out).mean().item()


def run(data, train_idx, val_idx, test_idx, labels,
        edge_i, edge_j, edge_w, a, b, log):
    torch.manual_seed(SEED)
    model = AE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)
    n_edges = len(edge_i) if FSCE else 0
    n_train = len(train_idx)
    best_metric, best_epoch, best_state = float("inf"), -1, None
    snapshot_epochs = {round(EPOCHS * p / 100) for p in SNAPSHOT_PERCENTS}

    for epoch in range(EPOCHS):
        try:
            model.train()
            lam_t = LAMBDA_FSCE * min(1.0, (epoch + 1) / WARMUP_EPOCHS)
            g = torch.Generator().manual_seed(SEED + epoch)
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

            val_metric = evaluate(model, data, val_idx)
            if val_metric < best_metric:
                best_metric, best_epoch = val_metric, epoch + 1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

            if (epoch + 1) % 50 == 0 or epoch == 0:
                test_metric = evaluate(model, data, test_idx)
                log(f"epoch {epoch + 1:4d} train_nll {total / len(perm):.5f} | "
                    f"train_fsce {total_fsce / len(perm):.5f} | val_{METRIC} {val_metric:.5f} | "
                    f"test_{METRIC} {test_metric:.5f}")

            if (epoch + 1) in snapshot_epochs or epoch == 0:
                z_snap, _ = encode_all(model, data)
                save_snapshot(z_snap, labels, epoch + 1, VERSION)
                model.train()

        except KeyboardInterrupt:
            log(f"\n[中斷] epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"\n最佳 checkpoint：epoch {best_epoch}，val_{METRIC} {best_metric:.5f}")
    torch.save(model.state_dict(), CKPT)

    test_dev = evaluate(model, data, test_idx)
    log(f"test_{METRIC} {test_dev:.5f}"
        f"（{len(test_idx)} 個 patch，全程未參與訓練與選 checkpoint）")

    z, err = encode_all(model, data)
    save_snapshot(z, labels, EPOCHS, VERSION)
    return z, err


def main():
    log = open_log(VERSION, {
        "EPOCHS": EPOCHS, "LR": LR, "LR_MIN": LR_MIN, "METRIC": METRIC,
        "SEED": SEED, "NOISE_P": NOISE_P, "NOISE_MODE": NOISE_MODE,
        "FSCE": FSCE, "GRAPH_MODE": GRAPH_MODE, "LAMBDA_FSCE": LAMBDA_FSCE, "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "PCGRAD": PCGRAD, "N_CLUSTERS": N_CLUSTERS, "SNAPSHOT_PERCENTS": SNAPSHOT_PERCENTS,
        "HIDDEN": HIDDEN, "WEIGHT_DECAY": WEIGHT_DECAY, "EDGE_BATCH": EDGE_BATCH,
        "N_NEIGHBORS": N_NEIGHBORS,
    })

    data = Patches(PATCHES)
    train_idx, val_idx, test_idx = make_split(data.lat, data.lon, seed=SEED)
    log(f"split：train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}\n")

    x_all_np = data.agg(torch.arange(data.n)).numpy()
    x_tfidf_all, idf_weights = compute_tfidf_features(x_all_np)
    log("TF-IDF POI 特徵轉換完成（IDF 獨特性加權計算完成）")
    for c in range(10):
        log(f"  {CATEGORIES[c]:35s} IDF = {idf_weights[c]:.4f}")

    labels = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit_predict(x_tfidf_all)
    log(f"\nTF-IDF 高維 POI 機能分群（K={N_CLUSTERS}）")

    if GRAPH_MODE == "tfidf":
        x_tfidf_tr = x_tfidf_all[train_idx.numpy()]
        ei, ej, edge_w, a, b = build_tfidf_fsce_graph(x_tfidf_tr, n_neighbors=N_NEIGHBORS)
        log(f"TF-IDF Weighted FSCE graph：{len(ei)} 條邊，a={a:.4f} b={b:.4f}")
    else:
        x_tr = np.log1p(data.agg(train_idx).numpy())
        ei, ej, edge_w, a, b = build_fsce_graph(x_tr, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)
        log(f"Plain FSCE graph（log1p count, {GRAPH_METRIC}）：{len(ei)} 條邊，a={a:.4f} b={b:.4f}")
    edge_i, edge_j = train_idx[ei], train_idx[ej]

    z, err = run(data, train_idx, val_idx, test_idx, labels,
                 edge_i, edge_j, edge_w, a, b, log)
    split = np.zeros(data.n, dtype=np.int8)
    split[val_idx.numpy()], split[test_idx.numpy()] = 1, 2
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err,
             split=split)
    log(f"done")


main()
