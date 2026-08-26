"""評估 experiment/model/ 下各模型的 5-fold checkpoint 在 test 集上的重建指標"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import N_CAT, PATCHES, make_kfold_split  # noqa: E402
from model.v3_ddae_tfidf.dataset import Patches  # noqa: E402
from model.v3_ddae_tfidf.model import METRICS  # noqa: E402

MODEL_DIR = os.path.join(HERE, "..", "model")

# pca 沒有 checkpoint，改成每個 fold 用該 fold 的 train 現場配一組
PCA_NAME = "pca"
PCA_COMPONENTS = 2
MODEL_TO_TEST = [PCA_NAME, "ae", "vae", "ddae_fsce_tfidf"]

SEED = 0
N_SPLITS = 5
TEST_FRAC = 0.2
BATCH = 256
METRIC_NAMES = ["mae", "mse", "wape", "deviance"]
EPS = 1e-8


class AE(nn.Module):
    def __init__(self, latent_dim, hidden):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_CAT, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, N_CAT),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


class VAE(nn.Module):
    def __init__(self, latent_dim, hidden):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_CAT, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, N_CAT),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z, self.decoder(z), mu, logvar


def ckpt_path(model_name, fold, metric):
    """model_name：experiment/model/ 下的模型目錄名。fold：fold 編號（1 起算）。metric：選 checkpoint 用的指標名。回傳該份 checkpoint 的路徑。"""
    return os.path.join(MODEL_DIR, model_name, f"fold{fold}_{metric}.pt")


def load_model(path):
    """path：checkpoint 檔案路徑。回傳依 state_dict 判斷出的模型（AE 或 VAE），已載入權重並設為 eval，
    hidden 與 latent_dim 由 state_dict 推得。"""
    sd = torch.load(path, map_location="cpu")
    hidden = sd["encoder.0.weight"].shape[0]
    latent_dim = sd["decoder.0.weight"].shape[1]
    cls = VAE if "to_mu.weight" in sd else AE
    model = cls(latent_dim, hidden)
    model.load_state_dict(sd)
    model.eval()
    return model


def nn_predictor(path):
    """path：checkpoint 檔案路徑。回傳 predict 函式（吃 (B, N_CAT) count tensor，回傳 (B, N_CAT) log_lam
    tensor）；VAE 取 mu 解碼、不取樣，避免重建指標帶到取樣雜訊。"""
    model = load_model(path)

    def predict(x):
        if isinstance(model, VAE):
            return model.decoder(model.to_mu(model.encoder(x)))
        return model(x)[1]
    return predict


def pca_predictor(data, train_idx):
    """data：Patches。train_idx：該 fold 的訓練 index。回傳 predict 函式（吃 (B, N_CAT) count tensor，
    回傳 (B, N_CAT) log_lam tensor）；PCA 在 log1p 空間配、重建後轉回 count 再取 log 當 log_lam。"""
    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit(
        np.log1p(data.agg(train_idx).numpy()))

    def predict(x):
        x_log = np.log1p(x.numpy())
        lam = np.expm1(pca.inverse_transform(pca.transform(x_log))).clip(min=0.0)
        return torch.from_numpy(np.log(np.maximum(lam, EPS))).float()
    return predict


def test_metric(predict, data, idx, metric):
    """predict：(B, N_CAT) count tensor -> (B, N_CAT) log_lam tensor 的函式。data：Patches。
    idx：test 集 patch 編號 tensor。metric：指標名。回傳該指標在 test 集上的平均值。"""
    fn = METRICS[metric]
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            x = data.agg(idx[i:i + BATCH])
            total += fn(predict(x), x).sum().item()
    return total / len(idx)


def main():
    data = Patches(PATCHES)
    test_idx, folds = make_kfold_split(data.lat, data.lon, seed=SEED,
                                       test_frac=TEST_FRAC, n_splits=N_SPLITS)
    print(f"test 集：{len(test_idx)} 個 patch")

    row_names = [f"fold{fold}" for fold in range(1, N_SPLITS + 1)] + ["mean"]

    for metric in METRIC_NAMES:
        # scores[model_name][row_name]：row_name 形如 fold1、mean
        scores = {name: {} for name in MODEL_TO_TEST}
        for model_name in MODEL_TO_TEST:
            vals = []
            for fold in range(1, N_SPLITS + 1):
                if model_name == PCA_NAME:
                    predict = pca_predictor(data, folds[fold - 1][0])
                else:
                    path = ckpt_path(model_name, fold, metric)
                    if not os.path.exists(path):
                        print(f"[跳過] 找不到 {path}")
                        continue
                    predict = nn_predictor(path)
                v = test_metric(predict, data, test_idx, metric)
                scores[model_name][f"fold{fold}"] = v
                vals.append(v)
                print(f"{model_name} fold{fold} {metric} = {v:.5f}")
            if vals:
                scores[model_name]["mean"] = float(np.mean(vals))

        out = os.path.join(HERE, f"{metric}.csv")
        with open(out, "w") as f:
            f.write("fold," + ",".join(MODEL_TO_TEST) + "\n")
            for row in row_names:
                cells = [f"{scores[m][row]:.5f}" if row in scores[m] else ""
                         for m in MODEL_TO_TEST]
                f.write(f"{row}," + ",".join(cells) + "\n")
        print(f"已存 {out}")


main()
