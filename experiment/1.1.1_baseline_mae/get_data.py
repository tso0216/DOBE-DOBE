"""計算 PCA / AE / VAE / Ours 在測試集上的重建 MAE"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import N_CAT, PATCHES, make_split  # noqa: E402
from model.v2_ddae_base.dataset import Patches  # noqa: E402

MODEL_DIR = os.path.join(HERE, "..", "model")
SEED = 0
PCA_LATENT_DIM = 2

# (權重檔名，圖上顯示的標籤)；PCA 用 None 代表不是讀權重檔，而是即時配 PCA
MODELS = [
    (None, "PCA"),
    ("ae.pt", "AE"),
    ("vae.pt", "VAE"),
    ("dae_tfidf.pt", "Ours"),
]


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


def load_model(path):
    """path：checkpoint 檔案路徑。回傳依 state_dict 判斷出的模型（AE 或 VAE），已載入權重並設為 eval。"""
    sd = torch.load(path, map_location="cpu")
    hidden = sd["encoder.0.weight"].shape[0]
    latent_dim = sd["decoder.0.weight"].shape[1]
    cls = VAE if "to_mu.weight" in sd else AE
    model = cls(latent_dim, hidden)
    model.load_state_dict(sd)
    model.eval()
    return model


def test_mae_nn(model, data, idx, batch=256):
    """model：AE 或 VAE。data：Patches。idx：測試集 patch 編號 tensor。回傳整個測試集的平均 MAE。"""
    diffs = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            x = data.agg(idx[i:i + batch])
            log_lam = model(x)[1]
            diffs.append((torch.exp(log_lam) - x).abs())
    return torch.cat(diffs).mean().item()


def test_mae_pca(data, train_idx, test_idx):
    """data：Patches。train_idx：拿來配 PCA 的 patch 編號 tensor。test_idx：測試集 patch 編號 tensor。回傳測試集的平均 MAE。"""
    x = data.agg(torch.arange(data.n)).numpy()
    x_log = np.log1p(x)

    pca = PCA(n_components=PCA_LATENT_DIM, random_state=SEED).fit(x_log[train_idx.numpy()])
    recon_log = pca.inverse_transform(pca.transform(x_log))
    lam = np.clip(np.expm1(recon_log), 1e-6, None)

    diff = np.abs(lam - x)[test_idx.numpy()]
    return float(diff.mean())


def main():
    data = Patches(PATCHES)
    train_idx, _, test_idx = make_split(data.lat, data.lon, seed=SEED)

    rows = []
    for name, label in MODELS:
        if name is None:
            mae = test_mae_pca(data, train_idx, test_idx)
        else:
            model = load_model(os.path.join(MODEL_DIR, name))
            mae = test_mae_nn(model, data, test_idx)
        rows.append((label, mae))
        print(f"{label}: test_mae = {mae:.5f}")

    out = os.path.join(HERE, "data.csv")
    with open(out, "w") as f:
        f.write("model,mae\n")
        for label, mae in rows:
            f.write(f"{label},{mae:.5f}\n")
    print(f"已存 {out}")


main()
