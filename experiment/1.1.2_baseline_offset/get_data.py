"""計算 PCA / AE / VAE / Ours 在各 POI 類別加入不同數量後的 latent 平均偏移"""
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import CATEGORIES, N_CAT, PATCHES, make_split  # noqa: E402
from model.other.v2_ddae_base.dataset import Patches  # noqa: E402

MODEL_DIR = os.path.join(HERE, "..", "model")
ADD_AMOUNT = [1, 2, 3]
SEED = 0


MODELS = [
    (None, "PCA"),
    ("ae/fold3_mae.pt", "AE"),
    ("vae/fold3_mae.pt", "VAE"),
    ("ddae_fsce_tfidf/fold3_mae.pt", "Ours"),
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


def encode(model, x):
    """model：AE 或 VAE。x：(B, N_CAT) count tensor。回傳 (B, latent_dim) latent（VAE 取 mu，不取樣，避免雜訊蓋過偏移訊號）。"""
    with torch.no_grad():
        if isinstance(model, VAE):
            return model.to_mu(model.encoder(x))
        return model.encoder(x)



def pca_encoder(data):
    """data：Patches。在 train split（log1p 後）配 PCA，回傳 encode 函式（吃 count tensor，回傳 latent tensor）。"""
    train_idx, _, _ = make_split(data.lat, data.lon, seed=SEED)
    x_log = np.log1p(data.agg(train_idx).numpy())
    pca = PCA(n_components=2, random_state=SEED).fit(x_log)

    def encode_fn(x):
        return torch.from_numpy(pca.transform(np.log1p(x.numpy()))).float()
    return encode_fn


def poi_shift(encode_fn, x, amounts):
    """encode_fn：(B, N_CAT) tensor -> (B, latent_dim) tensor 的 encode 函式。x：(B, N_CAT) count tensor。amounts：要測試的加入量清單。
    回傳 dict {(category_idx, amount): avg_offset}，偏移＝||encode(x+加入量) - encode(x)||，對 batch 取平均。
    """
    z0 = encode_fn(x)
    offsets = {}
    for c in range(N_CAT):
        for a in amounts:
            x_shift = x.clone()
            x_shift[:, c] += a
            offsets[(c, a)] = (encode_fn(x_shift) - z0).norm(dim=1).mean().item()
    return offsets


def main():
    data = Patches(PATCHES)
    _, _, test_idx = make_split(data.lat, data.lon, seed=SEED)
    x_test = data.agg(test_idx)

    encoders = []
    for name, label in MODELS:
        if name is None:
            encoders.append((label, pca_encoder(data)))
        else:
            model = load_model(os.path.join(MODEL_DIR, name))
            encoders.append((label, lambda x, model=model: encode(model, x)))

    out_path = os.path.join(HERE, "data.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "category", "amount", "avg_offset"])
        for label, encode_fn in encoders:
            offsets = poi_shift(encode_fn, x_test, ADD_AMOUNT)
            for c in range(N_CAT):
                for a in ADD_AMOUNT:
                    writer.writerow([label, CATEGORIES[c], a, offsets[(c, a)]])
    print(f"已存 {out_path}")


main()
