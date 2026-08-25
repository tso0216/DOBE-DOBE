import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from common.dataset import N_CAT, PATCHES, make_split  # noqa: E402
from model.v2_ddae_base.dataset import Patches  # noqa: E402

weight_dir = 'model_weight'
SEED = 0


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


def test_mse(model, data, idx, batch=256):
    """model：AE 或 VAE。data：Patches。idx：測試集 patch 編號 tensor。回傳整個測試集的平均 MSE。"""
    diffs = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            x = data.agg(idx[i:i + batch])
            log_lam = model(x)[1]
            diffs.append((torch.exp(log_lam) - x) ** 2)
    return torch.cat(diffs).mean().item()


def main():
    data = Patches(PATCHES)
    _, _, test_idx = make_split(data.lat, data.lon, seed=SEED)

    weight_path = os.path.join(ROOT, weight_dir)
    for name in sorted(os.listdir(weight_path)):
        if not name.endswith(".pt"):
            continue
        model = load_model(os.path.join(weight_path, name))
        score = test_mse(model, data, test_idx)
        print(f"{name}: test_mse = {score:.5f}")


if __name__ == "__main__":
    main()
