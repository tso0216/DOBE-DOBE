"""計算 experiment/model/ 六個 AE/DAE 變體（不含 VAE）在測試集上的重建 MAE"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import N_CAT, PATCHES, make_split  # noqa: E402
from model.v2_ddae_base.dataset import Patches  # noqa: E402

MODEL_DIR = os.path.join(HERE, "..", "model")
SEED = 0

# (權重檔名，圖上顯示的標籤)
MODELS = [
    ("ae.pt", "AE"),
    ("ae_fsce.pt", "AE+fsce"),
    ("ae_tfidf.pt", "AE+fsce+tfidf"),
    ("dae.pt", "DAE"),
    ("dae_fsce.pt", "DAE+fsce"),
    ("dae_tfidf.pt", "DAE+fsce+tfidf (Ours)"),
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


def load_model(path):
    """path：checkpoint 檔案路徑。回傳 AE，已載入權重並設為 eval。"""
    sd = torch.load(path, map_location="cpu")
    hidden = sd["encoder.0.weight"].shape[0]
    latent_dim = sd["decoder.0.weight"].shape[1]
    model = AE(latent_dim, hidden)
    model.load_state_dict(sd)
    model.eval()
    return model


def test_mae(model, data, idx, batch=256):
    """model：AE。data：Patches。idx：測試集 patch 編號 tensor。回傳整個測試集的平均 MAE。"""
    diffs = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            x = data.agg(idx[i:i + batch])
            log_lam = model(x)[1]
            diffs.append((torch.exp(log_lam) - x).abs())
    return torch.cat(diffs).mean().item()


def main():
    data = Patches(PATCHES)
    _, _, test_idx = make_split(data.lat, data.lon, seed=SEED)

    rows = []
    for name, label in MODELS:
        model = load_model(os.path.join(MODEL_DIR, name))
        mae = test_mae(model, data, test_idx)
        rows.append((label, mae))
        print(f"{label}: test_mae = {mae:.5f}")

    out = os.path.join(HERE, "data.csv")
    with open(out, "w") as f:
        f.write("model,mae\n")
        for label, mae in rows:
            f.write(f"{label},{mae:.5f}\n")
    print(f"已存 {out}")


main()
