"""計算 experiment/model/ 六個 AE/DAE 變體，在各 POI 類別加入不同數量後的 latent 平均偏移"""
import csv
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import CATEGORIES, N_CAT, PATCHES, make_split  # noqa: E402
from model.other.v2_ddae_base.dataset import Patches  # noqa: E402

MODEL_DIR = os.path.join(HERE, "..", "model")
ADD_AMOUNT = [1, 2, 3]
SEED = 0

# (權重檔名，圖上顯示的標籤)
MODELS = [
    ("ae/fold3_mae.pt", "AE"),
    ("ae_fsce/fold3_mae.pt", "AE+fsce"),
    ("ae_fsce_tfidf/fold3_mae.pt", "AE+fsce+tfidf"),
    ("dae.pt/fold3_mae", "DAE"),
    ("dae_fsce/fold3_mae.pt", "DAE+fsce"),
    ("dae_fsce_tfidf/fold3_mae.pt", "DAE+fsce+tfidf (Ours)"),
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


def encode(model, x):
    """model：AE。x：(B, N_CAT) count tensor。回傳 (B, latent_dim) latent。"""
    with torch.no_grad():
        return model.encoder(x)


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

    out_path = os.path.join(HERE, "data.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "category", "amount", "avg_offset"])
        for name, label in MODELS:
            model = load_model(os.path.join(MODEL_DIR, name))
            offsets = poi_shift(lambda x, model=model: encode(model, x), x_test, ADD_AMOUNT)
            for c in range(N_CAT):
                for a in ADD_AMOUNT:
                    writer.writerow([label, CATEGORIES[c], a, offsets[(c, a)]])
    print(f"已存 {out_path}")


main()
