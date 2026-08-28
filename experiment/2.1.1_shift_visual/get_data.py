"""計算六個 AE/DAE 變體，同時對所有 POI 類別加入 POI 後每個 patch 的 latent 位移軌跡"""
import csv
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from common.dataset import N_CAT, PATCHES, make_split  # noqa: E402
from model.other.v2_ddae_base.dataset import Patches  # noqa: E402

MODEL_DIR = os.path.join(HERE, "..", "model")
ADD_AMOUNT = [1, 2, 3]      # 想看多段位移就改成 [1, 2, 3] 之類的遞增清單
SEED = 0

# (權重檔名，圖上顯示的標籤)
MODELS = [
    ("ae/fold3_mae.pt", "AE"),
    ("ae_fsce/fold3_mae.pt", "AE+fsce"),
    ("ae_fsce_tfidf/fold3_mae.pt", "AE+fsce+tfidf"),
    ("ddae/fold3_mae.pt", "DAE"),
    ("ddae_fsce/fold3_mae.pt", "DAE+fsce"),
    ("ddae_fsce_tfidf/fold3_mae.pt", "DAE+fsce+tfidf (Ours)"),
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


def shift_path(encode_fn, x, amounts):
    """encode_fn：(B, N_CAT) tensor -> (B, 2) tensor 的 encode 函式。x：(B, N_CAT) count tensor。
    amounts：由小到大的加入量清單，每步對所有類別同時加入。
    回傳 (B, len(amounts)+1, 2) numpy array，第 0 格是原始 latent，之後依序是加入各個量之後的 latent。
    """
    zs = [encode_fn(x)]
    for a in amounts:
        x_shift = x.clone()
        x_shift += a
        zs.append(encode_fn(x_shift))
    return torch.stack(zs, dim=1).numpy()


def main():
    data = Patches(PATCHES)
    _, _, test_idx = make_split(data.lat, data.lon, seed=SEED)
    x_test = data.agg(test_idx)

    out_path = os.path.join(HERE, "data.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "sample_id", "step", "amount", "z1", "z2"])
        for name, label in MODELS:
            model = load_model(os.path.join(MODEL_DIR, name))
            path = shift_path(lambda x, model=model: encode(model, x), x_test, ADD_AMOUNT)
            cum_amounts = [0] + list(ADD_AMOUNT)
            for sample_id in range(path.shape[0]):
                for step, amount in enumerate(cum_amounts):
                    z1, z2 = path[sample_id, step]
                    writer.writerow([label, sample_id, step, amount, z1, z2])
    print(f"已存 {out_path}")


main()
