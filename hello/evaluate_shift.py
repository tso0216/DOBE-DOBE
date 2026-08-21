import csv
import glob
import os
import sys

models = ['v2_ddae_base','v2_deep_ae','v2_deep_vae']
model_name = models[0]
weight_dir = 'multi_seed_result/model_weight/ours/model_weight2000'
amounts = [1, 3, 5]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model", model_name))
from api import load_model, encode  # type: ignore
from cfg import SEED  # type: ignore
from dataset import Patches  # type: ignore
from common.dataset import PATCHES, CATEGORIES, N_CAT, make_split

ckpts = sorted(glob.glob(os.path.join(ROOT, weight_dir, "ae_seed*.pt")))
seed_models = [load_model(ckpt=c) for c in ckpts]

data = Patches(PATCHES)
_, _, test_idx = make_split(data.lat, data.lon, seed=SEED)
x_test = data.agg(test_idx)

z0s = [encode(m, x_test) for m in seed_models]

rows = []
for c in range(N_CAT):
    print(f"[{CATEGORIES[c]}]")
    for a in amounts:
        x_shift = x_test.clone()
        x_shift[:, c] += a
        dists = [(encode(m, x_shift) - z0).norm(dim=1).mean().item()
                  for m, z0 in zip(seed_models, z0s)]
        dist = sum(dists) / len(dists)
        print(f"  +{a}: average offset : {dist:.4f} ({len(dists)} seeds)")
        rows.append([CATEGORIES[c], a, dist])


out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shift_result.csv")
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "amount", "avg_offset"])
    writer.writerows(rows)
