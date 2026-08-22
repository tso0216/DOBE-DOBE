import os
import sys

import torch

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import PATCHES, result  # noqa: E402
from common.train_log import open_log  # noqa: E402

VERSION = "v2_deep_vae"
OUT = result(VERSION, "latents.npz")
SEED = int(os.environ.get("SEED", 0))
CKPT = result(VERSION, f"model_weight/ae_seed{SEED}.pt")

HIDDEN = int(os.environ.get("HIDDEN", 64))
LATENT_DIM = int(os.environ.get("LATENT_DIM", 2))

EPOCHS = int(os.environ.get("EPOCHS", 300))
BATCH = int(os.environ.get("BATCH", 256))
LR = float(os.environ.get("LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-6))

LAMBDA_KL = float(os.environ.get("LAMBDA_KL", 1e-3))          # KL 散度的固定權重

METRIC = os.environ.get("METRIC", "mae")   # 評估指標："wape"、"mae" 或 "mse"，見 model.METRICS

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
