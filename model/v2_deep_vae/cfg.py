import os
import sys

import torch

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import PATCHES, result  # noqa: E402
from common.train_log import open_log  # noqa: E402

VERSION = "v2_deep_vae"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

HIDDEN = 64
LATENT_DIM = 2

EPOCHS = 300
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-6
SEED = 0

LAMBDA_KL = 1e-3          # KL 散度的固定權重

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
