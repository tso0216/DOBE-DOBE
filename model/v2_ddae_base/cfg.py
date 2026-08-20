import os
import sys

import torch

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import PATCHES, result  # noqa: E402
from common.train_log import open_log  # noqa: E402

VERSION = "v2_ddae_base"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

HIDDEN = 64
LATENT_DIM = 2

EPOCHS = 300
BATCH = 128
LR = 1e-2
LR_MIN = 1e-5           
WEIGHT_DECAY = 5e-4
SEED = 0

N_NEIGHBORS = 18         # 建高維 fuzzy graph 的 kNN 數，跟 data/patch/umap_grid.py 一致
GRAPH_METRIC = "euclidean"  # 在 log1p 上算，組成與總量都敏感——跟 Poisson NLL 的要求一致
EDGE_BATCH = 256          # 每個 step 抽的正樣本邊數，負樣本抽等量
LAMBDA_FSCE = 1        # FSCE loss 的權重，warm-up 結束後的最終值
WARMUP_EPOCHS = 24       # lambda 從 0 線性升到 LAMBDA_FSCE 所花的 epoch 數

NOISE_P = 0.118             # 破壞強度：thinning 是每個 POI 被丟掉的機率，mask 是整類被抹成 0 的機率
NOISE_MODE = "thinning"   # "thinning"（逐 POI 丟）或 "mask"（整類歸零）

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
