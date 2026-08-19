import os
import sys

import torch

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.train_log import open_log  # noqa: E402

VERSION = "v2_ddae_moe"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
os.makedirs(RESULT_DIR, exist_ok=True)

HIDDEN = 64
LATENT_DIM = 2

EPOCHS = 3000
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-6
SEED = 0

N_NEIGHBORS = 15         # 建高維 fuzzy graph 的 kNN 數，跟 data/patch/umap_grid.py 一致
GRAPH_METRIC = "euclidean"  # 在 log1p 上算，組成與總量都敏感——跟 Poisson NLL 的要求一致
EDGE_BATCH = 256          # 每個 step 抽的正樣本邊數，負樣本抽等量
LAMBDA_FSCE = 0.01        # FSCE loss 的權重，warm-up 結束後的最終值
WARMUP_EPOCHS = 200       # lambda 從 0 線性升到 LAMBDA_FSCE 所花的 epoch 數

NOISE_P = 0.3             # 破壞強度：thinning 是每個 POI 被丟掉的機率

N_EXPERTS = 4             # encoder 每組的 expert 數
TOP_K = 4                 # 每組每次啟用幾個 expert
LAMBDA_MOE = 0.01         # load-balance loss 權重，避免所有 token 擠同一個 expert

OUT = os.path.join(RESULT_DIR, "latents.npz")
CKPT = os.path.join(RESULT_DIR, "ae.pt")

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
