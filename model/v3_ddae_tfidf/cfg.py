import os
import sys

import torch

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import PATCHES, result  # noqa: E402
from common.train_log import open_log  # noqa: E402

def env_bool(name, default):
    v = os.environ.get(name)
    return default if v is None else v.lower() in ("1", "true", "yes")


VERSION = "v3_ddae_tfidf"
OUT = result(VERSION, "latents.npz")
SEED = int(os.environ.get("SEED", 0))
CKPT = result(VERSION, f"model_weight/ae_seed{SEED}.pt")

HIDDEN = int(os.environ.get("HIDDEN", 64))
LATENT_DIM = int(os.environ.get("LATENT_DIM", 2))

EPOCHS = int(os.environ.get("EPOCHS", 2000))
BATCH = int(os.environ.get("BATCH", 256))
LR = float(os.environ.get("LR", 1e-2))
LR_MIN = float(os.environ.get("LR_MIN", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-6))

FSCE = env_bool("FSCE", True)
GRAPH_MODE = os.environ.get("GRAPH_MODE", "tfidf")   # "tfidf"（TF-IDF 加權 cosine）或 "plain"（log1p count 的 euclidean）
GRAPH_METRIC = os.environ.get("GRAPH_METRIC", "euclidean")   # GRAPH_MODE="plain" 時用
N_NEIGHBORS = int(os.environ.get("N_NEIGHBORS", 10))
N_CLUSTERS = int(os.environ.get("N_CLUSTERS", 8))
EDGE_BATCH = int(os.environ.get("EDGE_BATCH", 256))
LAMBDA_FSCE = float(os.environ.get("LAMBDA_FSCE", 0.25))
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", 200))

PCGRAD = env_bool("PCGRAD", True)

NOISE_P = float(os.environ.get("NOISE_P", 0.3))
NOISE_MODE = os.environ.get("NOISE_MODE", "thinning")

METRIC = os.environ.get("METRIC", "mae")

SNAPSHOT_PERCENTS = [int(p) for p in os.environ.get(
    "SNAPSHOT_PERCENTS", "10,20,25,30,40,50,60,70,80,90,100").split(",")]

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
