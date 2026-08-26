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
SEED = int(os.environ.get("SEED", 0))

HIDDEN = int(os.environ.get("HIDDEN", 64))
LATENT_DIM = int(os.environ.get("LATENT_DIM", 2))

EPOCHS = int(os.environ.get("EPOCHS", 2000))
BATCH = int(os.environ.get("BATCH", 256))
LR = float(os.environ.get("LR", 1e-2))
LR_MIN = float(os.environ.get("LR_MIN", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-6))


FSCE = env_bool("FSCE", True)
GRAPH_MODE = os.environ.get("GRAPH_MODE", "tfidf")   # "tfidf"（TF-IDF 加權 cosine）或 "plain"（log1p count 的 euclidean）
NOISE_P = float(os.environ.get("NOISE_P", 0.3))


N_NEIGHBORS = int(os.environ.get("N_NEIGHBORS", 10))
N_CLUSTERS = int(os.environ.get("N_CLUSTERS", 8))
EDGE_BATCH = int(os.environ.get("EDGE_BATCH", 256))
LAMBDA_FSCE = float(os.environ.get("LAMBDA_FSCE", 0.25))
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", 200))

PCGRAD = env_bool("PCGRAD", True)

NOISE_MODE = os.environ.get("NOISE_MODE", "thinning")

METRIC = os.environ.get("METRIC", "mae")   # 訓練途中 log 顯示、analyze 腳本預設用的指標
CKPT_METRICS = [m.strip() for m in os.environ.get(
    "CKPT_METRICS", "mae,mse,wape,deviance").split(",")]   # 每個指標各存一份最佳 checkpoint

N_SPLITS = int(os.environ.get("N_SPLITS", 5))
TEST_FRAC = float(os.environ.get("TEST_FRAC", 0.2))

ANALYZE_FOLD = int(os.environ.get("ANALYZE_FOLD", 1))
ANALYZE_METRIC = os.environ.get("ANALYZE_METRIC", METRIC)

SNAPSHOT_PERCENTS = [int(p) for p in os.environ.get(
    "SNAPSHOT_PERCENTS", "10,20,25,30,40,50,60,70,80,90,100").split(",")]

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def ckpt_path(fold, metric):

    return result(VERSION, f"model_weight/fold{fold}_{metric}.pt")


def latents_path(fold, metric):
    """傳入：fold 編號（1 起算）、選 checkpoint 用的指標名。回傳：該份模型 latent 輸出的 npz 路徑。"""
    return result(VERSION, f"latents/fold{fold}_{metric}.npz")
