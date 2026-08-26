import os
import sys

import torch

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import PATCHES, result  # noqa: E402
from common.train_log import open_log  # noqa: E402

VERSION = "v2_deep_vae"
SEED = int(os.environ.get("SEED", 0))

HIDDEN = int(os.environ.get("HIDDEN", 64))
LATENT_DIM = int(os.environ.get("LATENT_DIM", 2))

EPOCHS = int(os.environ.get("EPOCHS", 300))
BATCH = int(os.environ.get("BATCH", 256))
LR = float(os.environ.get("LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-6))

LAMBDA_KL = float(os.environ.get("LAMBDA_KL", 1e-3))          # KL 散度的固定權重

METRIC = os.environ.get("METRIC", "mae")   # 訓練途中 log 顯示用的指標
CKPT_METRICS = [m.strip() for m in os.environ.get(
    "CKPT_METRICS", "mae,mse,wape,deviance").split(",")]   # 每個指標各存一份最佳 checkpoint

N_SPLITS = int(os.environ.get("N_SPLITS", 5))
TEST_FRAC = float(os.environ.get("TEST_FRAC", 0.2))

ANALYZE_FOLD = int(os.environ.get("ANALYZE_FOLD", 1))
ANALYZE_METRIC = os.environ.get("ANALYZE_METRIC", METRIC)

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def ckpt_path(fold, metric):
    """傳入：fold 編號（1 起算）、選 checkpoint 用的指標名。回傳：該份 checkpoint 的路徑。"""
    return result(VERSION, f"model_weight/fold{fold}_{metric}.pt")


def latents_path(fold, metric):
    """傳入：fold 編號（1 起算）、選 checkpoint 用的指標名。回傳：該份模型 latent 輸出的 npz 路徑。"""
    return result(VERSION, f"latents/fold{fold}_{metric}.npz")
