import os
import sys

import torch

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import PATCHES, result  # noqa: E402
from common.train_log import open_log  # noqa: E402


def env_bool(name, default):
    v = os.environ.get(name)
    return default if v is None else v.lower() in ("1", "true", "yes")


VERSION = "v2_base_tsne"
OUT = result(VERSION, "latents.npz")
SEED = int(os.environ.get("SEED", 0))
CKPT = result(VERSION, f"model_weight/ae_seed{SEED}.pt")

HIDDEN = int(os.environ.get("HIDDEN", 64))
LATENT_DIM = int(os.environ.get("LATENT_DIM", 2))

EPOCHS = int(os.environ.get("EPOCHS", 2000))
BATCH = int(os.environ.get("BATCH", 256))
LR = float(os.environ.get("LR", 1e-2))
LR_MIN = float(os.environ.get("LR_MIN", 1e-3))             # cosine annealing 排程的下限
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-6))

TSNE = env_bool("TSNE", True)              # 關閉可省去建 P 矩陣的成本，訓練只剩 recon loss
PERPLEXITY = float(os.environ.get("PERPLEXITY", 12))     # 每個點的有效鄰居數，取代 FSCE 的 N_NEIGHBORS
TSNE_BATCH = int(os.environ.get("TSNE_BATCH", 0))          # 每個 step 算 KL 用幾個 train patch，0=全部（N 小，全 batch 才是精確的 KL）
TSNE_NORM = os.environ.get("TSNE_NORM", "row")             # "row"：逐點條件機率（每點貢獻 O(1)，收斂快）；"joint"：原始 t-SNE 的全域正規化
TSNE_SCALE = env_bool("TSNE_SCALE", True)                  # 算 Q 前把 z 等向縮放成單位尺度，避免無界排斥項把 latent 越推越大
TSNE_LEARN_SCALE = env_bool("TSNE_LEARN_SCALE", True)      # 再給 Q 一個可學的全域尺度：t-SNE 能撐尺度降 KL，decoder 看到的 z 仍是單位尺度
LAMBDA_TSNE = float(os.environ.get("LAMBDA_TSNE", 0.1))    # t-SNE loss 的權重，warm-up 結束後的最終值
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", 200))      # lambda 從 0 線性升到 LAMBDA_TSNE 所花的 epoch 數
DECAY_EPOCHS = int(os.environ.get("DECAY_EPOCHS", 0))          # >0：warm-up 之後 lambda 再 cosine 退回 0，讓 t-SNE 先塑形、後期把 recon 放開（A）
GAMMA = float(os.environ.get("GAMMA", 1.0))                    # 排斥項的權重，<1 等於 UMAP/LargeVis 的負樣本打折，1.0 是標準 t-SNE（B）
PCGRAD = env_bool("PCGRAD", False)                             # 把 t-SNE 梯度中與 recon 衝突的分量投影掉（C）
# 這個 epoch 之前不選 checkpoint：warm-up 沒跑完時 t-SNE 還沒生效，選到的會是「幾乎沒有 t-SNE」的模型
SELECT_AFTER = int(os.environ.get("SELECT_AFTER", 0))

NOISE_P = float(os.environ.get("NOISE_P", 0.0))             # 破壞強度：thinning 是每個 POI 被丟掉的機率，mask 是整類被抹成 0 的機率
NOISE_MODE = os.environ.get("NOISE_MODE", "thinning")   # "thinning"（逐 POI 丟）或 "mask"（整類歸零）

METRIC = os.environ.get("METRIC", "mae")   # 評估指標："wape"、"mae" 或 "mse"，見 model.METRICS

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
