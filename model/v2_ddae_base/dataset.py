import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from common.dataset import N_CAT  # noqa: E402,F401


class Patches:
    """稀疏點列表；agg() 把整個 patch 聚合成一個 (N_CAT,) 的 count 向量。"""

    def __init__(self, path):
        d = np.load(path)
        self.cat = torch.from_numpy(d["cat"].astype(np.int64))
        self.offsets = torch.from_numpy(d["offsets"])
        self.n_poi = d["n_poi"]
        self.lat = d["center_lat"]
        self.lon = d["center_lon"]
        self.n = len(self.n_poi)

    def agg(self, idx):
        """idx 是 patch 編號的 tensor，回傳 (B,N_CAT) 的整包類別 count 向量。"""
        b = len(idx)
        starts, ends = self.offsets[idx], self.offsets[idx + 1]
        lens = ends - starts
        pos = torch.repeat_interleave(starts, lens) + \
            torch.arange(int(lens.sum())) - torch.repeat_interleave(
                torch.cumsum(lens, 0) - lens, lens)
        owner = torch.repeat_interleave(torch.arange(b), lens)
        cat = self.cat[pos]

        flat = owner * N_CAT + cat
        counts = torch.bincount(flat, minlength=b * N_CAT)
        return counts.view(b, N_CAT).float()


def make_split(lat, lon, mode="random", seed=0, val_frac=0.1, test_frac=0.1,
               cell=0.02):
    """lat/lon 是每個 patch 的中心座標，回傳 (train_idx, val_idx, test_idx)
    三個 torch long tensor。

    mode="random"：純隨機切 patch，跟舊版 log 的切法一致，但相鄰 patch 會散在
    不同 split，test 等於在做空間內插、分數偏樂觀。
    mode="spatial"：先把 lat/lon 切成 cell 度的粗網格，整塊分到同一個 split，
    相鄰 patch 不跨 split，測的是真正的空間外推。
    """
    n = len(lat)
    rng = np.random.default_rng(seed)

    if mode == "random":
        member = [np.array([i]) for i in range(n)]
    elif mode == "spatial":
        key = (np.floor(np.asarray(lat) / cell).astype(np.int64) * 1_000_000 +
               np.floor(np.asarray(lon) / cell).astype(np.int64))
        member = [np.flatnonzero(key == b) for b in np.unique(key)]
    else:
        raise ValueError(f"未知的 split mode：{mode}")

    n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
    bucket, filled = [[], [], []], [0, 0]   # filled：test、val 目前裝了幾個 patch
    for u in rng.permutation(len(member)):
        idx = member[u]
        which = 2 if filled[0] < n_test else (1 if filled[1] < n_val else 0)
        bucket[which].append(idx)
        if which >= 1:
            filled[2 - which] += len(idx)

    def cat(xs):
        a = np.concatenate(xs) if xs else np.empty(0, dtype=np.int64)
        return torch.from_numpy(np.sort(a.astype(np.int64)))

    return cat(bucket[0]), cat(bucket[1]), cat(bucket[2])


def corrupt(x, p, mode="thinning", generator=None):
    """把乾淨的 count 向量破壞成 DAE 的輸入。

    x：(B,N_CAT) 的乾淨 count 向量（float，值是非負整數）。
    p：破壞強度 ∈[0,1)。thinning 是每個 POI 被丟掉的機率，mask 是每一類
       整維被抹成 0 的機率。p=0 時直接回傳 x。
    mode："thinning" 或 "mask"。
    generator：torch.Generator，給定就用它抽亂數，讓每個 epoch 的破壞可重現。

    回傳跟 x 同 shape 的 (B,N_CAT) tensor，已經除以 1-p 做過尺度補償，
    期望值等於 x，所以可以直接餵進跟推論時同一個 encoder。
    """
    if p <= 0:
        return x
    keep = 1.0 - p
    if mode == "thinning":
        # binomial 沒有吃 generator 的版本，用 x 個 Bernoulli 的和等價實作：
        # 每個 patch 的每一類最多 max_c 個 POI，各自擲一次銅板再依真實 count 遮掉
        max_c = int(x.max().item())
        if max_c == 0:
            return x
        coin = torch.rand(x.shape + (max_c,), generator=generator,
                          device=x.device) < keep
        alive = torch.arange(max_c, device=x.device) < x.unsqueeze(-1)
        noisy = (coin & alive).sum(dim=-1).float()
    elif mode == "mask":
        m = (torch.rand(x.shape, generator=generator, device=x.device) < keep)
        noisy = x * m.float()
    else:
        raise ValueError(f"未知的 mode：{mode}")
    return noisy / keep
