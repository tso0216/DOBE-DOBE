"""多種子訓練同一模型，統計 test_dev 的 mean / std / max / min。"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "ours": "v2_ddae_base",
    "ae": "v2_deep_ae",
    "vae": "v2_deep_vae",
}

TEST_DEV_RE = re.compile(r"test_dev ([0-9.eE+-]+)（")
BEST_EPOCH_RE = re.compile(r"最佳 checkpoint：epoch (\d+)")

PARAMETER = {
    "LR": [0.014,0.016],
    # "HIDDEN": [16, 32, 64],
}


def run_seed(version, seed, overrides=None):
    """跑單一 seed 的 train.py，回傳 (最終 test_dev 數值, 最佳 checkpoint 的 epoch，找不到為 None)。overrides 會蓋掉 cfg 對應的環境變數。"""
    env = os.environ.copy()
    env["SEED"] = str(seed)
    if overrides:
        for k, v in overrides.items():
            env[k] = str(v)
    proc = subprocess.run(
        [sys.executable, "train.py"],
        cwd=os.path.join(ROOT, "model", version),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"seed {seed} 失敗：\n{proc.stderr}")
    matches = TEST_DEV_RE.findall(proc.stdout)
    if not matches:
        raise RuntimeError(f"seed {seed} 找不到 test_dev：\n{proc.stdout[-2000:]}")
    epochs = BEST_EPOCH_RE.findall(proc.stdout)
    best_epoch = int(epochs[-1]) if epochs else None
    return float(matches[-1]), best_epoch


def run_multi_seed(version, amount, overrides=None, label=""):
    """跑 amount 個 seed，印出並回傳 test_dev 的 mean/std/max/min/q1/mid/q3。"""
    scores = []
    for seed in range(amount):
        score, best_epoch = run_seed(version, seed, overrides)
        epoch_txt = f"epoch {best_epoch}" if best_epoch is not None else "epoch ?"
        print(f"seed {seed}: test_dev {score:.5f}（{epoch_txt}）")
        scores.append(score)

    scores = np.array(scores)
    q1, med, q3 = np.percentile(scores, [25, 50, 75])
    print(f"{label},{scores.mean():.5f},{scores.std():.5f},{scores.max():.5f},{scores.min():.5f},"
          f"{q1:.5f},{med:.5f},{q3:.5f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS, nargs="?", default="ddae_base")
    parser.add_argument("--amount", type=int, default=5, help="要跑的 seed 數量（0..amount-1）")
    args = parser.parse_args()

    version = MODELS[args.model]

    if not PARAMETER:
        run_multi_seed(version, args.amount)
        return

    for param, values in PARAMETER.items():
        for value in values:
            run_multi_seed(version, args.amount, {param: value}, label=f"[{value}]")


if __name__ == "__main__":
    main()
