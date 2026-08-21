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

PARAMETER = {
    "LAMBDA_FSCE": [0.005,0.01,0.05,0.25,0.75],
    # "HIDDEN": [16, 32, 64],
}


def run_seed(version, seed, overrides=None):
    """跑單一 seed 的 train.py，回傳最終 test_dev 數值。overrides 會蓋掉 cfg 對應的環境變數。"""
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
    return float(matches[-1])


def run_multi_seed(version, amount, overrides=None, label=""):
    """跑 amount 個 seed，印出並回傳 test_dev 的 mean/std/max/min/q1/mid/q3。"""
    scores = []
    for seed in range(amount):
        score = run_seed(version, seed, overrides)
        print(f"{label}seed {seed}: test_dev {score:.5f}")
        scores.append(score)

    scores = np.array(scores)
    q1, med, q3 = np.percentile(scores, [25, 50, 75])
    print(f"{label}mean {scores.mean():.5f} | std {scores.std():.5f} | "
          f"max {scores.max():.5f} | min {scores.min():.5f} | "
          f"q1 {q1:.5f} | median {med:.5f} | q3 {q3:.5f}")
    print(f"{scores.mean():.5f},{scores.std():.5f},{scores.max():.5f},{scores.min():.5f},"
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
            run_multi_seed(version, args.amount, {param: value}, label=f"[{param}={value}] ")


if __name__ == "__main__":
    main()
