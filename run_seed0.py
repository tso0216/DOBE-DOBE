import os
import re
import subprocess
import sys
from itertools import product

ROOT = os.path.dirname(os.path.abspath(__file__))
BEST_EPOCH_RE = re.compile(r"最佳 checkpoint：epoch (\d+)")

models = {1: 'v2_ddae_base', 2: 'v2_deep_ae', 3: 'v2_base_tsne'}
models_name = {1: 'ours', 2: 'AE', 3: 'tSNE'}
model = 3
tunes = {  
    1: {'LAMBDA_FSCE': [1,0.5,0.1,0.01,0], 'METRIC':'mse'},
    2: {'LAMBDA_FSCE': [0.5]},
    3: {'LAMBDA_TSNE':  [1,0.5,0.1,0.01,0],'METRIC':'mse'},
}
tune = tunes[model]


def run_one(version, params):
    env = os.environ.copy()
    for k, v in params.items():
        env[k] = str(v)

    train_py = os.path.join(ROOT, 'model', version, 'train.py')
    proc = subprocess.run(
        [sys.executable, train_py],
        cwd=os.path.dirname(train_py),
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None, None, proc.stderr

    log_path = os.path.join(ROOT, 'model', version, 'result', 'result.log')
    with open(log_path, encoding='utf-8') as f:
        content = f.read()
    metric = params.get('METRIC', os.environ.get('METRIC', 'mae'))
    matches = re.findall(rf'test_{re.escape(metric)}\s+([\d.]+)', content)
    score = float(matches[-1]) if matches else None
    epochs = BEST_EPOCH_RE.findall(content)
    best_epoch = int(epochs[-1]) if epochs else None
    return score, best_epoch, content


def main():
    # tune 裡值是 list 的當作要掃的參數軸，其餘（純量）當固定參數
    axes = {k: v for k, v in tune.items() if isinstance(v, list)}
    fixed = {k: v for k, v in tune.items() if not isinstance(v, list)}

    version = models[model]
    out_path = os.path.join(ROOT, 'tune_results.log')
    with open(out_path, 'w', encoding='utf-8') as out:
        for combo in product(*axes.values()):
            params = dict(zip(axes.keys(), combo))
            params.update(fixed)
            tag = ' '.join(f'{k}={v}' for k, v in params.items())

            score, best_epoch, detail = run_one(version, params)
            if score is None:
                line = f"{models_name[model]} {tag} 失敗\n{detail}\n"
            else:
                metric = params.get('METRIC', 'mae')
                epoch_txt = f"epoch {best_epoch}" if best_epoch is not None else "epoch ?"
                line = f"{models_name[model]} {tag} {metric}={score:.5f}（{epoch_txt}）"
            print(line)
            out.write(line)
            out.flush()


if __name__ == '__main__':
    main()
