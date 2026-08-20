import importlib
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(version):
    model_dir = os.path.join(ROOT, "..", "model", version)
    sys.path.insert(0, model_dir)
    for name in ("api", "cfg", "dataset", "model"):
        sys.modules.pop(name, None)
    api = importlib.import_module("api")
    cfg = importlib.import_module("cfg")
    dataset = importlib.import_module("dataset")
    from common.dataset import PATCHES

    data = dataset.Patches(PATCHES)
    train_idx, _, test_idx = dataset.make_split(data.lat, data.lon, seed=cfg.SEED)

    sys.path.remove(model_dir)
    return api, data, train_idx, test_idx


def report(version, mse, mae, mape, n):
    print(f"[{version}] test set（{n} 個 patch）：")
    print(f"  MSE  = {mse:.6f}")
    print(f"  MAE  = {mae:.6f}")
    print(f"  MAPE = {mape:.2f}%")


for version in ("v2_ddae_base", "v2_deep_ae"):
    api, data, train_idx, test_idx = load(version)
    model = api.load_model()
    mse, mae, mape = api.test_mse(model, data, test_idx)
    report(version, mse, mae, mape, len(test_idx))

api, data, train_idx, test_idx = load("PCA")
model = api.fit(data, train_idx)
mse, mae, mape = api.test_mse(model, data, test_idx)
report("PCA", mse, mae, mape, len(test_idx))
