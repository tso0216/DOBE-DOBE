import os
import sys
import torch


model_name = "v2_ddae_base"
ckpt = "hello/model_pt/ddae_300_test.pt"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model", model_name))
from api import load_model  # type: ignore
from cfg import SEED  # type: ignore
from dataset import Patches  # type: ignore
from model import poisson_deviance  # type: ignore
from common.dataset import PATCHES, make_split

model = load_model(ckpt=ckpt)


def compute_mae(model, x):
    model.eval()
    with torch.no_grad():
        _, log_lam = model(x)
        return (torch.exp(log_lam) - x).abs().mean().item()


def compute_mse(model, x):
    model.eval()
    with torch.no_grad():
        _, log_lam = model(x)
        return (torch.exp(log_lam) - x).pow(2).mean().item()


def compute_mape(model, x):
    model.eval()
    with torch.no_grad():
        _, log_lam = model(x)
        diff = torch.exp(log_lam) - x
        mask = x > 0
        return (diff[mask].abs() / x[mask]).mean().item() * 100


def compute_deviance(model, x):
    model.eval()
    with torch.no_grad():
        _, log_lam = model(x)
        return poisson_deviance(log_lam, x).mean().item()


data = Patches(PATCHES)
_, _, test_idx = make_split(data.lat, data.lon, seed=SEED)
x_test = data.agg(test_idx)

print(f"MAE : {compute_mae(model, x_test):.3f}")
print(f"MSE : {compute_mse(model, x_test):.3f}")
print(f"MAPE: {compute_mape(model, x_test):.3f}")
print(f"Deviance: {compute_deviance(model, x_test):.3f}")
