"""
Quantitative Evaluation & Metrics Module
"""

import math
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score, adjusted_rand_score
from common.dataset import CATEGORIES, N_CAT


def evaluate_latent_cohesion(z_np, labels):
    """
    Evaluates cluster cohesion metrics for a set of labels inside 2D Latent space Z.
    Handles noise points (labeled as -1) appropriately.
    """
    valid_mask = labels != -1
    n_clusters = len(set(labels[valid_mask])) if valid_mask.sum() > 0 else 0

    if n_clusters < 2 or valid_mask.sum() <= n_clusters:
        return {
            "Silhouette Score": np.nan,
            "Davies-Bouldin Index": np.nan,
            "Calinski-Harabasz Index": np.nan,
            "Noise Percentage (%)": round((labels == -1).sum() / len(labels) * 100, 2)
        }

    z_valid = z_np[valid_mask]
    lbls_valid = labels[valid_mask]

    sil = silhouette_score(z_valid, lbls_valid)
    dbi = davies_bouldin_score(z_valid, lbls_valid)
    chi = calinski_harabasz_score(z_valid, lbls_valid)
    noise_pct = round((labels == -1).sum() / len(labels) * 100, 2)

    return {
        "Silhouette Score": round(sil, 4),
        "Davies-Bouldin Index": round(dbi, 4),
        "Calinski-Harabasz Index": round(chi, 2),
        "Noise Percentage (%)": noise_pct
    }


def evaluate_reconstruction(model, data, test_idx, batch_size=256):
    """
    Evaluates reconstruction performance (RMSE, MAE, WAPE, MAPE, Poisson NLL).
    """
    diffs, counts, nlls = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_idx), batch_size):
            idx = test_idx[i:i + batch_size]
            x = data.agg(idx)
            out = model(x)
            log_lam = out[1] if isinstance(out, tuple) else out
            lam = torch.exp(log_lam)
            diffs.append(lam - x)
            counts.append(x)
            nll = (lam - x * log_lam).mean(dim=1)
            nlls.append(nll)

    diff = torch.cat(diffs)
    x = torch.cat(counts)
    nll = torch.cat(nlls).mean().item()

    mse = diff.pow(2).mean().item()
    rmse = math.sqrt(mse)
    mae = diff.abs().mean().item()

    mask = x > 0
    mape = (diff[mask].abs() / x[mask]).mean().item() * 100.0
    wape = (diff.abs().sum() / x.sum().clamp_min(1e-8)).item() * 100.0

    return {
        "RMSE": rmse,
        "MAPE (%)": mape,
        "WAPE (%)": wape,
        "MAE": mae,
        "Poisson NLL": nll
    }


def compute_standardized_latent_shift(model, data, test_idx, amounts=[1, 3, 5]):
    """
    Computes POI Perturbation Latent Shift (+1, +3, +5) normalized by latent dimension std.
    """
    def encode_z(m, x):
        with torch.no_grad():
            res = m.encode(x)
            return res[0] if isinstance(res, tuple) else res

    with torch.no_grad():
        x_base = data.agg(test_idx)
        z_base = encode_z(model, x_base)
        z_std_dims = z_base.std(dim=0).clamp_min(1e-6)

    results = {}
    for amount in amounts:
        cat_shifts = []
        for c in range(N_CAT):
            x_pert = x_base.clone()
            x_pert[:, c] += amount
            with torch.no_grad():
                z_pert = encode_z(model, x_pert)
            shift_std = ((z_pert - z_base) / z_std_dims).pow(2).sum(dim=1).sqrt().mean().item()
            cat_shifts.append(shift_std)
        results[f"+{amount}"] = cat_shifts

    df_shift = pd.DataFrame(results, index=CATEGORIES)
    return df_shift, z_base.numpy(), z_std_dims.numpy()
