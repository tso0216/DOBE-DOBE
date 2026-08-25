"""
DOBE-DOBE Master Execution Pipeline
==================================
Runs end-to-end data processing, model retraining (optional), Optuna hyperparameter tuning,
KMeans/GMM/DBSCAN clustering, reconstruction & latent cohesion evaluation,
POI perturbation shift experiments, and generates all visual charts.

Usage Examples:
1. Standard evaluation & clustering:
   python run_pipeline.py

2. Re-train models with custom parameters:
   python run_pipeline.py --retrain --epochs 100 --lr 0.001 --n_clusters 8
"""

import os
import sys
import argparse
import importlib
import math
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from common.dataset import PATCHES, CATEGORIES, CAT_ZH, N_CAT, make_split
from clustering import (
    ClusteringPipeline,
    ModelTrainer,
    evaluate_latent_cohesion,
    evaluate_reconstruction,
    compute_standardized_latent_shift,
    Visualizer
)

BATCH = 256

MODELS_CONFIG = [
    {"name": "v2_deep_ae", "dir": "v2_deep_ae"},
    {"name": "v2_ddae_base", "dir": "v2_ddae_base"},
    {"name": "v3_ddae_tfidf", "dir": "v3_ddae_tfidf"},
]


class PatchesData:
    def __init__(self, path):
        d = np.load(path)
        self.cat = torch.from_numpy(d["cat"].astype(np.int64))
        self.offsets = torch.from_numpy(d["offsets"])
        self.n_poi = d["n_poi"]
        self.lat = d["center_lat"]
        self.lon = d["center_lon"]
        self.n = len(self.n_poi)

    def agg(self, idx):
        b = len(idx)
        starts, ends = self.offsets[idx], self.offsets[idx + 1]
        lens = ends - starts
        pos = torch.repeat_interleave(starts, lens) + \
            torch.arange(int(lens.sum())) - torch.repeat_interleave(
                torch.cumsum(lens, 0) - lens, lens)
        owner = torch.repeat_interleave(torch.arange(b), lens)
        cats = self.cat[pos]
        flat = owner * N_CAT + cats
        counts = torch.bincount(flat, minlength=b * N_CAT)
        return counts.view(b, N_CAT).float()


def load_ae_model(model_dir, seed=0):
    m_path = os.path.abspath(os.path.join(ROOT, "model", model_dir))
    for mod_name in ["model", "cfg", "moe", "dataset", "api"]:
        sys.modules.pop(mod_name, None)
    sys.path.insert(0, m_path)
    
    ckpt_candidates = [
        os.path.join(m_path, "result", "model_weight", f"ae_seed{seed}.pt"),
        os.path.join(m_path, "result", "ae.pt"),
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if os.path.exists(c):
            ckpt_path = c
            break
            
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for model: {model_dir}")
        
    model_mod = importlib.import_module("model")
    m = model_mod.AE(latent_dim=2)
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.eval()
    return m


def main(retrain=False, epochs=100, lr=1e-3, n_clusters=8, seed=0, optuna_tune=True, optuna_trials=30):
    print("=" * 80)
    print(" DOBE-DOBE Target Model Evaluation & Multi-Clustering Master Pipeline")
    print("=" * 80)
    print(f" Parameters: retrain={retrain}, epochs={epochs}, lr={lr}, n_clusters={n_clusters}, seed={seed}, optuna_tune={optuna_tune}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # 0. Re-train models if requested
    if retrain:
        print("\n[Step 0] Re-training Models with Custom Parameters...")
        trainer = ModelTrainer(root_dir=ROOT)
        trainer.train_all(epochs=epochs, lr=lr, n_clusters=n_clusters, seed=seed)

    # 1. Load Data & Split
    data = PatchesData(PATCHES)
    train_idx, val_idx, test_idx = make_split(data.lat, data.lon, seed=seed)
    print(f"\nData Loaded: Total Patches={data.n} | Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    x_all = data.agg(torch.arange(data.n)).numpy()

    # 2. Clustering Pipeline Setup & TF-IDF Extraction
    pipeline = ClusteringPipeline(seed=seed)
    x_tfidf, idf_weights = pipeline.fit_transform_tfidf(x_all)

    # K Selection Curves (K=2..15)
    print("\n[Step 1] Evaluating K Selection Curves (Silhouette, Elbow, DBI, CHI)...")
    df_k_metrics = pipeline.evaluate_k_selection(x_tfidf, k_range=range(2, 16))
    Visualizer.plot_k_selection_curves(df_k_metrics, out_path="k_selection_metrics_curves.png")

    # Fit Clustering Algorithms (with optional Optuna Tuning)
    print("\n[Step 2] Fitting Clustering Algorithms (KMeans, GMM, DBSCAN)...")
    labels_km, _ = pipeline.fit_clustering(x_tfidf, method="kmeans", tune=False, params={"n_clusters": n_clusters, "random_state": seed, "n_init": 10})
    labels_gmm, _ = pipeline.fit_clustering(x_tfidf, method="gmm", tune=False, params={"n_components": n_clusters, "random_state": seed, "covariance_type": "full"})

    if optuna_tune:
        print("[Optuna] Tuning DBSCAN hyper-parameters...")
        labels_db, _ = pipeline.fit_clustering(x_tfidf, method="dbscan", tune=True, n_trials=optuna_trials)
    else:
        labels_db, _ = pipeline.fit_clustering(x_tfidf, method="dbscan", tune=False, params={"eps": 2.0, "min_samples": 5, "metric": "euclidean"})

    labels_dict = {
        "kmeans": labels_km,
        "gmm": labels_gmm,
        "dbscan": labels_db
    }

    # 3. Model Loading, Latent Encoding, Reconstruction & Shift Experiments
    print("\n[Step 3] Loading Target Autoencoder Models & Evaluating Metrics...")
    recon_rows = []
    cohesion_rows = []
    models_dict = {}
    z_dict = {}

    for cfg in MODELS_CONFIG:
        model_name = cfg["name"]
        print(f" -> Processing Model: {model_name}")
        model = load_ae_model(cfg["dir"], seed=seed)
        models_dict[model_name] = model

        # Reconstruction Metrics (Table 1)
        recon_res = evaluate_reconstruction(model, data, test_idx, batch_size=BATCH)

        # Encode Latent Space Z
        with torch.no_grad():
            res = model.encode(torch.from_numpy(x_all))
            z_all = res[0].numpy() if isinstance(res, tuple) else res.numpy()
        z_dict[model_name] = z_all

        # Latent Cohesion Metrics
        km_cohesion = evaluate_latent_cohesion(z_all, labels_km)
        gmm_cohesion = evaluate_latent_cohesion(z_all, labels_gmm)
        db_cohesion = evaluate_latent_cohesion(z_all, labels_db)

        recon_rows.append({
            "Model": model_name,
            **recon_res,
            "TFIDF NMI": km_cohesion["Silhouette Score"],
        })

        cohesion_rows.append({
            "Model": model_name,
            "KMeans Latent Silhouette": km_cohesion["Silhouette Score"],
            "KMeans Latent DBI": km_cohesion["Davies-Bouldin Index"],
            "GMM Latent Silhouette": gmm_cohesion["Silhouette Score"],
            "GMM Latent DBI": gmm_cohesion["Davies-Bouldin Index"],
            "DBSCAN Latent Silhouette": db_cohesion["Silhouette Score"],
            "DBSCAN Latent DBI": db_cohesion["Davies-Bouldin Index"],
        })

        # Latent Shift Experiment (Table 2)
        df_shift, z_base_test, z_std_dims = compute_standardized_latent_shift(model, data, test_idx, amounts=[1, 3, 5])
        df_shift.to_csv(f"evaluation_table2_shift_{model_name}.csv")

        if model_name == "v3_ddae_tfidf":
            Visualizer.plot_latent_shift_barchart(df_shift, model_name, f"latent_shift_{model_name}.png")

    # 4. Save Quantitative Summary CSV Tables
    df_recon = pd.DataFrame(recon_rows)
    df_recon.to_csv("evaluation_table1_reconstruction.csv", index=False)
    print("\n--- Table 1: Reconstruction Metrics ---")
    print(df_recon.to_string(index=False))

    df_cohesion = pd.DataFrame(cohesion_rows)
    df_cohesion.to_csv("latent_cohesion_comparison_table.csv", index=False)
    print("\n--- Latent Space Cohesion Summary Table ---")
    print(df_cohesion.to_string(index=False))

    # 5. Generate 7-Model 3-Algorithm Comparison Grid Figure
    print("\n[Step 4] Generating Multi-Model 3-Algorithm Comparison Figure...")
    Visualizer.plot_multi_model_clustering_comparison(
        models_dict, z_dict, labels_dict, out_path="all_models_kmeans_gmm_dbscan_comparison.png"
    )

    print("\n" + "=" * 80)
    print(" Master Pipeline Completed Successfully!")
    print(" Results saved to evaluation_table1_reconstruction.csv, latent_cohesion_comparison_table.csv,")
    print(" and all_models_kmeans_gmm_dbscan_comparison.png")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOBE-DOBE Target Model Evaluation & Multi-Clustering Master Pipeline")
    parser.add_argument("--retrain", action="store_true", help="Re-train models with custom hyperparameters")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs if retraining")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate if retraining")
    parser.add_argument("--n_clusters", type=int, default=8, help="Number of clusters for TF-IDF / graph loss")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--no_tune", action="store_true", help="Disable Optuna hyperparameter tuning")
    parser.add_argument("--optuna_trials", type=int, default=30, help="Number of Optuna trials")

    args = parser.parse_args()

    main(
        retrain=args.retrain,
        epochs=args.epochs,
        lr=args.lr,
        n_clusters=args.n_clusters,
        seed=args.seed,
        optuna_tune=not args.no_tune,
        optuna_trials=args.optuna_trials
    )
