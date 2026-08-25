"""
Model Training & Re-training Module for DOBE-DOBE Autoencoders
"""

import os
import sys
import subprocess


class ModelTrainer:
    """
    Handles training and re-training of DOBE-DOBE Autoencoder models
    with custom hyperparameters (epochs, lr, seed, n_clusters, etc.).
    """
    def __init__(self, root_dir=None):
        if root_dir is None:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.root_dir = root_dir

    def train_model(self, model_name, epochs=100, lr=1e-3, n_clusters=8, seed=0, extra_env=None):
        """
        Triggers training of a specific model directory (e.g., 'v3_ddae_tfidf').

        Parameters:
        - model_name: str, name of the model directory inside model/
        - epochs: int, number of training epochs
        - lr: float, learning rate
        - n_clusters: int, number of clusters for graph/TF-IDF contrastive loss
        - seed: int, random seed
        - extra_env: dict, additional environment variables
        """
        model_dir = os.path.join(self.root_dir, "model", model_name)
        train_script = os.path.join(model_dir, "train.py")

        if not os.path.exists(train_script):
            raise FileNotFoundError(f"Training script not found at: {train_script}")

        env = os.environ.copy()
        env["EPOCHS"] = str(epochs)
        env["LR"] = str(lr)
        env["N_CLUSTERS"] = str(n_clusters)
        env["SEED"] = str(seed)

        if extra_env:
            for k, v in extra_env.items():
                env[k] = str(v)

        print(f"\n[ModelTrainer] Starting Re-training for {model_name} (Epochs={epochs}, LR={lr}, K={n_clusters}, Seed={seed})...")
        cmd = [sys.executable, train_script]

        result = subprocess.run(cmd, cwd=model_dir, env=env, text=True, capture_output=True)

        if result.returncode != 0:
            print(f"[ModelTrainer] Error during training of {model_name}:\n{result.stderr}")
            raise RuntimeError(f"Model training failed for {model_name} with exit code {result.returncode}")

        print(f"[ModelTrainer] Successfully finished training {model_name}!")
        return result.stdout

    def train_all(self, models=None, epochs=100, lr=1e-3, n_clusters=8, seed=0):
        """
        Re-trains multiple or all models sequentially.
        """
        if models is None:
            models = [
                "v2_deep_ae",
                "v2_ddae_base",
                "v3_ddae_pcgrad",
                "v3_ddae_twostage",
                "v3_ddae_dec",
                "v3_ddae_cluster",
                "v3_ddae_tfidf",
            ]

        for m in models:
            self.train_model(m, epochs=epochs, lr=lr, n_clusters=n_clusters, seed=seed)
