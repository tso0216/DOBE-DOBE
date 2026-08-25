"""
Visualization Module for DOBE-DOBE Clustering & Latent Space Analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 130


class Visualizer:
    @staticmethod
    def plot_k_selection_curves(df_k_metrics, out_path="k_selection_metrics_curves.png"):
        """
        Plots 4-panel curves for K Selection: Silhouette, Inertia (Elbow), DBI, CHI.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Silhouette Score
        axes[0, 0].plot(df_k_metrics["K"], df_k_metrics["Silhouette Score"], 'o-', color='crimson', linewidth=2, markersize=7)
        axes[0, 0].set_title("Silhouette Score vs. K (Higher is better)")
        axes[0, 0].set_xlabel("K (Number of Clusters)")
        axes[0, 0].set_ylabel("Silhouette Score")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Inertia (Elbow Curve)
        axes[0, 1].plot(df_k_metrics["K"], df_k_metrics["Inertia (SSE)"], 's-', color='darkblue', linewidth=2, markersize=7)
        axes[0, 1].set_title("Elbow Method: Inertia (SSE) vs. K")
        axes[0, 1].set_xlabel("K (Number of Clusters)")
        axes[0, 1].set_ylabel("Inertia / SSE")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Davies-Bouldin Index
        axes[1, 0].plot(df_k_metrics["K"], df_k_metrics["Davies-Bouldin Index"], '^-', color='green', linewidth=2, markersize=7)
        axes[1, 0].set_title("Davies-Bouldin Index vs. K (Lower is better)")
        axes[1, 0].set_xlabel("K (Number of Clusters)")
        axes[1, 0].set_ylabel("DB Index")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Calinski-Harabasz Index
        axes[1, 1].plot(df_k_metrics["K"], df_k_metrics["Calinski-Harabasz Index"], 'd-', color='purple', linewidth=2, markersize=7)
        axes[1, 1].set_title("Calinski-Harabasz Index vs. K (Higher is better)")
        axes[1, 1].set_xlabel("K (Number of Clusters)")
        axes[1, 1].set_ylabel("CH Index")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    @staticmethod
    def plot_gmm_bic_aic(df_gmm, out_path="gmm_bic_aic_curves.png"):
        """
        Plots GMM BIC and AIC Curves across K.
        """
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:blue'
        ax1.set_xlabel('K (Number of Gaussian Components)')
        ax1.set_ylabel('BIC (Bayesian Information Criterion)', color=color)
        ax1.plot(df_gmm["K"], df_gmm["GMM BIC"], 'o-', color=color, label='BIC')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('AIC (Akaike Information Criterion)', color=color)
        ax2.plot(df_gmm["K"], df_gmm["GMM AIC"], 's--', color=color, label='AIC')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('GMM Model Selection: BIC and AIC Curves')
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    @staticmethod
    def plot_multi_model_clustering_comparison(models_dict, z_dict, labels_dict, out_path="all_models_kmeans_gmm_dbscan_comparison.png"):
        """
        Plots 3-column multi-model Latent space comparison grid:
        Col 1: KMeans (K=8), Col 2: GMM (K=8), Col 3: DBSCAN (best params)
        """
        model_names = list(models_dict.keys())
        n_models = len(model_names)
        fig, axes = plt.subplots(n_models, 3, figsize=(18, 4 * n_models))

        labels_km = labels_dict["kmeans"]
        labels_gmm = labels_dict["gmm"]
        labels_db = labels_dict["dbscan"]

        for i, name in enumerate(model_names):
            z_all = z_dict[name]

            ax1 = axes[i, 0] if n_models > 1 else axes[0]
            ax2 = axes[i, 1] if n_models > 1 else axes[1]
            ax3 = axes[i, 2] if n_models > 1 else axes[2]

            # Col 1: KMeans
            sc1 = ax1.scatter(z_all[:, 0], z_all[:, 1], c=labels_km, cmap="tab10", s=12, alpha=0.75)
            ax1.set_title(f"[{name}] Latent Space: KMeans (K=8)", fontsize=11, fontweight='bold')
            ax1.set_xlabel("Latent Dim 1", fontsize=9)
            ax1.set_ylabel("Latent Dim 2", fontsize=9)
            plt.colorbar(sc1, ax=ax1, label="KMeans Cluster ID")

            # Col 2: GMM Hard Predictions
            sc2 = ax2.scatter(z_all[:, 0], z_all[:, 1], c=labels_gmm, cmap="tab10", s=12, alpha=0.75)
            ax2.set_title(f"[{name}] Latent Space: GMM (K=8)", fontsize=11, fontweight='bold')
            ax2.set_xlabel("Latent Dim 1", fontsize=9)
            ax2.set_ylabel("Latent Dim 2", fontsize=9)
            plt.colorbar(sc2, ax=ax2, label="GMM Cluster ID")

            # Col 3: DBSCAN
            unique_labels = sorted(set(labels_db))
            for lbl_idx, l in enumerate(unique_labels):
                m = labels_db == l
                if l == -1:
                    ax3.scatter(z_all[m, 0], z_all[m, 1], color="lightgray", s=10, alpha=0.4, label="Noise (-1)")
                else:
                    ax3.scatter(z_all[m, 0], z_all[m, 1], s=18, alpha=0.8, label=f"Cluster {l}")

            ax3.set_title(f"[{name}] Latent Space: Best DBSCAN", fontsize=11, fontweight='bold')
            ax3.set_xlabel("Latent Dim 1", fontsize=9)
            ax3.set_ylabel("Latent Dim 2", fontsize=9)
            ax3.legend(loc="upper right", fontsize=7)

        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    @staticmethod
    def plot_latent_shift_barchart(df_shift, model_name, out_path):
        """
        Plots Table 2 Latent Shift Bar Chart (+1, +3, +5) across POI categories.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        df_shift.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
        ax.set_title(f"Latent Shift Perturbation Analysis ({model_name})", fontsize=12)
        ax.set_ylabel("Normalized Latent Shift Distance Δz", fontsize=10)
        ax.set_xlabel("POI Category", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
