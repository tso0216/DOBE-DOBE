"""
Clustering Pipeline Core Module
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from .tuner import OptunaTuner
from .metrics import evaluate_latent_cohesion
from .visualizer import Visualizer


class ClusteringPipeline:
    """
    Unified Clustering Pipeline supporting multiple algorithms:
    - kmeans, gmm, dbscan, agglomerative
    - Optional Optuna hyperparameter tuning (tune=True)
    - Quantitative metrics & visualization generation
    """
    def __init__(self, seed=0):
        self.seed = seed
        self.tuner = OptunaTuner(seed=seed)
        self.fitted_models = {}
        self.labels = {}
        self.best_params = {}

    def fit_transform_tfidf(self, X_raw):
        """
        Computes L2-normalized TF-IDF features from raw POI count matrix.
        """
        tfidf_trans = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
        X_tfidf = tfidf_trans.fit_transform(X_raw).toarray()
        return X_tfidf, tfidf_trans.idf_

    def fit_clustering(self, X_tfidf, method="kmeans", tune=False, n_trials=30, params=None):
        """
        Runs clustering method with or without Optuna hyperparameter tuning.

        Parameters:
        - X_tfidf: TF-IDF feature matrix.
        - method: 'kmeans', 'gmm', 'dbscan', 'agglomerative'
        - tune: bool, whether to use Optuna for tuning.
        - n_trials: int, number of Optuna trials if tune=True.
        - params: dict, fixed hyperparameters if tune=False.
        """
        X_std = StandardScaler().fit_transform(X_tfidf)

        if tune:
            print(f"[ClusteringPipeline] Running Optuna Hyperparameter Tuning for {method} ({n_trials} trials)...")
            if method == "dbscan":
                tuned_params, best_score = self.tuner.tune_dbscan(X_tfidf, n_trials=n_trials)
                params = tuned_params
            elif method == "gmm":
                tuned_params, best_score = self.tuner.tune_gmm(X_tfidf, n_trials=n_trials)
                params = tuned_params
            elif method == "kmeans":
                tuned_params, best_score = self.tuner.tune_kmeans(X_tfidf, n_trials=n_trials)
                params = tuned_params
            else:
                raise ValueError(f"Tuning not supported for method: {method}")

            print(f"[ClusteringPipeline] Best {method} params found: {params} (Score: {best_score:.4f})")
            self.best_params[method] = params

        # Default fallback parameters if not provided
        if params is None:
            if method == "kmeans":
                params = {"n_clusters": 8, "random_state": self.seed, "n_init": 10}
            elif method == "gmm":
                params = {"n_components": 8, "random_state": self.seed, "covariance_type": "full"}
            elif method == "dbscan":
                params = {"eps": 2.0, "min_samples": 5, "metric": "euclidean"}
            elif method == "agglomerative":
                params = {"n_clusters": 8, "linkage": "ward"}

        # Instantiate & Fit
        if method == "kmeans":
            model = KMeans(**params)
            labels = model.fit_predict(X_tfidf)
        elif method == "gmm":
            model = GaussianMixture(**params)
            labels = model.fit_predict(X_tfidf)
        elif method == "dbscan":
            metric = params.get("metric", "euclidean")
            data = X_std if metric == "euclidean" else X_tfidf
            model = DBSCAN(eps=params.get("eps", 2.0), min_samples=params.get("min_samples", 5), metric=metric)
            labels = model.fit_predict(data)
        elif method == "agglomerative":
            model = AgglomerativeClustering(**params)
            labels = model.fit_predict(X_tfidf)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        self.fitted_models[method] = model
        self.labels[method] = labels
        return labels, model

    def evaluate_k_selection(self, X_tfidf, k_range=range(2, 16)):
        """
        Evaluates KMeans mathematical metrics across K=2..15 for K selection & Elbow curves.
        """
        metrics_list = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
            lbls = km.fit_predict(X_tfidf)

            sil = silhouette_score(X_tfidf, lbls)
            inertia = km.inertia_
            dbi = davies_bouldin_score(X_tfidf, lbls)
            chi = calinski_harabasz_score(X_tfidf, lbls)

            metrics_list.append({
                "K": k,
                "Silhouette Score": round(sil, 4),
                "Inertia (SSE)": round(inertia, 2),
                "Davies-Bouldin Index": round(dbi, 4),
                "Calinski-Harabasz Index": round(chi, 2)
            })

        return pd.DataFrame(metrics_list)
