"""
Optuna Hyperparameter Tuning Module
"""

import numpy as np
import optuna
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Disable noisy Optuna logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTuner:
    def __init__(self, seed=0):
        self.seed = seed

    def tune_dbscan(self, X, n_trials=50, max_noise_pct=25.0):
        """
        Optuna hyperparameter tuning for DBSCAN.
        Optimizes Silhouette Score on non-noise points while controlling noise percentage.
        """
        # Standardize for Euclidean distance
        X_std = StandardScaler().fit_transform(X)

        def objective(trial):
            metric = trial.suggest_categorical("metric", ["euclidean", "cosine"])
            if metric == "euclidean":
                eps = trial.suggest_float("eps", 0.3, 3.0, step=0.1)
                data = X_std
            else:
                eps = trial.suggest_float("eps", 0.01, 0.5, step=0.01)
                data = X

            min_samples = trial.suggest_int("min_samples", 3, 15)

            db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            lbls = db.fit_predict(data)

            n_clusters = len(set(lbls) - {-1})
            n_noise = (lbls == -1).sum()
            noise_pct = n_noise / len(X) * 100

            if n_clusters < 2 or noise_pct > max_noise_pct:
                return -1.0

            valid_mask = lbls != -1
            if valid_mask.sum() <= n_clusters:
                return -1.0

            score = silhouette_score(data[valid_mask], lbls[valid_mask])
            return score

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_score = study.best_value
        return best_params, best_score

    def tune_gmm(self, X, n_trials=30):
        """
        Optuna hyperparameter tuning for GMM.
        Optimizes BIC (minimization) or Silhouette score (maximization).
        """
        def objective(trial):
            n_components = trial.suggest_int("n_components", 2, 15)
            covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])

            gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=self.seed)
            gmm.fit(X)
            lbls = gmm.predict(X)

            if len(set(lbls)) < 2:
                return -1.0

            score = silhouette_score(X, lbls)
            return score

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=n_trials)

        return study.best_params, study.best_value

    def tune_kmeans(self, X, n_trials=20):
        """
        Optuna hyperparameter tuning for KMeans.
        """
        def objective(trial):
            n_clusters = trial.suggest_int("n_clusters", 2, 15)
            init = trial.suggest_categorical("init", ["k-means++", "random"])

            km = KMeans(n_clusters=n_clusters, init=init, random_state=self.seed, n_init=10)
            lbls = km.fit_predict(X)

            score = silhouette_score(X, lbls)
            return score

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=n_trials)

        return study.best_params, study.best_value
