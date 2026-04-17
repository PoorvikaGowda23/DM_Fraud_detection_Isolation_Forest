"""
Phase 2 — Feature-Weighted Isolation Forest (FW-iForest)
Innovation 1: Weighted feature selection using variance, kurtosis, mutual information.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.tree import ExtraTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from scipy.stats import kurtosis
from sklearn.feature_selection import mutual_info_classif
import joblib
import os


# ──────────────────────────────────────────────
# 1. Feature Importance Computation
# ──────────────────────────────────────────────

def compute_feature_weights(
    X: np.ndarray,
    y_pseudo: np.ndarray = None
) -> np.ndarray:
    """
    Compute per-feature importance scores using:
      1. Variance       — measures spread
      2. Kurtosis       — sensitive to outliers/anomalies
      3. Mutual Info    — relevance to pseudo-anomaly labels

    Returns normalized sampling probabilities for each feature.
    """
    n_features = X.shape[1]

    # --- Variance score ---
    var_scores = np.var(X, axis=0)
    var_scores = var_scores / (var_scores.sum() + 1e-10)

    # --- Kurtosis score (absolute, higher = more outlier-sensitive) ---
    kurt_scores = np.abs(kurtosis(X, axis=0))
    kurt_scores = kurt_scores / (kurt_scores.sum() + 1e-10)

    # --- Mutual Information (requires pseudo-labels) ---
    if y_pseudo is not None:
        mi_scores = mutual_info_classif(X, y_pseudo, random_state=42)
    else:
        # Fallback: use isolation-score-derived pseudo labels
        # (top 1% by variance rank treated as anomaly)
        pseudo = (np.sum(X ** 2, axis=1) > np.percentile(
            np.sum(X ** 2, axis=1), 99
        )).astype(int)
        mi_scores = mutual_info_classif(X, pseudo, random_state=42)

    mi_scores = mi_scores / (mi_scores.sum() + 1e-10)

    # --- Combine: equal weight to each criterion ---
    combined = (var_scores + kurt_scores + mi_scores) / 3.0

    # Normalize to probability distribution
    feature_weights = combined / combined.sum()

    return feature_weights


def feature_importance_table(
    feature_cols: list,
    weights: np.ndarray
) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances."""
    df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": weights
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# 2. Weighted Isolation Tree Node
# ──────────────────────────────────────────────

class WeightedITree:
    """
    Single Isolation Tree that selects features
    according to a probability distribution instead of uniformly.
    """

    def __init__(self, max_depth: int, feature_weights: np.ndarray):
        self.max_depth       = max_depth
        self.feature_weights = feature_weights
        self.tree_           = None

    def fit(self, X: np.ndarray, depth: int = 0):
        n, d = X.shape

        # Stopping conditions
        if depth >= self.max_depth or n <= 1 or np.all(X == X[0]):
            return {"type": "leaf", "size": n}

        # Weighted feature selection
        feat_idx = np.random.choice(d, p=self.feature_weights)
        col      = X[:, feat_idx]
        col_min, col_max = col.min(), col.max()

        if col_min == col_max:
            return {"type": "leaf", "size": n}

        split_val = np.random.uniform(col_min, col_max)

        left_mask  = col < split_val
        right_mask = ~left_mask

        return {
            "type":       "internal",
            "feature":    feat_idx,
            "split":      split_val,
            "left":       self.fit(X[left_mask],  depth + 1),
            "right":      self.fit(X[right_mask], depth + 1),
        }

    def path_length(self, x: np.ndarray, node: dict, depth: int = 0) -> float:
        if node["type"] == "leaf":
            return depth + _c(node["size"])
        if x[node["feature"]] < node["split"]:
            return self.path_length(x, node["left"],  depth + 1)
        else:
            return self.path_length(x, node["right"], depth + 1)


def _c(n: int) -> float:
    """Average path length of unsuccessful BST search (normalisation term)."""
    if n <= 1:
        return 0.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


# ──────────────────────────────────────────────
# 3. FW-iForest Ensemble
# ──────────────────────────────────────────────

class FWIsolationForest(BaseEstimator, OutlierMixin):
    """
    Feature-Weighted Isolation Forest.
    Weighted feature selection replaces uniform random selection.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples:  int = 256,
        contamination: float = 0.001,
        random_state: int = 42
    ):
        self.n_estimators  = n_estimators
        self.max_samples   = max_samples
        self.contamination = contamination
        self.random_state  = random_state
        self.trees_        = []
        self.feature_weights_ = None
        self.threshold_    = None

    def fit(self, X: np.ndarray, y_pseudo: np.ndarray = None):
        np.random.seed(self.random_state)
        n, d = X.shape

        # Compute feature weights
        self.feature_weights_ = compute_feature_weights(X, y_pseudo)

        max_depth = int(np.ceil(np.log2(self.max_samples)))
        self.trees_ = []

        for _ in range(self.n_estimators):
            # Sub-sample
            idx     = np.random.choice(n, size=min(self.max_samples, n), replace=False)
            X_sub   = X[idx]
            itree   = WeightedITree(max_depth, self.feature_weights_)
            node    = itree.fit(X_sub)
            itree.tree_ = node
            self.trees_.append(itree)

        # Calibrate threshold on training data
        scores = self._raw_scores(X)
        self.threshold_ = np.percentile(scores, (1 - self.contamination) * 100)
        return self

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        all_lengths = np.zeros(n)
        for itree in self.trees_:
            for i, x in enumerate(X):
                all_lengths[i] += itree.path_length(x, itree.tree_)
        avg_lengths = all_lengths / self.n_estimators
        cn = _c(self.max_samples)
        scores = 2 ** (-avg_lengths / (cn + 1e-10))
        return scores

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Returns [0,1] anomaly scores. Higher = more anomalous."""
        return self._raw_scores(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 = anomaly, 0 = normal."""
        scores = self.anomaly_scores(X)
        return (scores >= self.threshold_).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Negative anomaly score (sklearn convention)."""
        return -self.anomaly_scores(X)


# ──────────────────────────────────────────────
# 4. Evaluation
# ──────────────────────────────────────────────

def evaluate_phase2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray
) -> dict:
    auc_roc  = roc_auc_score(y_true, anomaly_scores)
    avg_prec = average_precision_score(y_true, anomaly_scores)
    cm       = confusion_matrix(y_true, y_pred)
    report   = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        "phase": "Phase 2 — Feature-Weighted Isolation Forest",
        "auc_roc":       round(auc_roc, 4),
        "avg_precision": round(avg_prec, 4),
        "precision":     round(report["1"]["precision"], 4),
        "recall":        round(report["1"]["recall"],    4),
        "f1_score":      round(report["1"]["f1-score"],  4),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n[Phase 2] AUC-ROC:    {metrics['auc_roc']}")
    print(f"[Phase 2] Avg Prec:   {metrics['avg_precision']}")
    print(f"[Phase 2] Precision:  {metrics['precision']}")
    print(f"[Phase 2] Recall:     {metrics['recall']}")
    print(f"[Phase 2] F1 Score:   {metrics['f1_score']}")

    return metrics


# ──────────────────────────────────────────────
# 5. Save / Load
# ──────────────────────────────────────────────

def save_fw_model(model, path: str = None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "phase2")
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, f"{path}/fw_iforest.pkl")
    print(f"[Phase 2] Model saved to {path}/")


def load_fw_model(path: str = "models/phase2"):
    return joblib.load(f"{path}/fw_iforest.pkl")


# ──────────────────────────────────────────────
# 6. Run Phase 2 pipeline
# ──────────────────────────────────────────────

def run_phase2(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    feature_cols: list
) -> dict:
    """Full Phase 2 pipeline. Expects preprocessed data from Phase 1."""
    print("[Phase 2] Computing feature weights...")
    fw_model = FWIsolationForest(
        n_estimators=100,
        max_samples=256,
        contamination=0.001,
        random_state=42
    )
    fw_model.fit(X_train)

    importance_df = feature_importance_table(feature_cols, fw_model.feature_weights_)
    print("[Phase 2] Top 5 features by importance:")
    print(importance_df.head())

    anomaly_scores = fw_model.anomaly_scores(X_test)
    y_pred         = fw_model.predict(X_test)
    metrics        = evaluate_phase2(y_test, y_pred, anomaly_scores)

    save_fw_model(fw_model)

    return {
        "metrics":        metrics,
        "model":          fw_model,
        "anomaly_scores": anomaly_scores,
        "y_pred":         y_pred,
        "importance_df":  importance_df,
        "feature_weights": fw_model.feature_weights_
    }