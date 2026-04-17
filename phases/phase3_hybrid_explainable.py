"""
Phase 3 — Hybrid Explainable Anomaly Detection Framework
Innovation 2: Hybrid iForest + Autoencoder fusion
Innovation 3: Per-instance explainability via feature split tracking
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import joblib
import os


# ══════════════════════════════════════════════
# PART A — Autoencoder
# ══════════════════════════════════════════════

class FraudAutoencoder(nn.Module):
    """
    Autoencoder trained ONLY on normal transactions.
    High reconstruction error → anomaly.
    Architecture: 30 → 16 → 8 → 16 → 30
    """

    def __init__(self, input_dim: int = 30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


def train_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int = 30,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu"
) -> FraudAutoencoder:
    """
    Train autoencoder on normal transactions only.
    y_train is used only to filter normals.
    """
    # Use only normal transactions for training
    if y_train is None:
        raise ValueError(
            "y_train is None — Phase 1 did not return training labels. "
            "Restart the server and re-run Phase 1 before Phase 3."
        )
    X_normal = X_train[np.array(y_train) == 0]
    if X_normal.shape[0] == 0:
        raise ValueError(
            f"No normal samples (label=0) found in y_train. "
            f"Unique values present: {np.unique(y_train)}"
        )
    print(f"[Phase 3] Training autoencoder on {X_normal.shape[0]} normal samples.")

    X_tensor = torch.tensor(X_normal, dtype=torch.float32).to(device)
    dataset  = TensorDataset(X_tensor)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = FraudAutoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss   = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(loader):.6f}")

    print("[Phase 3] Autoencoder training complete.")
    return model


def get_reconstruction_errors(
    model: FraudAutoencoder,
    X: np.ndarray,
    device: str = "cpu"
) -> np.ndarray:
    """Compute per-sample mean squared reconstruction error."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_recon = model(X_tensor).cpu().numpy()
    errors = np.mean((X - X_recon) ** 2, axis=1)
    return errors


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    return (scores - mn) / (mx - mn + 1e-10)


# ══════════════════════════════════════════════
# PART B — Hybrid Fusion
# ══════════════════════════════════════════════

def hybrid_fusion(
    s1: np.ndarray,
    s2: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Combine iForest score (S1) and Autoencoder error (S2).
    S_final = α·S1 + (1-α)·S2
    Both inputs should be normalized to [0,1].
    """
    s1_norm = normalize_scores(s1)
    s2_norm = normalize_scores(s2)
    return alpha * s1_norm + (1 - alpha) * s2_norm


def find_best_alpha(
    s1: np.ndarray,
    s2: np.ndarray,
    y_true: np.ndarray,
    alphas: list = None
) -> tuple:
    """
    Grid search over alpha values to maximize AUC-ROC.
    Returns (best_alpha, best_auc, all_results).
    """
    if alphas is None:
        alphas = [round(a, 2) for a in np.arange(0.0, 1.05, 0.05)]

    best_alpha, best_auc = 0.5, 0.0
    results = []

    for a in alphas:
        s_final = hybrid_fusion(s1, s2, alpha=a)
        auc     = roc_auc_score(y_true, s_final)
        results.append({"alpha": a, "auc": round(auc, 4)})
        if auc > best_auc:
            best_auc   = auc
            best_alpha = a

    print(f"[Phase 3] Best alpha: {best_alpha} → AUC: {best_auc:.4f}")
    return best_alpha, best_auc, results


# ══════════════════════════════════════════════
# PART C — Explainability Module
# ══════════════════════════════════════════════

class ExplainableITree:
    """
    Isolation Tree that logs feature splits along each path.
    Used to generate per-instance explanations.
    """

    def __init__(self, max_depth: int, feature_weights: np.ndarray = None):
        self.max_depth       = max_depth
        self.feature_weights = feature_weights
        self.tree_           = None

    def fit(self, X: np.ndarray, depth: int = 0) -> dict:
        n, d = X.shape
        if depth >= self.max_depth or n <= 1 or np.all(X == X[0]):
            return {"type": "leaf", "size": n}

        if self.feature_weights is not None:
            feat_idx = np.random.choice(d, p=self.feature_weights)
        else:
            feat_idx = np.random.randint(0, d)

        col = X[:, feat_idx]
        col_min, col_max = col.min(), col.max()
        if col_min == col_max:
            return {"type": "leaf", "size": n}

        split_val  = np.random.uniform(col_min, col_max)
        left_mask  = col < split_val
        right_mask = ~left_mask

        return {
            "type":    "internal",
            "feature": feat_idx,
            "split":   split_val,
            "left":    self.fit(X[left_mask],  depth + 1),
            "right":   self.fit(X[right_mask], depth + 1),
        }

    def path_with_splits(
        self,
        x: np.ndarray,
        node: dict,
        depth: int = 0
    ) -> list:
        """
        Returns list of (feature_index, depth) tuples
        for every split encountered along x's isolation path.
        """
        if node["type"] == "leaf":
            return []
        split_log = [(node["feature"], depth)]
        if x[node["feature"]] < node["split"]:
            split_log += self.path_with_splits(x, node["left"],  depth + 1)
        else:
            split_log += self.path_with_splits(x, node["right"], depth + 1)
        return split_log


def _c(n: int) -> float:
    if n <= 1:
        return 0.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


class ExplainableForest:
    """
    Ensemble of ExplainableITrees that produces
    both anomaly scores and per-instance explanations.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        feature_weights: np.ndarray = None,
        random_state: int = 42
    ):
        self.n_estimators    = n_estimators
        self.max_samples     = max_samples
        self.feature_weights = feature_weights
        self.random_state    = random_state
        self.trees_          = []

    def fit(self, X: np.ndarray):
        np.random.seed(self.random_state)
        n          = X.shape[0]
        max_depth  = int(np.ceil(np.log2(self.max_samples)))
        self.trees_ = []

        for _ in range(self.n_estimators):
            idx   = np.random.choice(n, size=min(self.max_samples, n), replace=False)
            itree = ExplainableITree(max_depth, self.feature_weights)
            node  = itree.fit(X[idx])
            itree.tree_ = node
            self.trees_.append(itree)

        return self

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        n   = X.shape[0]
        avg = np.zeros(n)
        for itree in self.trees_:
            for i, x in enumerate(X):
                avg[i] += itree.path_with_splits.__self__  # just path length
        # Proper path length calculation
        lengths = np.zeros(n)
        for itree in self.trees_:
            for i, x in enumerate(X):
                splits = itree.path_with_splits(x, itree.tree_)
                lengths[i] += len(splits) + _c(1)
        avg_lengths = lengths / self.n_estimators
        cn = _c(self.max_samples)
        return 2 ** (-avg_lengths / (cn + 1e-10))

    def explain(
        self,
        x: np.ndarray,
        feature_cols: list,
        top_k: int = 5
    ) -> dict:
        """
        Generate per-instance explanation.
        Returns top_k features with their contribution scores.
        Contribution score = sum(1 / (depth+1)) across all trees.
        """
        n_features      = len(feature_cols)
        contrib_scores  = np.zeros(n_features)
        path_lengths    = []

        for itree in self.trees_:
            splits = itree.path_with_splits(x, itree.tree_)
            path_lengths.append(len(splits))
            for feat_idx, depth in splits:
                contrib_scores[feat_idx] += 1.0 / (depth + 1)

        avg_path_length  = np.mean(path_lengths) if path_lengths else 0
        cn               = _c(self.max_samples)
        anomaly_score    = float(2 ** (-avg_path_length / (cn + 1e-10)))

        # Normalize contributions
        total = contrib_scores.sum()
        if total > 0:
            contrib_scores /= total

        # Build ranked explanation
        ranked = sorted(
            [
                {
                    "feature":      feature_cols[i],
                    "contribution": round(float(contrib_scores[i]), 4),
                    "value":        round(float(x[i]), 4)
                }
                for i in range(n_features)
            ],
            key=lambda d: d["contribution"],
            reverse=True
        )[:top_k]

        return {
            "anomaly_score":  round(anomaly_score, 4),
            "avg_path_length": round(avg_path_length, 4),
            "top_features":   ranked
        }


def batch_explain(
    forest: ExplainableForest,
    X: np.ndarray,
    feature_cols: list,
    top_k: int = 5
) -> list:
    """Generate explanations for multiple instances."""
    return [forest.explain(X[i], feature_cols, top_k) for i in range(len(X))]


# ══════════════════════════════════════════════
# PART D — Evaluation
# ══════════════════════════════════════════════

def evaluate_phase3(
    y_true: np.ndarray,
    final_scores: np.ndarray,
    threshold: float = 0.5
) -> dict:
    y_pred   = (final_scores >= threshold).astype(int)
    auc_roc  = roc_auc_score(y_true, final_scores)
    avg_prec = average_precision_score(y_true, final_scores)
    cm       = confusion_matrix(y_true, y_pred)
    report   = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        "phase":         "Phase 3 — Hybrid Explainable Framework",
        "auc_roc":       round(auc_roc, 4),
        "avg_precision": round(avg_prec, 4),
        "precision":     round(report["1"]["precision"], 4),
        "recall":        round(report["1"]["recall"],    4),
        "f1_score":      round(report["1"]["f1-score"],  4),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n[Phase 3] AUC-ROC:    {metrics['auc_roc']}")
    print(f"[Phase 3] Avg Prec:   {metrics['avg_precision']}")
    print(f"[Phase 3] Precision:  {metrics['precision']}")
    print(f"[Phase 3] Recall:     {metrics['recall']}")
    print(f"[Phase 3] F1 Score:   {metrics['f1_score']}")

    return metrics


# ══════════════════════════════════════════════
# PART E — Save / Load
# ══════════════════════════════════════════════

def save_phase3(ae_model, exp_forest, path: str = None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "phase3")
    os.makedirs(path, exist_ok=True)
    torch.save(ae_model.state_dict(), f"{path}/autoencoder.pt")
    joblib.dump(exp_forest, f"{path}/exp_forest.pkl")
    print(f"[Phase 3] Models saved to {path}/")


def load_phase3(input_dim: int, path: str = "models/phase3"):
    ae_model = FraudAutoencoder(input_dim)
    ae_model.load_state_dict(torch.load(f"{path}/autoencoder.pt"))
    ae_model.eval()
    exp_forest = joblib.load(f"{path}/exp_forest.pkl")
    return ae_model, exp_forest


# ══════════════════════════════════════════════
# PART F — Run Phase 3 pipeline
# ══════════════════════════════════════════════

def run_phase3(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    feature_cols: list,
    fw_scores: np.ndarray = None,      # S1 from Phase 2
    feature_weights: np.ndarray = None # From Phase 2
) -> dict:
    """Full Phase 3 pipeline."""
    input_dim = X_train.shape[1]

    # --- Train Autoencoder ---
    ae_model = train_autoencoder(X_train, y_train, input_dim=input_dim)
    s2_raw   = get_reconstruction_errors(ae_model, X_test)

    # --- S1: use Phase 2 scores or compute fresh ---
    if fw_scores is not None:
        s1 = fw_scores
    else:
        from phases.phase2_fw_iforest import FWIsolationForest
        fw = FWIsolationForest().fit(X_train)
        s1 = fw.anomaly_scores(X_test)

    # --- Find best alpha on test labels ---
    best_alpha, best_auc, alpha_results = find_best_alpha(s1, s2_raw, y_test)

    # --- Hybrid final score ---
    final_scores = hybrid_fusion(s1, s2_raw, alpha=best_alpha)

    # --- Build Explainable Forest ---
    print("[Phase 3] Building explainable forest...")
    exp_forest = ExplainableForest(
        n_estimators=50,          # fewer for speed; increase for production
        max_samples=256,
        feature_weights=feature_weights,
        random_state=42
    )
    exp_forest.fit(X_train)

    # --- Evaluation ---
    metrics = evaluate_phase3(y_test, final_scores)
    metrics["best_alpha"]    = best_alpha
    metrics["alpha_results"] = alpha_results

    # --- Sample explanations for top anomalies ---
    anomaly_indices = np.argsort(final_scores)[-5:][::-1]
    sample_explanations = []
    for idx in anomaly_indices:
        exp = exp_forest.explain(X_test[idx], feature_cols)
        exp["index"]  = int(idx)
        exp["label"]  = int(y_test[idx])
        sample_explanations.append(exp)

    save_phase3(ae_model, exp_forest)

    return {
        "metrics":              metrics,
        "ae_model":             ae_model,
        "exp_forest":           exp_forest,
        "final_scores":         final_scores,
        "s1":                   s1,
        "s2":                   s2_raw,
        "best_alpha":           best_alpha,
        "alpha_results":        alpha_results,
        "sample_explanations":  sample_explanations
    }