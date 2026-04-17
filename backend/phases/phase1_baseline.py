"""
Phase 1 — Baseline Isolation Forest
Implements the original iForest algorithm from Liu et al. (2008).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score, confusion_matrix
)
import joblib
import os


# ──────────────────────────────────────────────
# 1. Dataset Preparation
# ──────────────────────────────────────────────

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load credit card fraud CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"[Phase 1] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Handle missing values, duplicates, scaling.
    Returns X_train, X_test, y_train, y_test, scaler.
    """
    # Drop duplicates
    df = df.drop_duplicates()

    # Drop missing values
    df = df.dropna()

    # Separate features and label
    # Kaggle credit card dataset: label column is 'Class' (0=normal, 1=fraud)
    label_col = "Class"
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].values
    y = df[label_col].values

    # Normalize numerical features (Amount, Time — V1-V28 are already PCA-scaled)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split (stratified to preserve fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[Phase 1] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"[Phase 1] Fraud in test set: {y_test.sum()} / {len(y_test)}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


# ──────────────────────────────────────────────
# 2. Build Isolation Forest
# ──────────────────────────────────────────────

def build_isolation_forest(
    n_estimators: int = 100,
    max_samples: int = 256,
    contamination: float = 0.001,
    random_state: int = 42
) -> IsolationForest:
    """
    Build the baseline Isolation Forest model.
    contamination = expected fraction of anomalies in dataset.
    """
    model = IsolationForest(
        n_estimators=n_estimators,   # number of trees t
        max_samples=max_samples,     # sub-sampling size ψ
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    return model


# ──────────────────────────────────────────────
# 3. Training
# ──────────────────────────────────────────────

def train_phase1(X_train: np.ndarray, model: IsolationForest) -> IsolationForest:
    """Fit the Isolation Forest on training data (unsupervised)."""
    model.fit(X_train)
    print("[Phase 1] Isolation Forest trained successfully.")
    return model


# ──────────────────────────────────────────────
# 4. Anomaly Scoring & Prediction
# ──────────────────────────────────────────────

def get_anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Returns anomaly scores s(x,n) ∈ (0,1].
    sklearn's decision_function returns negative scores;
    we convert to a [0,1] anomaly score where 1 = most anomalous.
    """
    raw_scores = model.decision_function(X)       # lower = more anomalous
    anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (
        raw_scores.max() - raw_scores.min() + 1e-10
    )
    return anomaly_scores


def predict_labels(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Returns binary labels: 1 = anomaly, 0 = normal.
    sklearn predict returns -1 for anomalies, 1 for normal.
    """
    raw_pred = model.predict(X)
    return (raw_pred == -1).astype(int)


# ──────────────────────────────────────────────
# 5. Evaluation
# ──────────────────────────────────────────────

def evaluate_phase1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray
) -> dict:
    """Compute evaluation metrics for Phase 1."""
    auc_roc    = roc_auc_score(y_true, anomaly_scores)
    avg_prec   = average_precision_score(y_true, anomaly_scores)
    cm         = confusion_matrix(y_true, y_pred)
    report     = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        "phase": "Phase 1 — Baseline Isolation Forest",
        "auc_roc": round(auc_roc, 4),
        "avg_precision": round(avg_prec, 4),
        "precision": round(report["1"]["precision"], 4),
        "recall":    round(report["1"]["recall"], 4),
        "f1_score":  round(report["1"]["f1-score"], 4),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n[Phase 1] AUC-ROC:    {metrics['auc_roc']}")
    print(f"[Phase 1] Avg Prec:   {metrics['avg_precision']}")
    print(f"[Phase 1] Precision:  {metrics['precision']}")
    print(f"[Phase 1] Recall:     {metrics['recall']}")
    print(f"[Phase 1] F1 Score:   {metrics['f1_score']}")

    return metrics


# ──────────────────────────────────────────────
# 6. Save model
# ──────────────────────────────────────────────

def save_model(model, scaler, path: str = None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "phase1")
    os.makedirs(path, exist_ok=True)
    joblib.dump(model,  f"{path}/iforest.pkl")
    joblib.dump(scaler, f"{path}/scaler.pkl")
    print(f"[Phase 1] Model saved to {path}/")


def load_model(path: str = "models/phase1"):
    model  = joblib.load(f"{path}/iforest.pkl")
    scaler = joblib.load(f"{path}/scaler.pkl")
    return model, scaler


# ──────────────────────────────────────────────
# 7. Run Phase 1 pipeline
# ──────────────────────────────────────────────

def run_phase1(filepath: str) -> dict:
    """Full Phase 1 pipeline. Returns metrics + artifacts."""
    df = load_dataset(filepath)
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess_data(df)

    model = build_isolation_forest(
        n_estimators=100,
        max_samples=256,
        contamination=0.001
    )
    model = train_phase1(X_train, model)

    anomaly_scores = get_anomaly_scores(model, X_test)
    y_pred         = predict_labels(model, X_test)
    metrics        = evaluate_phase1(y_test, y_pred, anomaly_scores)

    save_model(model, scaler)

    return {
        "metrics": metrics,
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "anomaly_scores": anomaly_scores,
        "y_pred": y_pred,
        "feature_cols": feature_cols
    }