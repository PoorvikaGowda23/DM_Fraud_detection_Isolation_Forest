"""
Flask API — serves all 3 phases and exposes endpoints for the frontend.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from phases.phase1_baseline          import run_phase1, load_dataset, preprocess_data
from phases.phase2_fw_iforest        import run_phase2
from phases.phase3_hybrid_explainable import run_phase3

_BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(_BASE, "..", "frontend", "templates"),
    static_folder=os.path.join(_BASE,   "..", "frontend", "static")
)
CORS(app)

# ── Global state (in-memory for demo; use DB for production) ──
_state = {
    "dataset_path": None,
    "X_train": None, "X_test": None,
    "y_train": None, "y_test": None,
    "feature_cols": None, "scaler": None,
    "phase1": None,
    "phase2": None,
    "phase3": None,
}


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def _safe_metrics(metrics: dict) -> dict:
    """Convert numpy types to plain Python for JSON serialisation."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, np.floating)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


# ─────────────────────────────────────────────
# Frontend Route
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────
# Phase 1 — Upload dataset + run baseline
# ─────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    """Accept uploaded CSV and store path."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    upload_path = os.path.join(_BASE, "uploads", "dataset.csv")
    os.makedirs(os.path.join(_BASE, "uploads"), exist_ok=True)
    f.save(upload_path)
    _state["dataset_path"] = upload_path

    # Quick preview
    df = pd.read_csv(upload_path, nrows=5)
    return jsonify({
        "message": "Dataset uploaded successfully",
        "preview": df.to_dict(orient="records"),
        "columns": list(df.columns)
    })


@app.route("/api/phase1", methods=["POST"])
def phase1():
    """Run Phase 1 — Baseline Isolation Forest."""
    if not _state["dataset_path"]:
        return jsonify({"error": "Upload dataset first"}), 400

    result = run_phase1(_state["dataset_path"])

    _state["X_train"]      = result["X_train"]
    _state["X_test"]       = result["X_test"]
    _state["y_train"]      = result.get("y_train")
    _state["y_test"]       = result["y_test"]
    _state["feature_cols"] = result["feature_cols"]
    _state["scaler"]       = result["scaler"]
    _state["phase1"]       = result

    # Score distribution for chart
    scores = result["anomaly_scores"].tolist()
    labels = result["y_test"].tolist()

    return jsonify({
        "metrics": _safe_metrics(result["metrics"]),
        "score_distribution": {
            "scores": scores[:500],    # limit payload
            "labels": labels[:500]
        }
    })


# ─────────────────────────────────────────────
# Phase 2 — Feature-Weighted iForest
# ─────────────────────────────────────────────

@app.route("/api/phase2", methods=["POST"])
def phase2():
    """Run Phase 2 — FW-iForest."""
    if _state["X_train"] is None:
        return jsonify({"error": "Run Phase 1 first"}), 400

    result = run_phase2(
        _state["X_train"],
        _state["X_test"],
        _state["y_test"],
        _state["feature_cols"]
    )
    _state["phase2"] = result

    importance = result["importance_df"].to_dict(orient="records")
    scores      = result["anomaly_scores"].tolist()
    labels      = _state["y_test"].tolist()

    return jsonify({
        "metrics": _safe_metrics(result["metrics"]),
        "feature_importance": importance[:15],
        "score_distribution": {
            "scores": scores[:500],
            "labels": labels[:500]
        }
    })


# ─────────────────────────────────────────────
# Phase 3 — Hybrid + Explainability
# ─────────────────────────────────────────────

@app.route("/api/phase3", methods=["POST"])
def phase3():
    """Run Phase 3 — Hybrid + Explainability."""
    if _state["X_train"] is None:
        return jsonify({"error": "Run Phase 1 first"}), 400

    fw_scores       = _state["phase2"]["anomaly_scores"] if _state["phase2"] else None
    feature_weights = _state["phase2"]["feature_weights"] if _state["phase2"] else None

    result = run_phase3(
        _state["X_train"],
        _state["X_test"],
        _state["y_train"],
        _state["y_test"],
        _state["feature_cols"],
        fw_scores=fw_scores,
        feature_weights=feature_weights
    )
    _state["phase3"] = result

    return jsonify({
        "metrics":             _safe_metrics(result["metrics"]),
        "best_alpha":          result["best_alpha"],
        "alpha_results":       result["alpha_results"],
        "sample_explanations": result["sample_explanations"],
        "score_distribution": {
            "scores": result["final_scores"][:500].tolist(),
            "labels": _state["y_test"][:500].tolist()
        }
    })


# ─────────────────────────────────────────────
# Comparison endpoint
# ─────────────────────────────────────────────

@app.route("/api/compare", methods=["GET"])
def compare():
    """Return side-by-side metrics for all 3 phases."""
    phases = []
    for key in ("phase1", "phase2", "phase3"):
        if _state[key]:
            phases.append(_safe_metrics(_state[key]["metrics"]))
    return jsonify({"phases": phases})


# ─────────────────────────────────────────────
# Explain single transaction
# ─────────────────────────────────────────────

@app.route("/api/explain", methods=["POST"])
def explain():
    """
    Explain a single transaction.
    Body: { "index": <int> }  — index into test set
    """
    if not _state["phase3"]:
        return jsonify({"error": "Run Phase 3 first"}), 400

    body  = request.get_json()
    idx   = int(body.get("index", 0))
    x     = _state["X_test"][idx]
    exp   = _state["phase3"]["exp_forest"].explain(x, _state["feature_cols"])
    exp["true_label"]   = int(_state["y_test"][idx])
    exp["final_score"]  = round(float(_state["phase3"]["final_scores"][idx]), 4)
    return jsonify(exp)


if __name__ == "__main__":
    app.run(debug=True, port=5000)