# Isolation Forest Based Anomaly Detection System
# Credit Card Fraud Detection

## Project Structure

```
fraud_detection/
├── backend/
│   ├── app.py                          ← Flask API server
│   └── phases/
│       ├── phase1_baseline.py          ← Phase 1: Baseline Isolation Forest
│       ├── phase2_fw_iforest.py        ← Phase 2: Feature-Weighted iForest
│       └── phase3_hybrid_explainable.py← Phase 3: Hybrid + Explainability
├── frontend/
│   └── templates/
│       └── index.html                  ← Full dashboard UI
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
cd backend
python app.py
```

Then open http://localhost:5000 in your browser.

## Dataset

Download from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

File: creditcard.csv

## Phase Flow

Phase 1 → Phase 2 → Phase 3 → Compare

Each phase builds on the previous one.
Results from Phase 1 (preprocessing) are reused in Phases 2 and 3.
