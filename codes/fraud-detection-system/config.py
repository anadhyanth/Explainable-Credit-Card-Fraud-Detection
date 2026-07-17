"""
Central configuration for the fraud detection system.
All paths, hyperparameters, and runtime settings live here so that
training, evaluation, explanation, and serving code stay in sync.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", DATA_DIR / "creditcard.csv"))
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

MODEL_PATH = MODEL_DIR / "xgboost_fraud_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
EXPLAINER_PATH = MODEL_DIR / "shap_explainer.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PLOT = REPORTS_DIR / "shap_global_importance.png"
CONFUSION_MATRIX_PLOT = REPORTS_DIR / "confusion_matrix.png"
ROC_CURVE_PLOT = REPORTS_DIR / "roc_curve.png"

for d in (DATA_DIR, MODEL_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data schema (Kaggle "Credit Card Fraud Detection" dataset)
# ---------------------------------------------------------------------------
TARGET_COL = "Class"
AMOUNT_COL = "Amount"
TIME_COL = "Time"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = [TIME_COL, AMOUNT_COL] + PCA_FEATURES

# ---------------------------------------------------------------------------
# Train/test split & imbalance handling
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_SAMPLING_STRATEGY = 0.10  # oversample minority class to 10% of majority
USE_SMOTE = True

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
XGBOOST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

DECISION_TREE_PARAMS = {
    "max_depth": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# Classification threshold used to convert probability -> label.
# Tuned separately per business cost tradeoff; 0.5 is the default starting point.
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", 0.5))

# ---------------------------------------------------------------------------
# API / deployment
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
