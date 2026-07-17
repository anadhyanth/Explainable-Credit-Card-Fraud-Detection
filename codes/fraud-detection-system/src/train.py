"""
Train and compare XGBoost, Random Forest, Logistic Regression, and Decision
Tree classifiers on the (SMOTE-balanced) training set, evaluate all of them
on the untouched test set, and persist the best-performing model by ROC-AUC.

Usage:
    python -m src.train
"""

import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from xgboost import XGBClassifier

import config
from src.data_preprocessing import run_pipeline
from src.utils import get_logger, save_artifact, save_json

logger = get_logger(__name__)


MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS),
    "decision_tree": lambda: DecisionTreeClassifier(**config.DECISION_TREE_PARAMS),
    "random_forest": lambda: RandomForestClassifier(**config.RANDOM_FOREST_PARAMS),
    "xgboost": lambda: XGBClassifier(**config.XGBOOST_PARAMS),
}


def evaluate_model(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= config.DECISION_THRESHOLD).astype(int)

    return {
        "roc_auc": round(roc_auc_score(y_test, proba), 5),
        "pr_auc": round(average_precision_score(y_test, proba), 5),
        "precision": round(precision_score(y_test, preds), 5),
        "recall": round(recall_score(y_test, preds), 5),
        "f1_score": round(f1_score(y_test, preds), 5),
    }


def train_all_models(X_train, y_train, X_test, y_test) -> dict:
    results = {}
    fitted_models = {}

    for name, factory in MODEL_REGISTRY.items():
        logger.info("Training %s ...", name)
        start = time.time()
        model = factory()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        metrics = evaluate_model(model, X_test, y_test)
        metrics["train_seconds"] = round(elapsed, 2)
        results[name] = metrics
        fitted_models[name] = model

        logger.info("%s -> %s", name, metrics)

    return results, fitted_models


def select_best_model(results: dict, fitted_models: dict, metric: str = "roc_auc"):
    best_name = max(results, key=lambda k: results[k][metric])
    logger.info("Best model selected: %s (%s=%s)", best_name, metric, results[best_name][metric])
    return best_name, fitted_models[best_name]


def main():
    if not (config.TRAIN_DATA_PATH.exists() and config.TEST_DATA_PATH.exists()):
        logger.info("Processed train/test files not found, running preprocessing pipeline...")
        run_pipeline()

    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    X_train = train_df[config.FEATURE_COLUMNS]
    y_train = train_df[config.TARGET_COL]
    X_test = test_df[config.FEATURE_COLUMNS]
    y_test = test_df[config.TARGET_COL]

    results, fitted_models = train_all_models(X_train, y_train, X_test, y_test)
    best_name, best_model = select_best_model(results, fitted_models)

    save_artifact(best_model, config.MODEL_PATH)
    save_json(
        {"best_model": best_name, "results": results},
        config.METRICS_PATH,
    )
    logger.info("Saved best model (%s) to %s", best_name, config.MODEL_PATH)
    logger.info("Saved comparison metrics to %s", config.METRICS_PATH)


if __name__ == "__main__":
    main()
