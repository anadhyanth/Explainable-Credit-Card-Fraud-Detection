"""
Standalone evaluation of the persisted best model on the held-out test set.
Produces a confusion matrix plot, ROC curve plot, and a text classification
report saved under reports/.

Usage:
    python -m src.evaluate
"""

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for servers/CI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

import config
from src.utils import get_logger, load_artifact, save_json

logger = get_logger(__name__)


def plot_confusion_matrix(y_true, y_pred, path=config.CONFUSION_MATRIX_PLOT):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix plot to %s", path)


def plot_roc_curve(y_true, y_proba, path=config.ROC_CURVE_PLOT):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved ROC curve plot to %s", path)


def main():
    model = load_artifact(config.MODEL_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    X_test = test_df[config.FEATURE_COLUMNS]
    y_test = test_df[config.TARGET_COL]

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= config.DECISION_THRESHOLD).astype(int)

    report = classification_report(y_test, preds, target_names=["Legit", "Fraud"], output_dict=True)
    save_json(report, config.REPORTS_DIR / "classification_report.json")

    plot_confusion_matrix(y_test, preds)
    plot_roc_curve(y_test, proba)

    logger.info("Evaluation complete. See reports/ for artifacts.")


if __name__ == "__main__":
    main()
