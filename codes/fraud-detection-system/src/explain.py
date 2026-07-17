"""
SHAP-based explainability for the fraud detection model.

Provides:
  - build_explainer(): fits a SHAP TreeExplainer against the trained model
    and caches it to disk so the API doesn't recompute it on every request.
  - global_feature_importance(): saves a bar plot of mean |SHAP value| per
    feature across the test set (used for model-level reporting).
  - explain_transaction(): returns the top contributing features (with
    signed SHAP values) for a single transaction, used by the API to give
    human-readable, transaction-level explanations.

Usage (standalone, to regenerate the global importance plot + explainer):
    python -m src.explain
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

import config
from src.utils import get_logger, load_artifact, save_artifact

logger = get_logger(__name__)


def build_explainer(model=None):
    if model is None:
        model = load_artifact(config.MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    save_artifact(explainer, config.EXPLAINER_PATH)
    logger.info("Saved SHAP explainer to %s", config.EXPLAINER_PATH)
    return explainer


def load_or_build_explainer():
    try:
        return load_artifact(config.EXPLAINER_PATH)
    except FileNotFoundError:
        logger.info("No cached explainer found, building a new one.")
        return build_explainer()


def global_feature_importance(explainer, X_sample: pd.DataFrame, path=config.FEATURE_IMPORTANCE_PLOT):
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved global SHAP importance plot to %s", path)


def explain_transaction(explainer, x_row: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Return the top_n features driving the prediction for a single transaction.

    x_row must be a single-row DataFrame with the same columns/order used
    during training (config.FEATURE_COLUMNS).
    """
    shap_values = explainer.shap_values(x_row)
    values = shap_values[0] if hasattr(shap_values, "__len__") else shap_values

    contributions = pd.Series(values, index=x_row.columns).sort_values(
        key=lambda s: s.abs(), ascending=False
    )

    top = contributions.head(top_n)
    return [
        {
            "feature": feat,
            "shap_value": round(float(val), 5),
            "direction": "increases_fraud_risk" if val > 0 else "decreases_fraud_risk",
        }
        for feat, val in top.items()
    ]


def main():
    model = load_artifact(config.MODEL_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    X_sample = test_df[config.FEATURE_COLUMNS].sample(
        n=min(2000, len(test_df)), random_state=config.RANDOM_STATE
    )

    explainer = build_explainer(model)
    global_feature_importance(explainer, X_sample)

    # Demonstrate a single transaction explanation in the logs.
    example = explain_transaction(explainer, X_sample.iloc[[0]])
    logger.info("Example transaction explanation: %s", example)


if __name__ == "__main__":
    main()
