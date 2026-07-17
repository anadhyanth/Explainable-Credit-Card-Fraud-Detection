"""
Data preprocessing pipeline for the credit card fraud dataset.

Responsibilities:
  1. Load the raw Kaggle CSV.
  2. Validate schema and check for missing/duplicate records.
  3. Scale `Amount` and `Time` (the only two non-PCA features) with a
     RobustScaler, since transaction amounts are heavily skewed with outliers.
  4. Stratified train/test split so the (very rare) fraud class is
     represented proportionally in both sets.
  5. Handle class imbalance on the TRAINING split only (never touch test
     data) using SMOTE, so evaluation reflects real-world class ratios.

Run directly to produce data/train.csv and data/test.csv plus a saved scaler:
    python -m src.data_preprocessing
"""

from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import config
from src.utils import get_logger, save_artifact

logger = get_logger(__name__)


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'creditcard.csv' from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the data/ directory."
        )
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing_cols = set(config.FEATURE_COLUMNS + [config.TARGET_COL]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    n_dupes = df.duplicated().sum()
    n_na = df.isna().sum().sum()
    logger.info("Duplicate rows: %s | Missing values: %s", n_dupes, n_na)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    logger.info("Dropped %s duplicate/incomplete rows", before - len(df))
    return df


def split_data(df: pd.DataFrame):
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Split: train=%s (fraud=%.4f%%), test=%s (fraud=%.4f%%)",
        len(X_train), y_train.mean() * 100,
        len(X_test), y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale Amount and Time only — the V1..V28 columns are already PCA-whitened."""
    scaler = RobustScaler()
    cols_to_scale = [config.TIME_COL, config.AMOUNT_COL]

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    save_artifact(scaler, config.SCALER_PATH)
    logger.info("Saved fitted scaler to %s", config.SCALER_PATH)
    return X_train, X_test, scaler


def handle_imbalance(X_train: pd.DataFrame, y_train: pd.Series):
    if not config.USE_SMOTE:
        return X_train, y_train

    logger.info(
        "Class distribution before SMOTE: %s", y_train.value_counts().to_dict()
    )
    smote = SMOTE(
        sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
        random_state=config.RANDOM_STATE,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(
        "Class distribution after SMOTE: %s", pd.Series(y_res).value_counts().to_dict()
    )
    return X_res, y_res


def run_pipeline():
    df = load_raw_data()
    validate_schema(df)
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, _ = scale_features(X_train, X_test)
    X_train_res, y_train_res = handle_imbalance(X_train, y_train)

    train_df = X_train_res.copy()
    train_df[config.TARGET_COL] = y_train_res.values
    test_df = X_test.copy()
    test_df[config.TARGET_COL] = y_test.values

    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)
    logger.info(
        "Saved processed train (%s rows) and test (%s rows) sets to data/",
        len(train_df), len(test_df),
    )
    return train_df, test_df


if __name__ == "__main__":
    run_pipeline()
