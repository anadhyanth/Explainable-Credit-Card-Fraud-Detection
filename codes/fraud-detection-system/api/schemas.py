"""Pydantic schemas for request validation and response serialization."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

import config


class TransactionRequest(BaseModel):
    """
    A single credit-card transaction, matching the Kaggle dataset schema:
    Time, Amount, and 28 PCA-anonymized features V1..V28.
    """

    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    def to_ordered_dict(self) -> dict:
        data = self.model_dump()
        return {col: data[col] for col in config.FEATURE_COLUMNS}

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406.0, "Amount": 149.62,
                "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
                "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
                "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
                "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
                "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
                "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
                "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210,
            }
        }


class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRequest]


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    decision_threshold: float
    top_contributing_features: List[FeatureContribution]
    model_version: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    explainer_loaded: bool
