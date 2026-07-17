"""
Basic API tests. Requires a trained model + explainer to already exist
(run `python -m src.train` and `python -m src.explain` first), otherwise
the /predict tests are skipped and only /health is checked.

Run:
    pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

import config
from api.app import app
from api.schemas import TransactionRequest

client = TestClient(app)

MODEL_READY = config.MODEL_PATH.exists() and config.EXPLAINER_PATH.exists()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body


@pytest.mark.skipif(not MODEL_READY, reason="Model/explainer artifacts not trained yet")
def test_predict_endpoint_valid_transaction():
    example = TransactionRequest.Config.json_schema_extra["example"]
    response = client.post("/predict", json=example)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert len(body["top_contributing_features"]) > 0


def test_predict_endpoint_missing_field():
    incomplete_payload = {"Time": 100.0, "Amount": 50.0}  # missing V1..V28
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422  # FastAPI validation error


@pytest.mark.skipif(not MODEL_READY, reason="Model/explainer artifacts not trained yet")
def test_batch_predict_endpoint():
    example = TransactionRequest.Config.json_schema_extra["example"]
    response = client.post("/predict/batch", json={"transactions": [example, example]})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
