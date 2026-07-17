# Explainable Credit Card Fraud Detection (XGBoost + SHAP)

Production-grade, end-to-end pipeline for detecting fraudulent credit card
transactions, with model comparison, SHAP-based explainability, and a
FastAPI service ready for containerized deployment.

## Project structure

```
fraud-detection-system/
├── config.py                  # Central config: paths, hyperparameters, thresholds
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── data/                      # Raw + processed data (not committed)
├── models/                    # Saved model, scaler, SHAP explainer
├── reports/                   # Metrics, plots, classification report
├── src/
│   ├── data_preprocessing.py  # Load, clean, scale, split, SMOTE
│   ├── train.py               # Train & compare 4 models, save the best
│   ├── evaluate.py            # Confusion matrix / ROC plots on test set
│   ├── explain.py             # SHAP global + per-transaction explanations
│   └── utils.py                # Logging, artifact IO
├── api/
│   ├── app.py                 # FastAPI service (predict, batch, health)
│   └── schemas.py             # Pydantic request/response models
└── tests/
    └── test_api.py             # API unit tests
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place `creditcard.csv` into `data/`.

## 2. Train the models

```bash
python -m src.train
```

This will:
- Run preprocessing automatically if `data/train.csv` / `data/test.csv` don't exist yet
  (RobustScaler on `Amount`/`Time`, stratified 80/20 split, SMOTE oversampling on the
  training set only).
- Train **XGBoost, Random Forest, Logistic Regression, and Decision Tree**.
- Evaluate all four on the untouched test set using ROC-AUC, PR-AUC, precision,
  recall, and F1.
- Save the best model (by ROC-AUC) to `models/xgboost_fraud_model.joblib` and a
  full comparison to `reports/metrics.json`.

## 3. Evaluate & visualize

```bash
python -m src.evaluate
```

Produces `reports/confusion_matrix.png`, `reports/roc_curve.png`, and
`reports/classification_report.json`.

## 4. Generate SHAP explanations

```bash
python -m src.explain
```

Builds and caches a `shap.TreeExplainer`, saves a global feature-importance
bar chart (`reports/shap_global_importance.png`), and logs an example
per-transaction explanation.

## 5. Run the API locally

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Docs available at `http://localhost:8000/docs`.

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0, "Amount": 149.62,
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
    "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
    "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210
  }'
```

Example response:

```json
{
  "fraud_probability": 0.0123,
  "is_fraud": false,
  "decision_threshold": 0.5,
  "top_contributing_features": [
    {"feature": "V14", "shap_value": -0.842, "direction": "decreases_fraud_risk"},
    {"feature": "V4", "shap_value": 0.231, "direction": "increases_fraud_risk"}
  ],
  "model_version": "xgboost_fraud_model"
}
```

## 6. Deploy with Docker

```bash
docker compose up --build -d
```

This builds a slim, non-root, multi-stage image, mounts `models/` read-only,
and exposes the API on port 8000 with a built-in healthcheck.

For a single-container run without compose:

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro fraud-detection-api
```

## 7. Tests

```bash
pytest tests/ -v
```

## Design notes

- **Imbalance handling**: SMOTE is applied only to the training split (never
  the test split), so reported metrics reflect real-world class ratios.
  `scale_pos_weight`-style class weighting is also available via
  `RANDOM_FOREST_PARAMS["class_weight"]` / `LOGISTIC_REGRESSION_PARAMS["class_weight"]`
  for models where oversampling alone isn't sufficient.
- **Metric choice**: ROC-AUC and PR-AUC are prioritized over raw accuracy,
  since fraud is a rare-event problem where accuracy is misleading.
- **Explainability**: SHAP's `TreeExplainer` is used (exact, fast for tree
  ensembles) rather than `KernelExplainer`, which would be far slower for
  a serving path.
- **Threshold tuning**: `config.DECISION_THRESHOLD` is exposed as an
  environment variable so the fraud/legit cutoff can be tuned post-hoc for
  precision/recall tradeoffs without retraining.
- **Scaling to production traffic**: swap the `joblib` file load for a model
  registry (e.g., MLflow Model Registry) and put the API behind a queue or
  autoscaled deployment (Kubernetes HPA, AWS ECS, etc.) if request volume
  requires it.
