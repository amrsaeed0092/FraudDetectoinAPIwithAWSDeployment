# Fraud Detection Platform

Credit-card fraud detection platform built with FastAPI, Pydantic,
scikit-learn, PySpark, React, Docker, MLflow and AWS.

## First milestone

Run a secure FastAPI application locally and verify `GET /health`.

## Backend start

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` and test `GET /health`.

## Folder responsibilities

- `data/`: raw, cleaned and feature-ready data plus processing code.
- `model/`: model training, evaluation, registry integration and model artifacts.
- `validations/`: Pydantic request/response schemas.
- `utilities/`: reusable, side-effect-free helper functions.
- `configuration/`: settings and environment configuration.
- `log/`: runtime logs; log files are not committed.
- `api/`: FastAPI routers and dependency wiring.
- `database/`: database models, repositories and sessions.


## Incoming transaction
→ feature scaler
→ KMeans finds cluster
→ Cluster 0: XGBoost
→ Cluster 2: Extra Trees
→ Cluster 3: HistGradientBoosting
→ Clusters 1/4/5: global fallback Logistic Regression