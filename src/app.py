from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.predict import HousePricePredictor
from src.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Bengaluru House Price API", version="1.0.0")


def _load_metrics(metrics_path: str = "artifacts/metrics.json") -> dict:
    path = Path(metrics_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


try:
    predictor = HousePricePredictor()
except FileNotFoundError:
    predictor = None


@app.get("/health")
def health() -> dict:
    ready = predictor is not None
    return {"status": "ok", "model_loaded": ready}


@app.get("/metrics")
def metrics() -> dict:
    return _load_metrics()


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train first: python src/train.py",
        )
    value = predictor.predict(
        location=payload.location,
        total_sqft=payload.total_sqft,
        bath=payload.bath,
        bhk=payload.bhk,
    )
    return PredictionResponse(predicted_price_lakhs=round(value, 3))
