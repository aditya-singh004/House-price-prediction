# Bengaluru House Price Prediction API

Production-ready API for predicting Bengaluru house prices (in lakhs) using the cleaned workflow from `code.ipynb`.

## Project Structure

- `src/data.py`: Data cleaning and outlier removal logic
- `src/train.py`: Training + evaluation + artifact export
- `src/app.py`: FastAPI server
- `src/predict.py`: Model loader and inference helper
- `artifacts/model.joblib`: Trained model artifact
- `artifacts/metrics.json`: Evaluation metrics

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Train Model

```bash
python -m src.train --data-path Bengaluru_House_Data.csv --artifact-dir artifacts
```

## 3) Run API Locally

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Swagger docs: `http://127.0.0.1:8000/docs`

## 4) Test Prediction Endpoint

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"location\":\"Whitefield\",\"total_sqft\":1200,\"bath\":2,\"bhk\":2}"
```

## Docker

Build and run:

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

## Deployment Recommendation

For this project, deploy as a containerized FastAPI service.

- Best default: **Render**
  - Simple Docker deployment from GitHub
  - Free/low-cost starter options
  - Easy HTTPS and health checks
- Good alternative: **Railway**
  - Fast setup and smooth developer experience
- If you want an ML portfolio demo page + API: **Hugging Face Spaces** (Docker SDK)

Recommended production settings:
- Keep model artifacts in `artifacts/`
- Use `/health` for uptime checks
- Use `/metrics` to expose model performance metadata
- Add request logging and API rate limiting before public launch

## Current Model Metrics

Metrics are exported to `artifacts/metrics.json` during training.
