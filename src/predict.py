from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


class HousePricePredictor:
    def __init__(self, model_path: str = "artifacts/model.joblib") -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. Run: python src/train.py"
            )
        self.model = joblib.load(path)

    def predict(self, location: str, total_sqft: float, bath: float, bhk: int) -> float:
        sample = pd.DataFrame(
            [
                {
                    "location": location.strip(),
                    "total_sqft": total_sqft,
                    "bath": bath,
                    "bhk": bhk,
                }
            ]
        )
        pred = self.model.predict(sample)[0]
        return float(pred)
