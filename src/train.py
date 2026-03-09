from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src.data import load_and_clean_data


def build_model() -> Pipeline:
    numeric_features = ["total_sqft", "bath", "bhk"]
    categorical_features = ["location"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def evaluate_and_train(df: pd.DataFrame, rows_raw: int) -> tuple[Pipeline, dict]:
    x = df.drop("price", axis="columns")
    y = df["price"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
    model = build_model()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    test_r2 = model.score(x_test, y_test)
    train_r2 = model.score(x_train, y_train)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

    n = len(y_test)
    pre = model.named_steps["preprocessor"]
    transformed_features = pre.fit_transform(x_train).shape[1]
    adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - transformed_features - 1)

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    cv_scores = cross_val_score(build_model(), x, y, cv=cv)

    grid_results = _run_grid_search(x, y, cv)

    metrics = {
        "rows_raw": int(rows_raw),
        "rows_final": int(df.shape[0]),
        "feature_count": int(transformed_features),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "adjusted_test_r2": float(adjusted_r2),
        "mae_lakhs": float(mae),
        "rmse_lakhs": float(rmse),
        "mape_pct": float(mape),
        "cv_r2_scores": [float(v) for v in cv_scores],
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "model_comparison_cv_r2": grid_results,
    }
    return model, metrics


def _run_grid_search(x: pd.DataFrame, y: pd.Series, cv: ShuffleSplit) -> dict:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["total_sqft", "bath", "bhk"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["location"]),
        ]
    )

    models = {
        "linear_regression": {
            "pipeline": Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())]),
            "params": {"regressor__fit_intercept": [True, False]},
        },
        "lasso": {
            "pipeline": Pipeline([("preprocessor", preprocessor), ("regressor", Lasso())]),
            "params": {"regressor__alpha": [1, 2], "regressor__selection": ["random", "cyclic"]},
        },
        "decision_tree": {
            "pipeline": Pipeline([("preprocessor", preprocessor), ("regressor", DecisionTreeRegressor())]),
            "params": {
                "regressor__criterion": ["friedman_mse", "squared_error"],
                "regressor__splitter": ["best", "random"],
            },
        },
    }

    results: dict[str, dict] = {}
    for name, config in models.items():
        gs = GridSearchCV(config["pipeline"], config["params"], cv=cv, return_train_score=False)
        gs.fit(x, y)
        results[name] = {
            "best_score": float(gs.best_score_),
            "best_params": gs.best_params_,
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Bengaluru house price model and export artifacts.")
    parser.add_argument(
        "--data-path",
        default="Bengaluru_House_Data.csv",
        help="Path to Bengaluru_House_Data.csv",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Directory to write model and metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = int(pd.read_csv(args.data_path).shape[0])
    df = load_and_clean_data(args.data_path)
    model, metrics = evaluate_and_train(df, rows_raw=raw_rows)

    model_path = artifact_dir / "model.joblib"
    metrics_path = artifact_dir / "metrics.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Test R2: {metrics['test_r2']:.4f} | RMSE(lakhs): {metrics['rmse_lakhs']:.2f}")


if __name__ == "__main__":
    main()
