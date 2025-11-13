"""Training utilities for the tennis stat predictor."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import TrainingConfig
from .data import load_matches, split_train_test
from .features import select_model_features
from .model import build_regression_pipeline


def _evaluate_predictions(
    y_true: pd.DataFrame, y_pred: np.ndarray, target_columns: Iterable[str]
) -> Dict[str, Dict[str, float]]:
    """Compute MAE and RMSE metrics for each target column."""

    metrics: Dict[str, Dict[str, float]] = {}
    for index, column in enumerate(target_columns):
        column_true = y_true.iloc[:, index]
        column_pred = y_pred[:, index]
        mae = mean_absolute_error(column_true, column_pred)
        rmse = mean_squared_error(column_true, column_pred, squared=False)
        metrics[column] = {"mae": float(mae), "rmse": float(rmse)}
    return metrics


def train_and_evaluate(config: TrainingConfig) -> Dict[str, Dict[str, float]]:
    """Train the model defined by ``config`` and return evaluation metrics."""

    config.ensure_output_dir()

    raw_df = load_matches(config.data)
    train_df, test_df = split_train_test(raw_df, config.data)

    feature_columns = (config.model.categorical_features or []) + (
        config.model.numerical_features or []
    )
    x_train_full = select_model_features(train_df, feature_columns)
    x_test_full = select_model_features(test_df, feature_columns)

    y_train = train_df[config.data.target_columns]
    y_test = test_df[config.data.target_columns]

    pipeline = build_regression_pipeline(config.model, feature_columns)
    pipeline.fit(x_train_full, y_train)

    predictions = pipeline.predict(x_test_full)
    metrics = _evaluate_predictions(y_test, predictions, config.data.target_columns)

    _save_artifacts(
        pipeline=pipeline,
        metrics=metrics,
        config=config,
        feature_columns=feature_columns,
    )
    return metrics


def _save_artifacts(
    *,
    pipeline,
    metrics: Dict[str, Dict[str, float]],
    config: TrainingConfig,
    feature_columns: Iterable[str],
) -> None:
    """Persist the trained pipeline and evaluation metrics to disk."""

    model_path = Path(config.output_dir) / "trained_model.joblib"
    metrics_path = Path(config.output_dir) / "metrics.json"
    metadata_path = Path(config.output_dir) / "metadata.json"

    joblib.dump(pipeline, model_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "target_columns": config.data.target_columns,
        "feature_columns": list(feature_columns),
        "data_path": str(config.data.data_path),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
