"""Model construction helpers."""
from __future__ import annotations

from typing import Iterable

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from .config import ModelConfig
from .features import build_feature_pipeline


def build_regression_pipeline(config: ModelConfig, feature_columns: Iterable[str]) -> Pipeline:
    """Create a multi-output regression pipeline for tennis stats."""

    feature_transformer = build_feature_pipeline(
        categorical=config.categorical_features,
        numerical=config.numerical_features,
    )

    base_estimator = GradientBoostingRegressor(
        n_estimators=config.gradient_boosting_estimators,
        learning_rate=config.gradient_boosting_learning_rate,
        max_depth=config.gradient_boosting_max_depth,
        random_state=42,
    )

    model = MultiOutputRegressor(base_estimator)

    return Pipeline([
        ("features", feature_transformer),
        ("regressor", model),
    ])
