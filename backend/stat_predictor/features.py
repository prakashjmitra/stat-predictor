"""Feature engineering helpers for the stat predictor."""
from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_feature_pipeline(
    categorical: Optional[Iterable[str]], numerical: Optional[Iterable[str]]
) -> ColumnTransformer:
    """Create a column transformer that encodes categorical and numeric features."""

    categorical = list(categorical or [])
    numerical = list(numerical or [])

    transformers: List[tuple[str, object, List[str]]] = []
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            )
        )
    if numerical:
        transformers.append(
            (
                "numerical",
                StandardScaler(),
                numerical,
            )
        )

    if not transformers:
        raise ValueError("At least one of categorical or numerical features must be provided")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def select_model_features(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Return a dataframe with the requested columns if they exist."""

    missing = set(feature_columns) - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"The dataset is missing the following feature columns: {missing_str}")
    return df[list(feature_columns)].copy()
