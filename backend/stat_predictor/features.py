"""Feature engineering helpers for the stat predictor."""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_feature_pipeline(
    categorical: Optional[Iterable[str]], numerical: Optional[Iterable[str]]
) -> ColumnTransformer:
    """Create a column transformer that encodes categorical and numeric features."""

    categorical = list(categorical or [])
    numerical = list(numerical or [])

    transformers: List[Tuple[str, object, List[str]]] = []
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    if numerical:
        transformers.append(
            (
                "numerical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
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
