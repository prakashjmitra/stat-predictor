"""Prediction utilities for the tennis stat predictor."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import joblib
import pandas as pd


def load_model(model_path: Path) -> object:
    """Load a previously trained pipeline."""

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    return joblib.load(model_path)


def predict_stats(
    pipeline: object,
    records: Iterable[Mapping[str, object]],
    *,
    target_columns: Iterable[str],
) -> pd.DataFrame:
    """Run inference using the trained pipeline."""

    frame = pd.DataFrame(records)
    predictions = pipeline.predict(frame)
    return pd.DataFrame(predictions, columns=list(target_columns))
