"""Utilities for loading and validating tennis match data."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DataConfig


class DataValidationError(RuntimeError):
    """Raised when the input data does not match the expected schema."""


REQUIRED_COLUMNS = {
    "date",
    "tournament",
    "surface",
    "round",
    "player",
    "opponent",
    "player_rank",
    "opponent_rank",
    "player_age",
    "opponent_age",
    "player_height_cm",
    "opponent_height_cm",
    "player_hand",
    "opponent_hand",
    "best_of",
    "match_duration_minutes",
}


def load_matches(config: DataConfig) -> pd.DataFrame:
    """Load the tennis match dataset and validate that it is well-formed."""

    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}")

    df = pd.read_csv(path, parse_dates=[config.date_column])
    missing = REQUIRED_COLUMNS.union(config.target_columns) - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise DataValidationError(
            "The dataset is missing the following required columns: " f"{missing_str}"
        )

    df = df.sort_values(config.date_column).reset_index(drop=True)
    return df


def split_train_test(
    df: pd.DataFrame, config: DataConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train and test partitions by date."""

    if df.empty:
        raise DataValidationError("The dataset is empty. Add more historical matches.")

    split_index = int(len(df) * (1 - config.test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    if train_df.empty or test_df.empty:
        raise DataValidationError(
            "Unable to split dataset into train/test partitions. "
            "Please provide more historical records or adjust test_size."
        )
    return train_df, test_df


def get_feature_target_frames(
    df: pd.DataFrame, target_columns: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the feature and target frames from a dataframe."""

    targets = list(target_columns)
    features = df.drop(columns=targets)
    target_df = df[targets].copy()
    return features, target_df
