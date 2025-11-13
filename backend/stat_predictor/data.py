"""Utilities for loading and validating tennis match data."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import requests

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
        if not config.source_url:
            raise FileNotFoundError(
                textwrap.dedent(
                    f"""
                    Could not find dataset at {path} and no ``source_url`` was provided.
                    Provide a local CSV path or configure a download URL.
                    """
                ).strip()
            )
        _download_dataset(config.source_url, path)

    df = pd.read_csv(path)
    df = _normalize_dataset(df)

    missing = REQUIRED_COLUMNS.union(config.target_columns) - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise DataValidationError(
            "The dataset is missing the following required columns: " f"{missing_str}"
        )

    df = df.dropna(subset=config.target_columns).copy()
    df[config.date_column] = pd.to_datetime(df[config.date_column])
    df = df.sort_values(config.date_column).reset_index(drop=True)
    return df


def _download_dataset(url: str, destination: Path) -> None:
    """Download the dataset from ``url`` into ``destination``."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with destination.open("wb") as f:
        f.write(response.content)


def _normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe that matches the internal schema."""

    if {"player", "opponent"}.issubset(df.columns):
        # Already normalized
        return df

    if {"winner_name", "loser_name"}.issubset(df.columns):
        return _transform_atp_matches(df)

    raise DataValidationError(
        "Unrecognized dataset schema. Expected columns for ATP matches or the normalized format."
    )


def _transform_atp_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Transform Jeff Sackmann ATP dataset rows into the normalized format."""

    records: List[dict] = []
    for _, row in df.iterrows():
        raw_date = row.get("tourney_date")
        if pd.isna(raw_date):
            tourney_date = pd.NaT
        else:
            tourney_date = pd.to_datetime(str(int(raw_date)), format="%Y%m%d", errors="coerce")
        for perspective, opponent in (("winner", "loser"), ("loser", "winner")):
            stat_prefix = "w" if perspective == "winner" else "l"

            record = {
                "date": tourney_date,
                "tournament": row.get("tourney_name"),
                "surface": row.get("surface"),
                "round": row.get("round"),
                "player": row.get(f"{perspective}_name"),
                "opponent": row.get(f"{opponent}_name"),
                "player_rank": row.get(f"{perspective}_rank"),
                "opponent_rank": row.get(f"{opponent}_rank"),
                "player_age": row.get(f"{perspective}_age"),
                "opponent_age": row.get(f"{opponent}_age"),
                "player_height_cm": row.get(f"{perspective}_ht"),
                "opponent_height_cm": row.get(f"{opponent}_ht"),
                "player_hand": row.get(f"{perspective}_hand"),
                "opponent_hand": row.get(f"{opponent}_hand"),
                "best_of": row.get("best_of"),
                "match_duration_minutes": row.get("minutes"),
                "player_aces": row.get(f"{stat_prefix}_ace"),
                "player_double_faults": row.get(f"{stat_prefix}_df"),
                "player_first_serve_points_won": row.get(f"{stat_prefix}_1stWon"),
                "player_second_serve_points_won": row.get(f"{stat_prefix}_2ndWon"),
                "player_break_points_saved": row.get(f"{stat_prefix}_bpSaved"),
            }

            records.append(record)

    normalized = pd.DataFrame.from_records(records)
    normalized = normalized.dropna(subset=["date", "player", "opponent"])
    return normalized


def split_train_test(
    df: pd.DataFrame, config: DataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the feature and target frames from a dataframe."""

    targets = list(target_columns)
    features = df.drop(columns=targets)
    target_df = df[targets].copy()
    return features, target_df
