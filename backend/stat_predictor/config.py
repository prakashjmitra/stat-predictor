"""Configuration objects for the stat predictor backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    """Settings describing how to load the historical match dataset."""

    data_path: Path
    target_columns: List[str] = field(
        default_factory=lambda: [
            "player_aces",
            "player_double_faults",
            "player_service_games_won",
            "player_return_games_won",
            "player_games_won",
        ]
    )
    date_column: str = "date"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    """Settings that control model construction and training."""

    gradient_boosting_estimators: int = 300
    gradient_boosting_learning_rate: float = 0.05
    gradient_boosting_max_depth: int = 3
    categorical_features: Optional[List[str]] = None
    numerical_features: Optional[List[str]] = None


@dataclass
class TrainingConfig:
    """High level configuration used by the training script."""

    data: DataConfig
    model: ModelConfig
    output_dir: Path

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_matches.csv"


def default_training_config(output_dir: Path | None = None) -> TrainingConfig:
    output_dir = output_dir or Path(__file__).resolve().parents[2] / "backend" / "artifacts"
    data_config = DataConfig(data_path=DEFAULT_DATA_PATH)
    model_config = ModelConfig(
        categorical_features=[
            "surface",
            "tournament",
            "round",
            "player_hand",
            "opponent_hand",
        ],
        numerical_features=[
            "player_rank",
            "opponent_rank",
            "player_age",
            "opponent_age",
            "player_height_cm",
            "opponent_height_cm",
            "best_of",
            "match_duration_minutes",
        ],
    )
    return TrainingConfig(data=data_config, model=model_config, output_dir=output_dir)
