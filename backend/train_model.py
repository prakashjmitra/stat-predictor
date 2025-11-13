"""Command line entry point for training the tennis stat predictor."""
from __future__ import annotations

import argparse
from pathlib import Path

from stat_predictor.config import TrainingConfig, default_training_config
from stat_predictor.training import TrainingResult, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the CSV file containing historical match data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory where the trained model and metrics will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_training_config(output_dir=args.output)
    if args.data is not None:
        config.data.data_path = args.data
    result: TrainingResult = train_and_evaluate(config)
    for target, values in result.metrics.items():
        print(f"{target} - MAE: {values['mae']:.2f} | RMSE: {values['rmse']:.2f}")

    print()
    print("Artifacts saved to:")
    print(f"  Model: {result.model_path}")
    print(f"  Metrics: {result.metrics_path}")
    print(f"  Metadata: {result.metadata_path}")
    print(f"  Predictions: {result.predictions_path}")


if __name__ == "__main__":
    main()
