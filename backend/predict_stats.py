"""CLI utility for running stat predictions using a trained model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

from stat_predictor.predict import load_model, predict_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained model .joblib file",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to the metadata.json file generated during training",
    )
    parser.add_argument(
        "--records",
        type=Path,
        required=True,
        help="JSON file describing the matches to score",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to store the predictions as JSON",
    )
    return parser.parse_args()


def load_records(path: Path) -> Iterable[Mapping[str, object]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of record objects")
    return data


def main() -> None:
    args = parse_args()
    with args.metadata.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    pipeline = load_model(args.model)
    predictions = predict_stats(
        pipeline,
        load_records(args.records),
        target_columns=metadata["target_columns"],
    )

    if args.output:
        predictions.to_json(args.output, orient="records", indent=2)
    else:
        print(predictions.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
