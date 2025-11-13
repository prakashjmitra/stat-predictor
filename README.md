# Tennis Stat Predictor

This repository provides a complete, reproducible pipeline for training a machine
learning model that forecasts per-match tennis statistics such as aces, double
faults, first/second-serve points won, and break points saved for upcoming
tournaments.

## Project structure

```
backend/            Python package and command line tools for modeling
  requirements.txt  Dependency list for the backend
  train_model.py    CLI entry point that trains the model and exports predictions
  predict_stats.py  CLI entry point that generates predictions
  stat_predictor/   Core Python package with reusable components
    config.py       Configuration dataclasses and defaults
    data.py         Data loading and validation helpers
    features.py     Feature engineering utilities
    model.py        Model creation (multi-output regression pipeline)
    predict.py      Programmatic prediction helpers
    training.py     Training and evaluation orchestration

data/
  sample_matches.csv  Small synthetic dataset used in documentation and tests
```

## Getting started

1. Create and activate a virtual environment (recommended) and install the backend
   dependencies:

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Train the model. Without any additional flags the command downloads the latest
   ATP results for 2023 from Jeff Sackmann's public tennis dataset, performs
   training, evaluation, and exports a set of ready-to-use predictions for the
   most recent matches:

   ```bash
   python train_model.py
   ```

   The command creates an `artifacts/` directory (or uses the one you pass via
   `--output`) with the following files:

   * `trained_model.joblib` – the serialized scikit-learn pipeline
   * `metrics.json` – evaluation metrics (MAE and RMSE) per predicted statistic
    * `metadata.json` – helper metadata for downstream prediction jobs
    * `predictions.json` – predicted statistics for the latest evaluation matches

   To train on a different dataset, supply `--data <path-to-csv>` and optionally
   `--output <directory>`.

3. Score upcoming matches by supplying a JSON file describing the fixtures and the
   artifacts from the training step:

   ```bash
   python predict_stats.py \
       --model artifacts/trained_model.joblib \
       --metadata artifacts/metadata.json \
       --records ../data/upcoming_matches.json
   ```

   The script prints the predictions to stdout unless `--output` is provided, in
   which case the predictions are persisted to disk as JSON.

## Data requirements

The historical training CSV must contain the following columns:

* `date` – ISO formatted date of the match
* `tournament` – tournament name
* `surface` – playing surface (e.g., Hard, Clay, Grass)
* `round` – round description (e.g., Quarterfinal)
* `player`, `opponent` – competitor names
* `player_rank`, `opponent_rank` – ATP/WTA ranking at the time of the match
* `player_age`, `opponent_age` – ages in years
* `player_height_cm`, `opponent_height_cm` – heights in centimeters
* `player_hand`, `opponent_hand` – playing hand (R/L)
* `best_of` – match format (3 or 5 sets)
* `match_duration_minutes` – total match duration (use an estimated value if unknown)
* Target columns for the statistics you want to predict. By default, the model
  expects:
  * `player_aces`
  * `player_double_faults`
  * `player_first_serve_points_won`
  * `player_second_serve_points_won`
  * `player_break_points_saved`

You can customize the target and feature sets by editing
`stat_predictor/config.py` or by creating your own `TrainingConfig` instance when
using the package programmatically.

## Reproducibility

The project ships with a small synthetic dataset (`data/sample_matches.csv`) so
that the entire pipeline can be executed end-to-end without external data. Replace
this file with richer historical data to train a production-ready model or let the
default training script download the public ATP results automatically.
