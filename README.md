# AutoResearch Timeseries

Automated time-series research harness for trading experiments with open-source forecasting and classification models.

This repo is directly inspired by Andrej Karpathy's original [`autoresearch`](https://github.com/karpathy/autoresearch) project and adapts that edit/run/evaluate/keep-or-revert loop from short LLM pretraining runs to market prediction. The current scaffold is set up for 3-trading-day direction classification on daily price data, with a fixed prep/evaluation harness and a mutable training surface for agent-driven iteration.

The core idea is simple:

- `prepare.py` is the fixed data and evaluation prep layer.
- `train.py` is the mutable experiment surface.
- `program.md` tells an agent how to run the keep/revert loop.

The initial baseline uses IBM Granite TSPulse classification via `granite-tsfm`, but the structure is meant for broader time-series research on trading datasets.

## Credit

The research loop, repo shape, and autonomous experiment pattern are adapted from Andrej Karpathy's original [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch). This repo changes the domain and model stack, but the core workflow idea is his.

## What This Scaffold Assumes

- Input data is daily market history.
- The prediction target is a 3-trading-day move bucket: `down`, `flat`, or `up`.
- Labels are volatility-scaled so the same target definition can span contracts with different natural vol levels.
- The primary experiment metric is `val_sharpe` from a simple long/flat/short policy on non-overlapping 3-day windows.

This is a starting point, not a finished trading system. The harness is for controlled iteration on features, training, and model settings.

## Quick Start

Requirements: Python 3.11+, `uv`, internet access for the first model download.

```bash
uv sync

# Default input is the checked-in TSLA Yahoo daily CSV
uv run prepare.py

# Option 2: point at your own daily market CSV or parquet
# required columns: timestamp, symbol, close
# optional columns: open, high, low, volume, open_interest
# uv run prepare.py --input data/raw/my_market_data.csv

# Run one 5-minute baseline experiment
uv run train.py
```

## Input Format

The prep step expects one row per symbol per timestamp.

Required columns:

- `timestamp`
- `symbol`
- `close`

Optional columns:

- `open`
- `high`
- `low`
- `volume`
- `open_interest`

If present, the optional columns are turned into extra model features automatically.

Current default input:

- `data/raw/tsla_daily_yahoo_2018_to_2026-03-10.csv`

## Files That Matter

- `prepare.py`: validates raw data, engineers features, creates labels, and writes a prepared parquet + metadata cache.
- `train.py`: loads the prepared cache, fine-tunes Granite TSPulse for classification, and reports validation/test metrics.
- `src/autoresearch_futures/data.py`: fixed feature engineering and window dataset logic.
- `src/autoresearch_futures/eval.py`: fixed metrics, including the non-overlapping validation Sharpe calculation.
- `scripts/log_result.py`: appends a `run.log` summary into `results.tsv`.
- `program.md`: autonomous experiment instructions.

## Output Metric

Each training run prints a summary like:

```text
---
val_sharpe:       0.423100
val_macro_f1:     0.401200
val_bal_acc:      0.487500
test_sharpe:      0.215700
training_seconds: 300.1
total_seconds:    332.4
num_steps:        412
num_params_M:     8.7
context_length:   512
num_features:     10
device:           mps
```

The autonomous loop should optimize `val_sharpe`. The other metrics are there as guardrails.

## Viability Notes

This scaffold is viable for overnight research loops if you keep the fixed evaluator honest:

- use walk-forward date splits,
- avoid changing the evaluation code inside the loop,
- treat `test_sharpe` as a holdout sanity check, not the optimization target,
- and do not confuse label quality with strategy quality.

Before putting real capital behind it, you still need instrument-specific handling for rolls, contract selection, slippage, fees, and execution.
