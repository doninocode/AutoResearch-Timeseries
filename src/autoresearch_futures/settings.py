import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".cache" / "futures_autoresearch"
PREPARED_DATA_PATH = CACHE_DIR / "prepared.parquet"
METADATA_PATH = CACHE_DIR / "metadata.json"

DEFAULT_RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "tsla_daily_yahoo_2018_to_2026-03-10.csv"
DEFAULT_DEMO_DATA_PATH = REPO_ROOT / "data" / "raw" / "demo_futures.csv"

TIME_BUDGET_SECONDS = int(os.getenv("AUTORESEARCH_TIME_BUDGET_SECONDS", "300"))
FORECAST_HORIZON_DAYS = 3
PRIMARY_METRIC = "val_sharpe"

CLASS_NAMES = ("down", "flat", "up")
CLASS_TO_SIGNAL = {0: -1.0, 1: 0.0, 2: 1.0}

REQUIRED_COLUMNS = ("timestamp", "symbol", "close")
OPTIONAL_COLUMNS = ("open", "high", "low", "volume", "open_interest")
