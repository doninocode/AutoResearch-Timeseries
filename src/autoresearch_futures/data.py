from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .settings import (
    CLASS_NAMES,
    DEFAULT_DEMO_DATA_PATH,
    FORECAST_HORIZON_DAYS,
    METADATA_PATH,
    OPTIONAL_COLUMNS,
    PREPARED_DATA_PATH,
    REQUIRED_COLUMNS,
)


@dataclass
class PreparedMetadata:
    input_path: str
    feature_columns: list[str]
    symbols: list[str]
    rows: int
    split_counts: dict[str, int]
    label_counts: dict[str, int]
    train_end: str
    val_end: str
    horizon_days: int
    move_threshold_sigma: float
    train_fraction: float
    val_fraction: float

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> "PreparedMetadata":
        return cls(**json.loads(path.read_text()))


def generate_demo_market_data(
    output_path: Path = DEFAULT_DEMO_DATA_PATH,
    periods: int = 900,
    seed: int = 7,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=periods)
    specs = [
        ("ES", 4600.0, 0.00020, 0.010),
        ("NQ", 16000.0, 0.00035, 0.013),
        ("CL", 72.0, 0.00010, 0.018),
    ]

    frames = []
    shared_shock = rng.normal(0.0, 0.004, size=len(dates))
    for symbol, start_price, drift, noise_scale in specs:
        regime = np.sin(np.linspace(0, 8 * math.pi, len(dates))) * 0.0015
        idiosyncratic = rng.normal(0.0, noise_scale, size=len(dates))
        log_returns = drift + regime + 0.5 * shared_shock + idiosyncratic
        close = start_price * np.exp(np.cumsum(log_returns))
        open_ = np.concatenate(([close[0]], close[:-1])) * np.exp(rng.normal(0.0, 0.002, size=len(dates)))
        intraday_spread = np.abs(rng.normal(0.0, noise_scale * 0.6, size=len(dates)))
        high = np.maximum(open_, close) * (1.0 + intraday_spread)
        low = np.minimum(open_, close) * (1.0 - intraday_spread)
        volume = rng.lognormal(mean=12.0, sigma=0.35, size=len(dates))
        open_interest = start_price * 100 + np.cumsum(rng.normal(0.0, 45.0, size=len(dates)))

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "open_interest": np.maximum(open_interest, 1.0),
                }
            )
        )

    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(output_path, index=False)
    return output_path


def load_market_frame(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing raw market data at {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(input_path, parse_dates=["timestamp"])
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(input_path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    else:
        raise ValueError(f"Unsupported input format {suffix!r}. Use .csv or .parquet")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    keep_columns = list(REQUIRED_COLUMNS) + [column for column in OPTIONAL_COLUMNS if column in frame.columns]
    frame = frame[keep_columns].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["symbol", "timestamp"], ignore_index=True)
    return frame


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    std = std.replace(0.0, np.nan)
    return (series - mean) / std


def _engineer_group(group: pd.DataFrame, horizon_days: int, move_threshold_sigma: float) -> pd.DataFrame:
    group = group.sort_values("timestamp").copy()
    close = group["close"].clip(lower=1e-6)
    log_close = np.log(close)

    group["ret_1d"] = log_close.diff(1)
    group["ret_5d"] = log_close.diff(5)
    group["ret_10d"] = log_close.diff(10)
    group["vol_5d"] = group["ret_1d"].rolling(5, min_periods=5).std()
    group["vol_20d"] = group["ret_1d"].rolling(20, min_periods=20).std()
    group["vol_ratio"] = group["vol_5d"] / group["vol_20d"].replace(0.0, np.nan)
    group["trend_z20"] = _rolling_zscore(close, 20)

    if {"open", "high", "low"}.issubset(group.columns):
        open_ = group["open"].clip(lower=1e-6)
        group["close_to_open"] = np.log(close / open_)
        group["range_frac"] = (group["high"] - group["low"]) / close
        group["range_frac_5d"] = group["range_frac"].rolling(5, min_periods=5).mean()

    if "volume" in group.columns:
        log_volume = np.log1p(group["volume"].clip(lower=0.0))
        group["volume_z20"] = _rolling_zscore(log_volume, 20)

    if "open_interest" in group.columns:
        log_oi = np.log1p(group["open_interest"].clip(lower=0.0))
        group["open_interest_z20"] = _rolling_zscore(log_oi, 20)

    group["future_timestamp"] = group["timestamp"].shift(-horizon_days)
    group["future_log_return"] = log_close.shift(-horizon_days) - log_close
    scale = (group["vol_20d"] * math.sqrt(horizon_days)).clip(lower=1e-4)
    group["target_score"] = group["future_log_return"] / scale
    group["label"] = np.select(
        [group["target_score"] <= -move_threshold_sigma, group["target_score"] >= move_threshold_sigma],
        [0, 2],
        default=1,
    ).astype(np.int64)
    return group


def _split_by_time(
    frame: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    unique_dates = np.sort(frame["timestamp"].drop_duplicates().to_numpy())
    if len(unique_dates) < 30:
        raise ValueError("Need at least 30 timestamps to build train/val/test splits.")

    train_cut = max(1, int(len(unique_dates) * train_fraction))
    val_cut = max(train_cut + 1, int(len(unique_dates) * (train_fraction + val_fraction)))
    if val_cut >= len(unique_dates):
        val_cut = len(unique_dates) - 1

    train_end = pd.Timestamp(unique_dates[train_cut - 1])
    val_end = pd.Timestamp(unique_dates[val_cut - 1])

    train_mask = (frame["timestamp"] <= train_end) & (frame["future_timestamp"] <= train_end)
    val_mask = (
        (frame["timestamp"] > train_end)
        & (frame["timestamp"] <= val_end)
        & (frame["future_timestamp"] <= val_end)
    )
    test_mask = frame["timestamp"] > val_end

    split_frame = frame.copy()
    split_frame["split"] = "drop"
    split_frame.loc[train_mask, "split"] = "train"
    split_frame.loc[val_mask, "split"] = "val"
    split_frame.loc[test_mask, "split"] = "test"
    split_frame = split_frame[split_frame["split"] != "drop"].copy()
    split_frame["position_index"] = split_frame.groupby(["symbol", "split"]).cumcount()
    return split_frame, train_end, val_end


def prepare_market_data(
    input_path: Path,
    output_path: Path = PREPARED_DATA_PATH,
    metadata_path: Path = METADATA_PATH,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    move_threshold_sigma: float = 0.50,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> PreparedMetadata:
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction and val_fraction must be positive and leave room for a test split.")

    raw = load_market_frame(input_path)
    enriched_groups = []
    for _, group in raw.groupby("symbol", sort=False):
        enriched_groups.append(
            _engineer_group(
                group,
                horizon_days=horizon_days,
                move_threshold_sigma=move_threshold_sigma,
            )
        )
    enriched = pd.concat(enriched_groups, ignore_index=True)

    candidate_features = [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "vol_5d",
        "vol_20d",
        "vol_ratio",
        "trend_z20",
        "close_to_open",
        "range_frac",
        "range_frac_5d",
        "volume_z20",
        "open_interest_z20",
    ]
    feature_columns = [column for column in candidate_features if column in enriched.columns]
    enriched = enriched.dropna(subset=feature_columns + ["future_log_return", "future_timestamp"]).copy()
    enriched, train_end, val_end = _split_by_time(enriched, train_fraction=train_fraction, val_fraction=val_fraction)
    enriched["symbol_index"] = pd.Categorical(enriched["symbol"], categories=sorted(enriched["symbol"].unique())).codes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False)

    metadata = PreparedMetadata(
        input_path=str(input_path),
        feature_columns=feature_columns,
        symbols=sorted(enriched["symbol"].unique().tolist()),
        rows=len(enriched),
        split_counts={key: int(value) for key, value in enriched["split"].value_counts().sort_index().items()},
        label_counts={
            CLASS_NAMES[int(key)]: int(value)
            for key, value in enriched["label"].value_counts().sort_index().items()
        },
        train_end=str(train_end.date()),
        val_end=str(val_end.date()),
        horizon_days=horizon_days,
        move_threshold_sigma=move_threshold_sigma,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    metadata.to_json(metadata_path)
    return metadata


def load_prepared_dataset(
    data_path: Path = PREPARED_DATA_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple[pd.DataFrame, PreparedMetadata]:
    if not data_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Prepared dataset not found. Run `uv run prepare.py` first.")
    frame = pd.read_parquet(data_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["future_timestamp"] = pd.to_datetime(frame["future_timestamp"])
    metadata = PreparedMetadata.from_json(metadata_path)
    return frame, metadata


class FuturesWindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        split: str,
        context_length: int,
        stride: int = 1,
    ) -> None:
        self.context_length = context_length
        self.feature_columns = feature_columns
        self.samples: list[tuple[str, int]] = []
        self.groups: dict[str, dict[str, np.ndarray]] = {}

        split_frame = frame[frame["split"] == split].copy()
        if split_frame.empty:
            raise ValueError(f"No rows available for split={split!r}.")

        for symbol, group in split_frame.groupby("symbol", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            payload = {
                "features": group[feature_columns].to_numpy(dtype=np.float32),
                "labels": group["label"].to_numpy(dtype=np.int64),
                "future_returns": group["future_log_return"].to_numpy(dtype=np.float32),
                "symbol_index": group["symbol_index"].to_numpy(dtype=np.int64),
                "position_index": group["position_index"].to_numpy(dtype=np.int64),
                "timestamp_ns": group["timestamp"].astype("int64").to_numpy(),
            }
            self.groups[symbol] = payload
            for index in range(0, len(group), stride):
                self.samples.append((symbol, index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        symbol, end_index = self.samples[item]
        payload = self.groups[symbol]
        feature_block = payload["features"]

        start_index = max(0, end_index - self.context_length + 1)
        history = feature_block[start_index : end_index + 1]
        pad_rows = self.context_length - len(history)

        past_values = np.zeros((self.context_length, history.shape[1]), dtype=np.float32)
        observed_mask = np.zeros_like(past_values, dtype=np.bool_)
        past_values[pad_rows:] = history
        observed_mask[pad_rows:] = True

        return {
            "past_values": torch.from_numpy(past_values),
            "past_observed_mask": torch.from_numpy(observed_mask),
            "target_values": torch.tensor(payload["labels"][end_index], dtype=torch.long),
            "forward_return": torch.tensor(payload["future_returns"][end_index], dtype=torch.float32),
            "symbol_index": torch.tensor(payload["symbol_index"][end_index], dtype=torch.long),
            "position_index": torch.tensor(payload["position_index"][end_index], dtype=torch.long),
            "timestamp_ns": torch.tensor(payload["timestamp_ns"][end_index], dtype=torch.long),
        }
