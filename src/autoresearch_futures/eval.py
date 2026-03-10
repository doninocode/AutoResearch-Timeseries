from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .settings import CLASS_TO_SIGNAL


def evaluate_predictions(
    logits: np.ndarray,
    labels: np.ndarray,
    forward_returns: np.ndarray,
    symbol_indices: np.ndarray,
    position_indices: np.ndarray,
    horizon_days: int,
    trading_cost_bps: float,
) -> dict[str, float]:
    predictions = logits.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }

    frame = pd.DataFrame(
        {
            "prediction": predictions,
            "forward_return": forward_returns,
            "symbol_index": symbol_indices,
            "position_index": position_indices,
        }
    ).sort_values(["symbol_index", "position_index"])
    frame = frame[frame["position_index"] % horizon_days == 0].copy()
    frame["signal"] = frame["prediction"].map(CLASS_TO_SIGNAL).astype(float)
    frame["prev_signal"] = frame.groupby("symbol_index")["signal"].shift(1).fillna(0.0)
    frame["turnover"] = (frame["signal"] - frame["prev_signal"]).abs()
    frame["cost"] = frame["turnover"] * (trading_cost_bps / 10_000.0)
    frame["strategy_return"] = frame["signal"] * frame["forward_return"] - frame["cost"]

    pnl = frame["strategy_return"].to_numpy(dtype=np.float64)
    pnl_std = float(np.std(pnl))
    sharpe = 0.0 if pnl_std == 0.0 else float(np.mean(pnl) / pnl_std * math.sqrt(252 / horizon_days))

    active = frame["signal"] != 0.0
    if active.any():
        hits = np.sign(frame.loc[active, "signal"]) == np.sign(frame.loc[active, "forward_return"])
        hit_rate = float(hits.mean())
    else:
        hit_rate = 0.0

    metrics.update(
        {
            "sharpe": sharpe,
            "avg_return_bps": float(np.mean(pnl) * 10_000.0),
            "coverage": float(active.mean()),
            "hit_rate": hit_rate,
        }
    )
    return metrics


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}

