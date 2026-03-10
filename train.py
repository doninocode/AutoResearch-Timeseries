"""
Fine-tune Granite TSPulse for 3-day futures move classification.

This is the mutable experiment surface for the autoresearch loop.
"""

from __future__ import annotations

import math
import time
import warnings
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.autoresearch_futures.data import FuturesWindowDataset, load_prepared_dataset
from src.autoresearch_futures.eval import evaluate_predictions, prefix_metrics
from src.autoresearch_futures.settings import (
    CLASS_NAMES,
    FORECAST_HORIZON_DAYS,
    PRIMARY_METRIC,
    TIME_BUDGET_SECONDS,
)
from tsfm_public.models.tspulse import TSPulseConfig, TSPulseForClassification

# ---------------------------------------------------------------------------
# Hyperparameters: this is the file the agent should modify.
# ---------------------------------------------------------------------------

MODEL_ID = "ibm-granite/granite-timeseries-tspulse-r1"
MODEL_REVISION = "tspulse-block-dualhead-512-p16-r1"

CONTEXT_LENGTH = 512
PATCH_LENGTH = 16
MASK_RATIO = 0.20
FREEZE_BACKBONE = False

DEVICE_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.10
WARMDOWN_RATIO = 0.35
FINAL_LR_FRAC = 0.20
LOG_EVERY_STEPS = 10
NUM_WORKERS = 0

TRADING_COST_BPS = 1.0

warnings.filterwarnings(
    "ignore",
    message="An output with one or more elements was resized since it had shape",
)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if key == "past_observed_mask":
            moved[key] = value.to(device=device, dtype=torch.float32)
        elif value.dtype.is_floating_point:
            moved[key] = value.to(device=device, dtype=torch.float32)
        else:
            moved[key] = value.to(device=device)
    return moved


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def get_lr_multiplier(progress: float) -> float:
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown + (1.0 - cooldown) * FINAL_LR_FRAC


def set_learning_rate(optimizer: torch.optim.Optimizer, multiplier: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * multiplier


def build_model(num_input_channels: int, device: torch.device) -> TSPulseForClassification:
    config = TSPulseConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    config.context_length = CONTEXT_LENGTH
    config.patch_length = PATCH_LENGTH
    config.patch_stride = PATCH_LENGTH
    config.num_input_channels = num_input_channels
    config.num_targets = len(CLASS_NAMES)
    config.loss = "cross_entropy"
    config.mask_ratio = MASK_RATIO
    config.data_actual_context_length = None

    model = TSPulseForClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        config=config,
        ignore_mismatched_sizes=True,
    )

    if FREEZE_BACKBONE:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "head" in name

    return model.to(device)


def build_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_split(
    model: TSPulseForClassification,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    model.eval()
    logits_list = []
    labels_list = []
    forward_returns = []
    symbol_indices = []
    position_indices = []
    losses = []

    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            outputs = model(
                past_values=moved["past_values"],
                past_observed_mask=moved["past_observed_mask"],
                return_loss=False,
            )
            logits = outputs.prediction_outputs
            loss = F.cross_entropy(logits, moved["target_values"])

            losses.append(float(loss.item()))
            logits_list.append(logits.cpu().numpy())
            labels_list.append(batch["target_values"].numpy())
            forward_returns.append(batch["forward_return"].numpy())
            symbol_indices.append(batch["symbol_index"].numpy())
            position_indices.append(batch["position_index"].numpy())

    logits_np = np.concatenate(logits_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)
    forward_np = np.concatenate(forward_returns, axis=0)
    symbols_np = np.concatenate(symbol_indices, axis=0)
    positions_np = np.concatenate(position_indices, axis=0)

    metrics = evaluate_predictions(
        logits=logits_np,
        labels=labels_np,
        forward_returns=forward_np,
        symbol_indices=symbols_np,
        position_indices=positions_np,
        horizon_days=FORECAST_HORIZON_DAYS,
        trading_cost_bps=TRADING_COST_BPS,
    )
    return metrics, float(np.mean(losses))


def main() -> None:
    t_start = time.time()
    torch.manual_seed(42)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = choose_device()
    frame, metadata = load_prepared_dataset()

    train_dataset = FuturesWindowDataset(frame, metadata.feature_columns, split="train", context_length=CONTEXT_LENGTH)
    val_dataset = FuturesWindowDataset(frame, metadata.feature_columns, split="val", context_length=CONTEXT_LENGTH)
    test_dataset = FuturesWindowDataset(frame, metadata.feature_columns, split="test", context_length=CONTEXT_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=DEVICE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=DEVICE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=DEVICE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(num_input_channels=len(metadata.feature_columns), device=device)
    num_params = count_parameters(model)
    class_weights = build_class_weights(
        frame.loc[frame["split"] == "train", "label"].to_numpy(dtype=np.int64),
        device=device,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = LEARNING_RATE

    print(f"Device: {device}")
    print(f"Time budget: {TIME_BUDGET_SECONDS}s")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")
    print(f"Features ({len(metadata.feature_columns)}): {', '.join(metadata.feature_columns)}")

    train_iter = cycle(train_loader)
    smooth_loss = 0.0
    step = 0

    while True:
        step_start = time.time()
        progress = min((step_start - t_start) / TIME_BUDGET_SECONDS, 1.0)
        set_learning_rate(optimizer, get_lr_multiplier(progress))
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        for _ in range(GRAD_ACCUM_STEPS):
            batch = move_batch(next(train_iter), device)
            outputs = model(
                past_values=batch["past_values"],
                past_observed_mask=batch["past_observed_mask"],
                return_loss=False,
            )
            logits = outputs.prediction_outputs
            loss = F.cross_entropy(logits, batch["target_values"], weight=class_weights)
            (loss / GRAD_ACCUM_STEPS).backward()
            total_loss += float(loss.item())

        if MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        smooth_loss = 0.9 * smooth_loss + 0.1 * total_loss
        debiased_loss = smooth_loss / (1.0 - 0.9 ** (step + 1))

        step_seconds = time.time() - step_start
        if step % LOG_EVERY_STEPS == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            remaining = max(0.0, TIME_BUDGET_SECONDS - (time.time() - t_start))
            print(
                f"step {step:05d} | loss {debiased_loss:.4f} | lr {current_lr:.6g} | "
                f"dt {step_seconds:.2f}s | remaining {remaining:.0f}s",
                flush=True,
            )

        step += 1
        if time.time() - t_start >= TIME_BUDGET_SECONDS:
            break

    t_train = time.time()
    val_metrics, val_loss = evaluate_split(model, val_loader, device=device)
    test_metrics, test_loss = evaluate_split(model, test_loader, device=device)
    summary = {}
    summary.update(prefix_metrics("val", val_metrics))
    summary.update(prefix_metrics("test", test_metrics))

    print("---")
    print(f"val_sharpe:       {summary['val_sharpe']:.6f}")
    print(f"val_macro_f1:     {summary['val_macro_f1']:.6f}")
    print(f"val_bal_acc:      {summary['val_balanced_accuracy']:.6f}")
    print(f"test_sharpe:      {summary['test_sharpe']:.6f}")
    print(f"test_macro_f1:    {summary['test_macro_f1']:.6f}")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"test_loss:        {test_loss:.6f}")
    print(f"training_seconds: {t_train - t_start:.1f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.2f}")
    print(f"context_length:   {CONTEXT_LENGTH}")
    print(f"num_features:     {len(metadata.feature_columns)}")
    print(f"primary_metric:   {PRIMARY_METRIC}")
    print(f"device:           {device}")


if __name__ == "__main__":
    main()
