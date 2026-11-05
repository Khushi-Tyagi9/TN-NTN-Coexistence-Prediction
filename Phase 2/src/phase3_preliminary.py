"""Exploratory LSTM run scored on the VALIDATION split only (not the final protocol).

The test split is reserved for the final protocol, so nothing here reads it: the caller passes train and
validation arrays only. Validation is also used for early stopping / best-epoch choice, so the validation
numbers are mildly optimistic and are not gate results.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .phase3_metrics import DbMetrics, evaluate_db_metrics, fit_ar_baseline
from .phase3_prediction import (
    LSTMTrainingConfig,
    _build_lstm_model,
    _fit_model,
    _fit_normalization_arrays,
    _make_train_val_loaders,
    _predict,
    _require_torch,
)
from .reproducibility import seed_everything

# Repo-default config, single seed, short fixed budget (fixed before the run; not a search).
PRELIMINARY_CONFIG = LSTMTrainingConfig(
    hidden_size=96,
    num_layers=1,
    learning_rate=1e-3,
    batch_size=64,
    max_epochs=40,
    early_stopping_patience=6,
    device="cpu",
    seed=42,
)


def train_validation_only(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: LSTMTrainingConfig = PRELIMINARY_CONFIG,
    *,
    epoch_callback: Callable[[int, float, float], None] | None = None,
) -> dict[str, Any]:
    """Train on train, score on validation (with all baselines). The test split is not an argument."""
    started = time.perf_counter()
    torch, _ = _require_torch()
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)
    stats = _fit_normalization_arrays(x_train, y_train)
    x_train_n, y_train_n = stats.transform_features(x_train), stats.transform_targets(y_train)
    x_val_n, y_val_n = stats.transform_features(x_val), stats.transform_targets(y_val)

    model = _build_lstm_model(config)
    train_loader, val_loader = _make_train_val_loaders(x_train_n, y_train_n, x_val_n, y_val_n, config.batch_size, config.seed)
    model, best_epoch, best_val_loss, history = _fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=config.max_epochs,
        early_stopping_patience=config.early_stopping_patience,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        device=device,
        epoch_callback=epoch_callback,
    )
    val_pred = stats.inverse_targets(_predict(model, x_val_n, batch_size=config.batch_size, device=device))
    metrics: DbMetrics = evaluate_db_metrics(y_val, val_pred, x_val, ar_coefficients=fit_ar_baseline(x_train, y_train))
    return {
        "best_epoch": int(best_epoch),
        "epochs_run": len(history["val_loss"]),
        "best_val_loss_normalized_mse": float(best_val_loss),
        "history": history,
        "val_predictions": val_pred,
        "metrics": metrics,
        "seconds": time.perf_counter() - started,
    }
