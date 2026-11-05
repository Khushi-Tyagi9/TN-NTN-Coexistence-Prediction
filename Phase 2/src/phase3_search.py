"""Validation-only hyperparameter search for the LSTM (Stage B).

Protocol as amended BEFORE any trial result existed (time limit; see docs/STAGE_B_PROTOCOL.md):

* Training budget for every run: at most 40 epochs, early-stopping patience 6 on validation loss, batch 64,
  no dropout/weight decay, gradient clip 1.0.
* Reduced grid, searched on URBAN only with seed 42: the repo default (hidden 96, lr 1e-3) plus hidden
  {64, 128} x lr {1e-3, 3e-4}, all 1 layer = 5 trials. The chosen config is reused unchanged for rural.
  (The originally declared 12-config grid with 120 epochs / patience 12 was dropped for time.)
* Selection criterion: lowest validation loss (normalised MSE); ties: smaller hidden size, fewer layers,
  larger learning rate. The functions here accept train/validation arrays only, so the test split cannot
  influence selection. After the search the grid is closed.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Callable

import numpy as np

from .phase3_metrics import mae_db, rmse
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

DEFAULT_MAX_EPOCHS = 40
DEFAULT_EARLY_STOPPING_PATIENCE = 6
SEARCH_SCENARIO = "urban"
# (hidden size, learning rate, layers); the first entry is the repo default
SEARCH_GRID: tuple[tuple[int, float, int], ...] = (
    (96, 1e-3, 1),
    (64, 1e-3, 1),
    (64, 3e-4, 1),
    (128, 1e-3, 1),
    (128, 3e-4, 1),
)
SEARCH_SELECTION_CRITERION = (
    "validation loss (normalised MSE), lowest wins; ties: smaller hidden size, fewer layers, larger learning rate"
)
ORIGINAL_GRID_NOT_RUN = "hidden {64,96,128} x lr {1e-3,3e-4} x layers {1,2} = 12 configs, 120 epochs, patience 12"
REDUCTION_NOTE = (
    "The hyperparameter search was reduced because of a time limit: 5 configs on urban only (seed 42) with a "
    "40-epoch / patience-6 budget instead of the originally declared 12-config grid (120 epochs / patience 12). "
    "The urban-selected config is reused unchanged for rural; there was no separate rural search."
)


def grid_configs(
    *,
    seed: int = 42,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    batch_size: int = 64,
    device: str = "cpu",
) -> list[LSTMTrainingConfig]:
    """The 5 configurations of the amended grid, in a fixed order (repo default first)."""
    return [
        LSTMTrainingConfig(
            hidden_size=hidden,
            num_layers=layers,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        for hidden, learning_rate, layers in SEARCH_GRID
    ]


def selection_key(trial: dict[str, Any]) -> tuple[float, int, int, float]:
    config = trial["config"]
    return (trial["best_val_loss_normalized_mse"], config["hidden_size"], config["num_layers"], -config["learning_rate"])


def select_best(trials: list[dict[str, Any]]) -> int:
    """Index of the winning trial by validation loss with the documented tie-breaks."""
    if not trials:
        raise ValueError("No trials to select from.")
    return min(range(len(trials)), key=lambda i: selection_key(trials[i]))


def run_trial(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: LSTMTrainingConfig,
) -> dict[str, Any]:
    """Train one config on train, score it on validation. The test split is not an argument."""
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
    )
    val_pred = stats.inverse_targets(_predict(model, x_val_n, batch_size=config.batch_size, device=device))
    return {
        "config": asdict(config),
        "best_epoch": int(best_epoch),
        "epochs_run": len(history["val_loss"]),
        "best_val_loss_normalized_mse": float(best_val_loss),
        "val_mae_db": mae_db(y_val, val_pred),
        "val_rmse_db": rmse(y_val, val_pred),
        "seconds": round(time.perf_counter() - started, 1),
    }


def run_validation_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    configs: list[LSTMTrainingConfig],
    *,
    on_trial: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every config; ``on_trial(index, trial)`` fires after each so progress can be logged."""
    trials: list[dict[str, Any]] = []
    for index, config in enumerate(configs):
        trial = run_trial(x_train, y_train, x_val, y_val, config)
        trials.append(trial)
        if on_trial is not None:
            on_trial(index, trial)
    return trials
