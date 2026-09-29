"""Metrics for Phase 3 interference prediction quality gates."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


def _as_vector(values: np.ndarray, *, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be a vector with shape (N,) or (N, 1).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain finite values only.")
    return array


def normalized_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE normalized by mean absolute true value."""
    truth = _as_vector(y_true, field_name="y_true")
    pred = _as_vector(y_pred, field_name="y_pred")
    if truth.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    denominator = float(np.mean(np.abs(truth)))
    if denominator <= 0:
        raise ValueError("Cannot compute normalized MAE when mean(|y_true|) <= 0.")
    mae = float(np.mean(np.abs(truth - pred)))
    return mae / denominator


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    truth = _as_vector(y_true, field_name="y_true")
    pred = _as_vector(y_pred, field_name="y_pred")
    if truth.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    mse = float(np.mean((truth - pred) ** 2))
    return sqrt(mse)


def within_tolerance_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tolerance_ratio: float = 0.10,
    epsilon: float = 1e-12,
) -> float:
    """Fraction of predictions with relative absolute error <= tolerance."""
    if tolerance_ratio <= 0:
        raise ValueError("tolerance_ratio must be > 0.")
    truth = _as_vector(y_true, field_name="y_true")
    pred = _as_vector(y_pred, field_name="y_pred")
    if truth.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    relative_error = np.abs(pred - truth) / np.maximum(np.abs(truth), epsilon)
    return float(np.mean(relative_error <= tolerance_ratio))


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    """Canonical Phase 3 metrics for prediction quality gates."""

    within_10pct_accuracy: float
    normalized_mae: float
    rmse: float

    def __post_init__(self) -> None:
        for field_name in ("within_10pct_accuracy", "normalized_mae", "rmse"):
            value = getattr(self, field_name)
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
        if not (0.0 <= self.within_10pct_accuracy <= 1.0):
            raise ValueError("within_10pct_accuracy must be in [0, 1].")
        if self.normalized_mae < 0:
            raise ValueError("normalized_mae must be >= 0.")
        if self.rmse < 0:
            raise ValueError("rmse must be >= 0.")


def evaluate_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> PredictionMetrics:
    """Compute all prototype metrics in one place."""
    return PredictionMetrics(
        within_10pct_accuracy=within_tolerance_accuracy(y_true, y_pred, tolerance_ratio=0.10),
        normalized_mae=normalized_mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
    )
