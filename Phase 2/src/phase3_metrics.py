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


# ---------------------------------------------------------------------------
# dB-domain metrics and persistence-relative gate (audit C1)
#
# The targets are interference levels in dBm. A *relative* error on a negative
# dBm value is meaningless (10% of -60 dBm is 6 dB, and the tolerance grows as
# the interference gets weaker), so the gate works on absolute errors in dB and
# on skill relative to a "repeat the last observed value" baseline.
# ---------------------------------------------------------------------------

PERSISTENCE_FEATURE_INDEX = 0  # feature 0 is interference_dbm (see timeseries.FEATURE_NAMES)
DEFAULT_MIN_SKILL_VS_PERSISTENCE = 0.15  # fixed a priori; must not be tuned after seeing results
_ZERO_BASELINE_EPS = 1e-12


def _paired_vectors(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = _as_vector(y_true, field_name="y_true")
    pred = _as_vector(y_pred, field_name="y_pred")
    if truth.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if truth.shape[0] == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")
    return truth, pred


def mae_db(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in dB (inputs are dB-domain values)."""
    truth, pred = _paired_vectors(y_true, y_pred)
    return float(np.mean(np.abs(truth - pred)))


def within_db_share(y_true: np.ndarray, y_pred: np.ndarray, *, tolerance_db: float) -> float:
    """Fraction of predictions with absolute error <= tolerance_db."""
    if not np.isfinite(tolerance_db) or tolerance_db <= 0:
        raise ValueError("tolerance_db must be finite and > 0.")
    truth, pred = _paired_vectors(y_true, y_pred)
    return float(np.mean(np.abs(truth - pred) <= tolerance_db))


def persistence_prediction(x: np.ndarray, *, feature_index: int = PERSISTENCE_FEATURE_INDEX) -> np.ndarray:
    """Baseline: predict the next value as the last observed value in the window."""
    windows = np.asarray(x, dtype=np.float64)
    if windows.ndim != 3:
        raise ValueError("x must have shape (N, lookback, features).")
    if not (0 <= feature_index < windows.shape[2]):
        raise ValueError("feature_index is out of range.")
    return windows[:, -1, feature_index]


def skill_vs_baseline(model_mae_db: float, baseline_mae_db: float) -> float:
    """1 - MAE_model / MAE_baseline (positive = better than baseline).

    A degenerate zero-error baseline yields 0.0 so the gate cannot be passed by accident.
    """
    if model_mae_db < 0 or baseline_mae_db < 0:
        raise ValueError("MAE values must be >= 0.")
    if baseline_mae_db <= _ZERO_BASELINE_EPS:
        return 0.0
    return float(1.0 - model_mae_db / baseline_mae_db)


@dataclass(frozen=True, slots=True)
class DbMetrics:
    """dB-domain quality metrics plus the persistence baseline they are judged against."""

    mae_db: float
    rmse_db: float
    within_1db: float
    within_3db: float
    persistence_mae_db: float
    persistence_rmse_db: float
    skill_vs_persistence: float

    def __post_init__(self) -> None:
        for name in (
            "mae_db",
            "rmse_db",
            "within_1db",
            "within_3db",
            "persistence_mae_db",
            "persistence_rmse_db",
            "skill_vs_persistence",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if min(self.mae_db, self.rmse_db, self.persistence_mae_db, self.persistence_rmse_db) < 0:
            raise ValueError("Error metrics must be >= 0.")
        for name in ("within_1db", "within_3db"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1].")


def evaluate_db_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x: np.ndarray,
    *,
    persistence_feature_index: int = PERSISTENCE_FEATURE_INDEX,
) -> DbMetrics:
    """Score ``y_pred`` in dB and against the persistence baseline built from windows ``x``."""
    truth, pred = _paired_vectors(y_true, y_pred)
    baseline = persistence_prediction(x, feature_index=persistence_feature_index)
    if baseline.shape[0] != truth.shape[0]:
        raise ValueError("x must contain one window per target.")
    model_mae = mae_db(truth, pred)
    baseline_mae = mae_db(truth, baseline)
    return DbMetrics(
        mae_db=model_mae,
        rmse_db=rmse(truth, pred),
        within_1db=within_db_share(truth, pred, tolerance_db=1.0),
        within_3db=within_db_share(truth, pred, tolerance_db=3.0),
        persistence_mae_db=baseline_mae,
        persistence_rmse_db=rmse(truth, baseline),
        skill_vs_persistence=skill_vs_baseline(model_mae, baseline_mae),
    )


def meets_skill_gate(metrics: DbMetrics, *, min_skill: float = DEFAULT_MIN_SKILL_VS_PERSISTENCE) -> bool:
    """Gate: the model must beat persistence MAE by at least ``min_skill`` (a fraction)."""
    if not (0.0 <= min_skill < 1.0):
        raise ValueError("min_skill must be in [0, 1).")
    return metrics.skill_vs_persistence >= min_skill
