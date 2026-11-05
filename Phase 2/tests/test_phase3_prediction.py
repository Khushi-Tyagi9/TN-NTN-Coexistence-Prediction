"""Tests for Phase 3 prediction-stage orchestration gates."""

import numpy as np

from src.phase3_data import Phase3DatasetSplit
from src.phase3_metrics import PredictionMetrics
from src.phase3_prediction import PredictionModelResult, run_prediction_stage


def _dummy_split() -> Phase3DatasetSplit:
    x = np.ones((12, 10, 3), dtype=np.float32)
    y = np.linspace(1.0, 2.2, 12, dtype=np.float32)
    return Phase3DatasetSplit(
        X_train=x[:8],
        y_train=y[:8],
        X_val=x[8:10],
        y_val=y[8:10],
        X_test=x[10:],
        y_test=y[10:],
    )


def _result(name: str, meets_target: bool) -> PredictionModelResult:
    metrics = PredictionMetrics(
        within_10pct_accuracy=0.85 if meets_target else 0.70,
        normalized_mae=0.15,
        rmse=0.1,
    )
    return PredictionModelResult(
        model_name=name,
        best_epoch=3,
        best_validation_loss=0.02,
        validation_metrics=metrics,
        test_metrics=metrics,
        target_accuracy=0.8,
        meets_target=meets_target,
        checkpoint_path=None,
        last_test_prediction=1.1,
        last_test_target=1.2,
        test_target_min=0.9,
        test_target_max=1.3,
    )


def test_transformer_not_executed_when_lstm_meets_target(monkeypatch) -> None:
    monkeypatch.setattr("src.phase3_prediction.train_lstm_forecaster", lambda *args, **kwargs: _result("lstm", True))
    monkeypatch.setattr("src.phase3_prediction.train_transformer_forecaster", lambda *args, **kwargs: _result("transformer", True))

    outcome = run_prediction_stage(_dummy_split(), run_transformer_if_eligible=True)

    assert not outcome.transformer_executed
    assert outcome.transformer is None


def test_transformer_fallback_runs_when_lstm_misses_target(monkeypatch) -> None:
    monkeypatch.setattr("src.phase3_prediction.train_lstm_forecaster", lambda *args, **kwargs: _result("lstm", False))
    monkeypatch.setattr(
        "src.phase3_prediction.train_transformer_forecaster",
        lambda *args, **kwargs: _result("transformer", True),
    )

    outcome = run_prediction_stage(_dummy_split(), run_transformer_if_eligible=True)

    assert outcome.transformer_executed
    assert outcome.transformer is not None
    assert outcome.transformer.model_name == "transformer"


# --- audit M2: normalization + legacy-gate helpers (no torch needed) ---------------------------

import pytest  # noqa: E402

from src.phase3_prediction import (  # noqa: E402
    NormalizationStats,
    _apply_normalization,
    _fit_normalization,
    meets_prediction_target,
)


def _toy_split() -> Phase3DatasetSplit:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=[-80.0, -73.0, 7.0], scale=[6.0, 1.0, 5.0], size=(60, 10, 3)).astype(np.float32)
    y = rng.normal(-80.0, 6.0, size=60).astype(np.float32)
    return Phase3DatasetSplit(x[:36], y[:36], x[36:48], y[36:48], x[48:], y[48:])


def test_normalization_standardises_train_and_round_trips_targets() -> None:
    split = _toy_split()
    stats = _fit_normalization(split)
    normalized = _apply_normalization(split, stats)

    assert stats.x_mean.shape == (1, 1, 3) and stats.x_std.shape == (1, 1, 3)
    assert np.allclose(normalized.X_train.mean(axis=(0, 1)), 0.0, atol=1e-4)
    assert np.allclose(normalized.X_train.std(axis=(0, 1)), 1.0, atol=1e-4)
    assert np.allclose(stats.inverse_targets(stats.transform_targets(split.y_test)), split.y_test, atol=1e-3)
    # statistics come from TRAIN only: val/test are transformed with the same mean/std
    assert not np.allclose(normalized.X_test.mean(axis=(0, 1)), 0.0, atol=1e-6)


def test_normalization_handles_constant_channels() -> None:
    split = _toy_split()
    x = split.X_train.copy()
    x[:, :, 2] = 3.0  # zero-variance channel (e.g. a masked ablation channel)
    stats = _fit_normalization(
        Phase3DatasetSplit(x, split.y_train, split.X_val, split.y_val, split.X_test, split.y_test)
    )

    assert stats.x_std[0, 0, 2] == 1.0
    assert np.all(np.isfinite(stats.transform_features(split.X_test)))


def test_normalization_stats_validation() -> None:
    good = _fit_normalization(_toy_split())
    with pytest.raises(ValueError):
        NormalizationStats(x_mean=np.zeros((1, 3)), x_std=good.x_std, y_mean=0.0, y_std=1.0)
    with pytest.raises(ValueError):
        NormalizationStats(x_mean=good.x_mean, x_std=np.zeros((1, 1, 3), dtype=np.float32), y_mean=0.0, y_std=1.0)
    with pytest.raises(ValueError):
        NormalizationStats(x_mean=good.x_mean, x_std=good.x_std, y_mean=0.0, y_std=0.0)


def test_legacy_accuracy_helper_is_still_available_but_informational() -> None:
    from src.phase3_metrics import PredictionMetrics

    assert meets_prediction_target(PredictionMetrics(0.85, 0.1, 1.0), target_accuracy=0.8)
    assert not meets_prediction_target(PredictionMetrics(0.75, 0.1, 1.0), target_accuracy=0.8)
    with pytest.raises(ValueError):
        meets_prediction_target(PredictionMetrics(0.85, 0.1, 1.0), target_accuracy=0.0)
