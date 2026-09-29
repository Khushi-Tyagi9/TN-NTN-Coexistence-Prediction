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
