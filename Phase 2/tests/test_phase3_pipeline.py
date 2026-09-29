"""Tests for Phase 3 mitigation-stage gating and predictor handoff."""

import pytest

from src.phase3_metrics import PredictionMetrics
from src.phase3_pipeline import run_mitigation_track
from src.phase3_prediction import PredictionModelResult, PredictionStageResult


def _prediction_result(model_name: str, meets_target: bool, prediction: float, target: float) -> PredictionModelResult:
    metrics = PredictionMetrics(
        within_10pct_accuracy=0.82 if meets_target else 0.65,
        normalized_mae=0.2,
        rmse=0.1,
    )
    return PredictionModelResult(
        model_name=model_name,
        best_epoch=5,
        best_validation_loss=0.01,
        validation_metrics=metrics,
        test_metrics=metrics,
        target_accuracy=0.8,
        meets_target=meets_target,
        checkpoint_path=None,
        last_test_prediction=prediction,
        last_test_target=target,
        test_target_min=0.5,
        test_target_max=1.5,
    )


def test_mitigation_uses_lstm_when_lstm_meets_target() -> None:
    stage = PredictionStageResult(
        lstm=_prediction_result("lstm", True, prediction=1.1, target=1.0),
        transformer=None,
        transformer_executed=False,
    )

    mitigation = run_mitigation_track(
        prediction_result=stage,
        desired_angle_rad=0.0,
        ntn_angle_rad=0.35,
        antenna_count=64,
    )

    assert mitigation.predictor_used == "lstm"
    assert mitigation.zf_gate_passed


def test_mitigation_falls_back_to_transformer_when_lstm_fails() -> None:
    stage = PredictionStageResult(
        lstm=_prediction_result("lstm", False, prediction=1.3, target=1.1),
        transformer=_prediction_result("transformer", True, prediction=1.05, target=1.0),
        transformer_executed=True,
    )

    mitigation = run_mitigation_track(
        prediction_result=stage,
        desired_angle_rad=0.0,
        ntn_angle_rad=0.35,
        antenna_count=64,
    )

    assert mitigation.predictor_used == "transformer"


def test_mitigation_blocks_when_no_predictor_meets_target() -> None:
    stage = PredictionStageResult(
        lstm=_prediction_result("lstm", False, prediction=1.2, target=1.1),
        transformer=_prediction_result("transformer", False, prediction=1.2, target=1.1),
        transformer_executed=True,
    )

    with pytest.raises(ValueError, match="at least one prediction model meets target"):
        run_mitigation_track(
            prediction_result=stage,
            desired_angle_rad=0.0,
            ntn_angle_rad=0.35,
            antenna_count=64,
        )
