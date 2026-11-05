"""Phase 3 end-to-end orchestration with explicit stage gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .phase3_beamforming import (
    ZFNullingResult,
    evaluate_zf_nulling,
    map_interference_power_to_ntn_angle,
)
from .phase3_data import Phase3DatasetSplit, load_phase3_dataset
from .phase3_drl import DrlEnvConfig, PPOTrainingConfig, train_ppo_mitigator
from .phase3_prediction import (
    LSTMTrainingConfig,
    PredictionStageResult,
    TransformerTrainingConfig,
    run_prediction_stage,
)


@dataclass(frozen=True, slots=True)
class Phase3Gates:
    """Quality thresholds used to move between Phase 3 stages."""

    lstm_target_accuracy: float = 0.80  # legacy within-10% metric: reported, NOT gated on
    lstm_min_skill_vs_persistence: float = 0.15  # the actual gate: MAE at least 15% below persistence
    zf_required_suppression_db: float = 15.0

    def __post_init__(self) -> None:
        if not (0 < self.lstm_target_accuracy <= 1):
            raise ValueError("lstm_target_accuracy must be in (0, 1].")
        if not (0.0 <= self.lstm_min_skill_vs_persistence < 1.0):
            raise ValueError("lstm_min_skill_vs_persistence must be in [0, 1).")


@dataclass(frozen=True, slots=True)
class MitigationStageResult:
    """Mitigation outputs after prediction stage completion."""

    zf_result: ZFNullingResult
    predictor_used: str
    estimated_ntn_angle_rad: float
    true_ntn_angle_rad: float
    predicted_interference_power: float
    true_interference_power: float
    zf_gate_passed: bool
    ppo_trained: bool


@dataclass(frozen=True, slots=True)
class Phase3RunResult:
    """Complete Phase 3 orchestration outcome."""

    prediction: PredictionStageResult
    mitigation: MitigationStageResult


def run_prediction_track(
    *,
    data_root: str | Path,
    scenario: str | None = None,
    seed: int = 42,
    gates: Phase3Gates | None = None,
    lstm_config: LSTMTrainingConfig | None = None,
    transformer_config: TransformerTrainingConfig | None = None,
    checkpoint_dir: str | Path | None = None,
    allow_window_shuffle: bool = False,
) -> tuple[Phase3DatasetSplit, PredictionStageResult]:
    """Run Track A prediction in required order: LSTM first, Transformer as fallback."""
    active_gates = Phase3Gates() if gates is None else gates
    dataset = load_phase3_dataset(
        data_root, scenario=scenario, seed=seed, allow_window_shuffle=allow_window_shuffle
    )
    prediction_result = run_prediction_stage(
        dataset,
        lstm_config=lstm_config,
        transformer_config=transformer_config,
        target_accuracy=active_gates.lstm_target_accuracy,
        min_skill=active_gates.lstm_min_skill_vs_persistence,
        run_transformer_if_eligible=True,
        checkpoint_dir=checkpoint_dir,
    )
    return dataset, prediction_result


def run_mitigation_track(
    *,
    prediction_result: PredictionStageResult,
    desired_angle_rad: float,
    ntn_angle_rad: float,
    antenna_count: int = 64,
    max_tracking_offset_rad: float = 0.12,
    gates: Phase3Gates | None = None,
    train_drl: bool = False,
    env_config: DrlEnvConfig | None = None,
    ppo_config: PPOTrainingConfig | None = None,
) -> MitigationStageResult:
    """Run Track B in required order: ZF first, optional DRL second."""
    predictor = prediction_result.lstm
    predictor_name = "lstm"
    if not predictor.meets_target:
        transformer = prediction_result.transformer
        if transformer is None or not transformer.meets_target:
            raise ValueError("Cannot start mitigation track before at least one prediction model meets target.")
        predictor = transformer
        predictor_name = "transformer"

    active_gates = Phase3Gates() if gates is None else gates
    estimated_ntn_angle = map_interference_power_to_ntn_angle(
        predictor.last_test_prediction,
        power_min=predictor.test_target_min,
        power_max=predictor.test_target_max,
        nominal_angle_rad=ntn_angle_rad,
        max_tracking_offset_rad=max_tracking_offset_rad,
    )
    true_ntn_angle = map_interference_power_to_ntn_angle(
        predictor.last_test_target,
        power_min=predictor.test_target_min,
        power_max=predictor.test_target_max,
        nominal_angle_rad=ntn_angle_rad,
        max_tracking_offset_rad=max_tracking_offset_rad,
    )
    zf_result = evaluate_zf_nulling(
        desired_angle_rad=desired_angle_rad,
        ntn_angle_rad=true_ntn_angle,
        steering_ntn_angle_rad=estimated_ntn_angle,
        antenna_count=antenna_count,
    )
    zf_gate_passed = zf_result.suppression_db >= active_gates.zf_required_suppression_db
    ppo_trained = False
    if train_drl:
        train_ppo_mitigator(
            zf_suppression_db=zf_result.suppression_db,
            zf_required_db=active_gates.zf_required_suppression_db,
            env_config=env_config,
            ppo_config=ppo_config,
        )
        ppo_trained = True

    return MitigationStageResult(
        zf_result=zf_result,
        predictor_used=predictor_name,
        estimated_ntn_angle_rad=estimated_ntn_angle,
        true_ntn_angle_rad=true_ntn_angle,
        predicted_interference_power=predictor.last_test_prediction,
        true_interference_power=predictor.last_test_target,
        zf_gate_passed=zf_gate_passed,
        ppo_trained=ppo_trained,
    )


def run_phase3_pipeline(
    *,
    data_root: str | Path,
    scenario: str | None = None,
    seed: int = 42,
    desired_angle_rad: float = 0.0,
    ntn_angle_rad: float = 0.35,
    antenna_count: int = 64,
    max_tracking_offset_rad: float = 0.12,
    gates: Phase3Gates | None = None,
    lstm_config: LSTMTrainingConfig | None = None,
    transformer_config: TransformerTrainingConfig | None = None,
    checkpoint_dir: str | Path | None = None,
    allow_window_shuffle: bool = False,
    train_drl: bool = False,
    env_config: DrlEnvConfig | None = None,
    ppo_config: PPOTrainingConfig | None = None,
) -> Phase3RunResult:
    """Execute full Phase 3 with stage gating."""
    _, prediction_result = run_prediction_track(
        data_root=data_root,
        scenario=scenario,
        seed=seed,
        gates=gates,
        lstm_config=lstm_config,
        transformer_config=transformer_config,
        checkpoint_dir=checkpoint_dir,
        allow_window_shuffle=allow_window_shuffle,
    )
    mitigation_result = run_mitigation_track(
        prediction_result=prediction_result,
        desired_angle_rad=desired_angle_rad,
        ntn_angle_rad=ntn_angle_rad,
        antenna_count=antenna_count,
        max_tracking_offset_rad=max_tracking_offset_rad,
        gates=gates,
        train_drl=train_drl,
        env_config=env_config,
        ppo_config=ppo_config,
    )
    return Phase3RunResult(prediction=prediction_result, mitigation=mitigation_result)


def _db_metrics_payload(model: Any) -> dict[str, Any] | None:
    metrics = model.test_db_metrics
    if metrics is None:
        return None
    return {
        "mae_db": metrics.mae_db,
        "rmse_db": metrics.rmse_db,
        "within_1db": metrics.within_1db,
        "within_3db": metrics.within_3db,
        "persistence_mae_db": metrics.persistence_mae_db,
        "persistence_rmse_db": metrics.persistence_rmse_db,
        "skill_vs_persistence": metrics.skill_vs_persistence,
        "min_skill_required": model.min_skill,
    }


def to_serializable_summary(result: Phase3RunResult) -> dict[str, Any]:
    """Convert orchestration outputs to JSON-safe payload."""
    transformer = result.prediction.transformer
    return {
        "prediction": {
            "lstm": {
                "best_epoch": result.prediction.lstm.best_epoch,
                "best_validation_loss": result.prediction.lstm.best_validation_loss,
                "test_within_10pct_accuracy": result.prediction.lstm.test_metrics.within_10pct_accuracy,
                "test_normalized_mae": result.prediction.lstm.test_metrics.normalized_mae,
                "test_rmse": result.prediction.lstm.test_metrics.rmse,
                "target_accuracy": result.prediction.lstm.target_accuracy,
                "meets_target": result.prediction.lstm.meets_target,
                "test_db_metrics": _db_metrics_payload(result.prediction.lstm),
                "checkpoint_path": result.prediction.lstm.checkpoint_path,
                "last_test_prediction": result.prediction.lstm.last_test_prediction,
                "last_test_target": result.prediction.lstm.last_test_target,
            },
            "transformer_executed": result.prediction.transformer_executed,
            "transformer": None
            if transformer is None
            else {
                "best_epoch": transformer.best_epoch,
                "best_validation_loss": transformer.best_validation_loss,
                "test_within_10pct_accuracy": transformer.test_metrics.within_10pct_accuracy,
                "test_normalized_mae": transformer.test_metrics.normalized_mae,
                "test_rmse": transformer.test_metrics.rmse,
                "meets_target": transformer.meets_target,
                "test_db_metrics": _db_metrics_payload(transformer),
                "checkpoint_path": transformer.checkpoint_path,
                "last_test_prediction": transformer.last_test_prediction,
                "last_test_target": transformer.last_test_target,
            },
        },
        "mitigation": {
            "desired_gain_db": result.mitigation.zf_result.desired_gain_db,
            "ntn_gain_db": result.mitigation.zf_result.ntn_gain_db,
            "suppression_db": result.mitigation.zf_result.suppression_db,
            "predictor_used": result.mitigation.predictor_used,
            "estimated_ntn_angle_rad": result.mitigation.estimated_ntn_angle_rad,
            "true_ntn_angle_rad": result.mitigation.true_ntn_angle_rad,
            "predicted_interference_power": result.mitigation.predicted_interference_power,
            "true_interference_power": result.mitigation.true_interference_power,
            "zf_gate_passed": result.mitigation.zf_gate_passed,
            "ppo_trained": result.mitigation.ppo_trained,
        },
    }
