"""Tests for dB-domain metrics and the persistence-relative gate (audit C1)."""

import numpy as np
import pytest

from src.phase3_metrics import (
    DEFAULT_MIN_SKILL_VS_PERSISTENCE,
    DbMetrics,
    evaluate_db_metrics,
    mae_db,
    meets_skill_gate,
    persistence_prediction,
    skill_vs_baseline,
    within_db_share,
    within_tolerance_accuracy,
)
from src.phase3_pipeline import Phase3Gates, Phase3RunResult, to_serializable_summary
from src.phase3_prediction import PredictionModelResult, PredictionStageResult, run_prediction_stage
from src.phase3_metrics import PredictionMetrics
from src.phase3_data import Phase3DatasetSplit


def _windows_from_series(series: np.ndarray, lookback: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Single-feature-padded windows: feature 0 is the series, features 1-2 are zeros."""
    n = len(series) - lookback
    x = np.zeros((n, lookback, 3), dtype=np.float32)
    for i in range(n):
        x[i, :, 0] = series[i : i + lookback]
    y = series[lookback:].astype(np.float32)
    return x, y


def _ar1_dbm_series(n: int = 5000, rho: float = 0.7, sigma_db: float = 3.0, mean_dbm: float = -80.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma_db)
    innovations = rng.normal(0.0, sigma_db * np.sqrt(1 - rho**2), n)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + innovations[t]
    return mean_dbm + out


def test_default_min_skill_is_fixed_at_fifteen_percent() -> None:
    assert DEFAULT_MIN_SKILL_VS_PERSISTENCE == 0.15
    assert Phase3Gates().lstm_min_skill_vs_persistence == 0.15


def test_mae_and_within_db_share_known_values() -> None:
    truth = np.array([-80.0, -80.0, -80.0, -80.0])
    pred = np.array([-80.5, -78.0, -75.0, -80.0])  # errors 0.5, 2, 5, 0

    assert mae_db(truth, pred) == pytest.approx(1.875)
    assert within_db_share(truth, pred, tolerance_db=1.0) == pytest.approx(0.5)
    assert within_db_share(truth, pred, tolerance_db=3.0) == pytest.approx(0.75)


def test_within_db_share_rejects_bad_tolerance_and_length() -> None:
    with pytest.raises(ValueError):
        within_db_share(np.zeros(3), np.zeros(3), tolerance_db=0.0)
    with pytest.raises(ValueError):
        mae_db(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        mae_db(np.zeros(0), np.zeros(0))


def test_persistence_prediction_uses_last_slot_of_interference_feature() -> None:
    x = np.zeros((2, 10, 3), dtype=np.float32)
    x[0, -1, 0] = -70.0
    x[1, -1, 0] = -90.0
    x[:, -1, 1] = 5.0  # other features must be ignored

    assert persistence_prediction(x).tolist() == [-70.0, -90.0]
    with pytest.raises(ValueError):
        persistence_prediction(np.zeros((4, 10)))
    with pytest.raises(ValueError):
        persistence_prediction(x, feature_index=3)


def test_skill_vs_baseline_math_and_degenerate_baseline() -> None:
    assert skill_vs_baseline(0.8, 1.0) == pytest.approx(0.2)
    assert skill_vs_baseline(1.0, 1.0) == pytest.approx(0.0)
    assert skill_vs_baseline(1.5, 1.0) == pytest.approx(-0.5)
    assert skill_vs_baseline(0.0, 0.0) == 0.0  # zero-error baseline can never pass the gate
    with pytest.raises(ValueError):
        skill_vs_baseline(-1.0, 1.0)


def test_persistence_scores_high_on_legacy_metric_but_has_zero_skill() -> None:
    """The audit's core finding: the old metric rewards 'repeat the last value'."""
    series = _ar1_dbm_series()
    x, y = _windows_from_series(series)
    persistence = persistence_prediction(x)

    legacy = within_tolerance_accuracy(y, persistence)  # old gate, 0.80 required
    metrics = evaluate_db_metrics(y, persistence, x)

    assert legacy > 0.80  # old metric: a do-nothing predictor "passes"
    assert metrics.within_1db < 0.50  # in dB, most predictions miss by more than 1 dB
    assert legacy - metrics.within_1db > 0.40  # the legacy metric massively overstates accuracy
    assert metrics.skill_vs_persistence == pytest.approx(0.0)
    assert not meets_skill_gate(metrics)  # new gate: persistence does not pass


def test_skill_gate_boundaries() -> None:
    series = _ar1_dbm_series()
    x, y = _windows_from_series(series)
    persistence = persistence_prediction(x)
    baseline_mae = mae_db(y, persistence)

    # Construct predictions whose error is exactly scaled relative to persistence.
    def scaled(factor: float) -> np.ndarray:
        return y + (persistence - y) * factor

    better = evaluate_db_metrics(y, scaled(0.80), x)  # 20% lower MAE
    worse = evaluate_db_metrics(y, scaled(0.90), x)  # 10% lower MAE

    assert better.skill_vs_persistence == pytest.approx(0.20, abs=1e-6)
    assert meets_skill_gate(better)
    assert not meets_skill_gate(worse)
    assert meets_skill_gate(worse, min_skill=0.05)
    assert better.persistence_mae_db == pytest.approx(baseline_mae)
    with pytest.raises(ValueError):
        meets_skill_gate(better, min_skill=1.0)


def test_evaluate_db_metrics_perfect_prediction() -> None:
    x, y = _windows_from_series(_ar1_dbm_series(n=500))
    metrics = evaluate_db_metrics(y, y, x)

    assert metrics.mae_db == 0.0
    assert metrics.rmse_db == 0.0
    assert metrics.within_1db == 1.0
    assert metrics.skill_vs_persistence == pytest.approx(1.0)


def test_evaluate_db_metrics_validates_window_count() -> None:
    x, y = _windows_from_series(_ar1_dbm_series(n=100))
    with pytest.raises(ValueError, match="one window per target"):
        evaluate_db_metrics(y[:-1], y[:-1], x)


def test_db_metrics_dataclass_validation() -> None:
    with pytest.raises(ValueError):
        DbMetrics(1.0, 1.0, 1.5, 0.5, 1.0, 1.0, 0.0)  # share > 1
    with pytest.raises(ValueError):
        DbMetrics(-1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0)  # negative error
    with pytest.raises(ValueError):
        DbMetrics(float("nan"), 1.0, 0.5, 0.5, 1.0, 1.0, 0.0)


# --- plumbing through the stage / pipeline ---------------------------------------------------


def _model_result(name: str, skill: float, *, min_skill: float = 0.15) -> PredictionModelResult:
    db = DbMetrics(
        mae_db=4.0, rmse_db=5.0, within_1db=0.2, within_3db=0.5,
        persistence_mae_db=5.0, persistence_rmse_db=6.0, skill_vs_persistence=skill,
    )
    legacy = PredictionMetrics(within_10pct_accuracy=0.9, normalized_mae=0.05, rmse=5.0)
    return PredictionModelResult(
        model_name=name, best_epoch=3, best_validation_loss=0.1,
        validation_metrics=legacy, test_metrics=legacy, target_accuracy=0.8,
        meets_target=meets_skill_gate(db, min_skill=min_skill),
        validation_db_metrics=db, test_db_metrics=db, min_skill=min_skill,
        last_test_prediction=-70.0, last_test_target=-71.0, test_target_min=-90.0, test_target_max=-50.0,
    )


def test_legacy_accuracy_no_longer_decides_the_gate() -> None:
    passing_legacy_failing_skill = _model_result("lstm", skill=0.05)
    assert passing_legacy_failing_skill.test_metrics.within_10pct_accuracy >= 0.8
    assert not passing_legacy_failing_skill.meets_target


def test_run_prediction_stage_forwards_min_skill(monkeypatch) -> None:
    seen: dict[str, float] = {}

    def fake_lstm(*args, **kwargs):
        seen["lstm"] = kwargs["min_skill"]
        return _model_result("lstm", skill=0.05, min_skill=kwargs["min_skill"])

    def fake_transformer(*args, **kwargs):
        seen["transformer"] = kwargs["min_skill"]
        return _model_result("transformer", skill=0.30, min_skill=kwargs["min_skill"])

    monkeypatch.setattr("src.phase3_prediction.train_lstm_forecaster", fake_lstm)
    monkeypatch.setattr("src.phase3_prediction.train_transformer_forecaster", fake_transformer)
    x = np.zeros((6, 10, 3), dtype=np.float32)
    y = np.zeros(6, dtype=np.float32)
    split = Phase3DatasetSplit(x[:2], y[:2], x[2:4], y[2:4], x[4:], y[4:])

    outcome = run_prediction_stage(split, min_skill=0.25)

    assert seen == {"lstm": 0.25, "transformer": 0.25}
    assert outcome.transformer_executed  # LSTM missed the skill gate -> fallback ran


def test_summary_contains_db_metrics_for_lstm_and_transformer() -> None:
    stage = PredictionStageResult(
        lstm=_model_result("lstm", skill=0.05),
        transformer=_model_result("transformer", skill=0.30),
        transformer_executed=True,
    )
    from src.phase3_pipeline import MitigationStageResult
    from src.phase3_beamforming import evaluate_zf_nulling

    mitigation = MitigationStageResult(
        zf_result=evaluate_zf_nulling(desired_angle_rad=0.0, ntn_angle_rad=0.35, antenna_count=16),
        predictor_used="transformer", estimated_ntn_angle_rad=0.3, true_ntn_angle_rad=0.3,
        predicted_interference_power=-70.0, true_interference_power=-71.0,
        zf_gate_passed=True, ppo_trained=False,
    )
    payload = to_serializable_summary(Phase3RunResult(prediction=stage, mitigation=mitigation))

    lstm_db = payload["prediction"]["lstm"]["test_db_metrics"]
    assert lstm_db["skill_vs_persistence"] == 0.05
    assert lstm_db["min_skill_required"] == 0.15
    assert lstm_db["persistence_mae_db"] == 5.0
    assert payload["prediction"]["transformer"]["test_db_metrics"]["skill_vs_persistence"] == 0.30
    assert payload["prediction"]["lstm"]["meets_target"] is False
    assert payload["prediction"]["lstm"]["test_within_10pct_accuracy"] == 0.9  # legacy kept


def test_gates_validate_min_skill_range() -> None:
    with pytest.raises(ValueError):
        Phase3Gates(lstm_min_skill_vs_persistence=1.0)
    with pytest.raises(ValueError):
        Phase3Gates(lstm_min_skill_vs_persistence=-0.1)
    with pytest.raises(ValueError):
        PredictionModelResult(
            model_name="x", best_epoch=1, best_validation_loss=0.0,
            validation_metrics=PredictionMetrics(0.5, 0.1, 0.1), test_metrics=PredictionMetrics(0.5, 0.1, 0.1),
            target_accuracy=0.8, meets_target=False, min_skill=1.5,
        )
