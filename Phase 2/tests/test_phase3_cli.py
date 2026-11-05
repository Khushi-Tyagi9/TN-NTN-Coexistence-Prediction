"""CLI / pipeline behaviour found by the smoke run: a failed gate is a result, not a crash."""

import json

import numpy as np
import pytest

from run_phase3_colab import build_argument_parser, main
from src.config import SimulationConfig
from src.phase3_dataset_builder import DatasetBuildConfig, build_phase3_dataset, save_dataset
from src.phase3_metrics import DbMetrics, PredictionMetrics
from src.phase3_pipeline import run_phase3_pipeline, to_serializable_summary
from src.phase3_prediction import PredictionModelResult, PredictionStageResult
from src.timeseries import TimeSeriesConfig


def _model(name: str, *, prototype: bool, final: bool = False) -> PredictionModelResult:
    db = DbMetrics(
        mae_db=2.0, rmse_db=3.0, within_1db=0.3, within_3db=0.7,
        persistence_mae_db=4.0, persistence_rmse_db=5.0, skill_vs_persistence=0.5,
        moving_average_mae_db=2.5, ar_mae_db=2.1, best_baseline="linear_ar_10",
        best_baseline_mae_db=2.1, skill_vs_best_baseline=0.05,
    )
    legacy = PredictionMetrics(0.9, 0.05, 3.0)
    return PredictionModelResult(
        model_name=name, best_epoch=2, best_validation_loss=0.1, validation_metrics=legacy, test_metrics=legacy,
        target_accuracy=0.8, meets_target=prototype, meets_final_target=final,
        validation_db_metrics=db, test_db_metrics=db,
        last_test_prediction=-70.0, last_test_target=-71.0, test_target_min=-90.0, test_target_max=-50.0,
    )


def _fake_track(stage: PredictionStageResult, captured: dict | None = None):
    def fake(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return None, stage

    return fake


def test_failed_prototype_gate_skips_mitigation_instead_of_crashing(monkeypatch) -> None:
    stage = PredictionStageResult(lstm=_model("lstm", prototype=False), transformer=None, transformer_executed=False)
    monkeypatch.setattr("src.phase3_pipeline.run_prediction_track", _fake_track(stage))

    result = run_phase3_pipeline(data_root="unused")
    payload = to_serializable_summary(result)

    assert result.mitigation is None
    assert "prototype gate" in result.mitigation_skipped_reason
    assert payload["mitigation"] is None and payload["mitigation_skipped_reason"] == result.mitigation_skipped_reason
    assert payload["prediction"]["lstm"]["meets_target"] is False  # the failed gate is reported
    json.dumps(payload)


def test_transformer_that_passes_unblocks_mitigation(monkeypatch) -> None:
    stage = PredictionStageResult(
        lstm=_model("lstm", prototype=False), transformer=_model("transformer", prototype=True), transformer_executed=True
    )
    monkeypatch.setattr("src.phase3_pipeline.run_prediction_track", _fake_track(stage))

    result = run_phase3_pipeline(data_root="unused", antenna_count=16)

    assert result.mitigation is not None and result.mitigation.predictor_used == "transformer"
    assert result.mitigation_skipped_reason is None


def test_prediction_only_never_runs_mitigation(monkeypatch) -> None:
    stage = PredictionStageResult(lstm=_model("lstm", prototype=True), transformer=None, transformer_executed=False)
    monkeypatch.setattr("src.phase3_pipeline.run_prediction_track", _fake_track(stage))
    monkeypatch.setattr(
        "src.phase3_pipeline.run_mitigation_track", lambda **kw: pytest.fail("mitigation must not run")
    )

    result = run_phase3_pipeline(data_root="unused", skip_mitigation=True)

    assert result.mitigation is None and "prediction-only" in result.mitigation_skipped_reason


def test_transformer_fallback_flag_is_forwarded(monkeypatch) -> None:
    captured: dict = {}
    stage = PredictionStageResult(lstm=_model("lstm", prototype=True), transformer=None, transformer_executed=False)
    monkeypatch.setattr("src.phase3_pipeline.run_prediction_track", _fake_track(stage, captured))

    run_phase3_pipeline(data_root="unused", run_transformer_if_eligible=False, skip_mitigation=True)

    assert captured["run_transformer_if_eligible"] is False


def test_cli_flags_parse() -> None:
    parser = build_argument_parser()
    defaults = parser.parse_args(["--data-root", "x"])
    flagged = parser.parse_args(["--data-root", "x", "--prediction-only", "--no-transformer-fallback"])

    assert not defaults.prediction_only and not defaults.no_transformer_fallback
    assert flagged.prediction_only and flagged.no_transformer_fallback


def test_dataset_metadata_carries_provenance() -> None:
    _, metadata = build_phase3_dataset(
        SimulationConfig(),
        TimeSeriesConfig(n_slots=60),
        DatasetBuildConfig(scenario="urban", user_counts=(4,), acir_points_db=(20.0, 30.0), runs_per_point=3),
    )
    provenance = metadata["provenance"]
    assert provenance["seed"] == 1000
    assert {"numpy", "python", "git_sha", "git_dirty", "generated_utc"} <= provenance.keys()
    json.dumps(metadata)


# --- real end-to-end runs (need torch) ------------------------------------------------------------


@pytest.fixture()
def tiny_dataset(tmp_path):
    split, metadata = build_phase3_dataset(
        SimulationConfig(),
        TimeSeriesConfig(n_slots=60),
        DatasetBuildConfig(scenario="urban", user_counts=(4,), acir_points_db=(20.0, 30.0), runs_per_point=3),
    )
    return save_dataset(split, metadata, tmp_path / "urban")


def _run_cli(dataset_dir, tmp_path, *extra: str) -> dict:
    summary = tmp_path / "out" / "summary.json"
    code = main(
        [
            "--data-root", str(dataset_dir), "--scenario", "urban", "--lstm-epochs", "1", "--seed", "5",
            "--checkpoint-dir", str(tmp_path / "ckpt"), "--summary-json", str(summary), *extra,
        ]
    )
    assert code == 0
    return json.loads(summary.read_text(encoding="utf-8"))


def test_cli_prediction_only_end_to_end(tiny_dataset, tmp_path) -> None:
    pytest.importorskip("torch", reason="PyTorch not installed: end-to-end CLI is NOT exercised")

    payload = _run_cli(tiny_dataset, tmp_path, "--prediction-only", "--no-transformer-fallback")

    assert payload["mitigation"] is None and "prediction-only" in payload["mitigation_skipped_reason"]
    assert payload["prediction"]["transformer_executed"] is False
    lstm = payload["prediction"]["lstm"]
    db = lstm["test_db_metrics"]
    assert {"mae_db", "within_1db", "within_3db", "persistence_mae_db", "moving_average_mae_db", "ar_mae_db",
            "skill_vs_persistence", "skill_vs_best_baseline"} <= db.keys()
    assert isinstance(lstm["meets_target"], bool) and isinstance(lstm["meets_final_target"], bool)
    assert payload["run_info"]["seed"] == 5 and payload["run_info"]["scenario"] == "urban"
    assert payload["dataset"]["target_kind"] == "local_mean"
    assert {"git_sha", "git_dirty", "numpy", "seed"} <= payload["dataset"]["provenance"].keys()  # dataset provenance kept
    assert (tmp_path / "ckpt" / "lstm_best.pt").exists()


def test_cli_with_an_unreachable_gate_still_writes_the_summary(tiny_dataset, tmp_path) -> None:
    pytest.importorskip("torch", reason="PyTorch not installed: end-to-end CLI is NOT exercised")

    payload = _run_cli(tiny_dataset, tmp_path, "--min-skill", "0.99", "--no-transformer-fallback")

    assert payload["prediction"]["lstm"]["meets_target"] is False
    assert payload["mitigation"] is None
    assert "prototype gate" in payload["mitigation_skipped_reason"]


def test_cli_run_is_reproducible_for_a_seed(tiny_dataset, tmp_path) -> None:
    pytest.importorskip("torch", reason="PyTorch not installed: end-to-end CLI is NOT exercised")

    first = _run_cli(tiny_dataset, tmp_path / "a", "--prediction-only", "--no-transformer-fallback")
    second = _run_cli(tiny_dataset, tmp_path / "b", "--prediction-only", "--no-transformer-fallback")

    assert first["prediction"]["lstm"]["test_db_metrics"] == second["prediction"]["lstm"]["test_db_metrics"]
