"""Preliminary validation-only run: the test split must never be read."""

import json

import numpy as np
import pytest

import run_preliminary
from src.phase3_preliminary import PRELIMINARY_CONFIG

torch = pytest.importorskip("torch", reason="PyTorch not installed: preliminary run is NOT exercised")


@pytest.fixture()
def tiny_root(tmp_path):
    from src.config import SimulationConfig
    from src.phase3_dataset_builder import DatasetBuildConfig, build_phase3_dataset, save_dataset
    from src.timeseries import TimeSeriesConfig

    root = tmp_path / "data"
    split, metadata = build_phase3_dataset(
        SimulationConfig(), TimeSeriesConfig(n_slots=60),
        DatasetBuildConfig(scenario="urban", user_counts=(4,), acir_points_db=(20.0, 30.0), runs_per_point=3),
    )
    save_dataset(split, metadata, root / "urban")
    for name in ("X_test.npy", "y_test.npy"):
        (root / "urban" / name).unlink()  # any attempt to read the test split would now fail
    return root


def test_the_declared_preliminary_config() -> None:
    c = PRELIMINARY_CONFIG
    assert (c.hidden_size, c.num_layers, c.learning_rate, c.seed) == (96, 1, 1e-3, 42)
    assert (c.max_epochs, c.early_stopping_patience, c.batch_size) == (40, 6, 64)


def test_preliminary_run_never_needs_the_test_split_and_reports_validation_metrics(tiny_root, tmp_path, monkeypatch) -> None:
    from src.phase3_prediction import LSTMTrainingConfig

    monkeypatch.setattr(run_preliminary, "PRELIMINARY_CONFIG", LSTMTrainingConfig(max_epochs=2, device="cpu", seed=42))
    out = tmp_path / "prelim"
    assert run_preliminary.main(["--scenario", "urban", "--data-root", str(tiny_root), "--out", str(out), "--threads", "2"]) == 0

    folder = out / "urban"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert "test split was not read" in summary["split_evaluated"] and "NOT the final protocol" in summary["status"]
    validation = summary["validation"]
    assert {"mae_db", "within_1db", "within_3db", "skill_vs_persistence", "skill_vs_best_baseline", "best_baseline",
            "baselines_mae_db"} <= validation.keys()
    assert set(validation["baselines_mae_db"]) == {"persistence", "moving_average_10", "linear_ar_10"}
    assert "test" not in json.dumps(summary["dataset"]["window_counts"]) and set(summary["dataset"]["sha256"]) == {
        "X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"}
    assert len(summary["seconds_per_epoch"]) == summary["epochs_run"] and summary["run_info"]["seed"] == 42
    assert not any("gate" in key for key in validation)  # no gate result for exploratory runs
    for plot in ("training_curves.png", "actual_vs_predicted_val.png"):
        assert (folder / plot).exists() and (folder / plot).stat().st_size > 5_000

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        run_preliminary.main(["--scenario", "urban", "--data-root", str(tiny_root), "--out", str(out)])


def test_eta_over_the_limit_stops_after_epoch_one_and_writes_nothing(tiny_root, tmp_path, monkeypatch, capsys) -> None:
    from src.phase3_prediction import LSTMTrainingConfig

    monkeypatch.setattr(run_preliminary, "PRELIMINARY_CONFIG", LSTMTrainingConfig(max_epochs=3, device="cpu", seed=42))
    out = tmp_path / "prelim"
    code = run_preliminary.main(["--scenario", "urban", "--data-root", str(tiny_root), "--out", str(out),
                                 "--threads", "2", "--eta-limit-minutes", "0"])

    text = capsys.readouterr().out
    assert code == 3 and "s/epoch" in text and "ETA" in text and "STOPPED" in text
    assert not out.exists()


def test_missing_dataset_is_a_clear_error(tmp_path) -> None:
    with pytest.raises(SystemExit, match="not found"):
        run_preliminary.main(["--scenario", "rural", "--data-root", str(tmp_path), "--out", str(tmp_path / "o")])


def test_parser_defaults() -> None:
    args = run_preliminary.build_parser().parse_args(["--scenario", "urban", "--variant", "instantaneous"])
    assert (args.threads, args.eta_limit_minutes, args.variant) == (12, 20.0, "instantaneous")
    with pytest.raises(SystemExit):
        run_preliminary.build_parser().parse_args(["--scenario", "urban", "--variant", "nosinr"])


def test_run_info_is_snapshotted_before_training(tiny_root, tmp_path, monkeypatch) -> None:
    from src.phase3_prediction import LSTMTrainingConfig

    events: list[str] = []
    real_collect, real_train = run_preliminary.collect_run_info, run_preliminary.train_validation_only
    monkeypatch.setattr(run_preliminary, "PRELIMINARY_CONFIG", LSTMTrainingConfig(max_epochs=1, device="cpu", seed=42))
    monkeypatch.setattr(run_preliminary, "collect_run_info", lambda *a, **k: (events.append("run_info"), real_collect(*a, **k))[1])
    monkeypatch.setattr(run_preliminary, "train_validation_only", lambda *a, **k: (events.append("train"), real_train(*a, **k))[1])

    run_preliminary.main(["--scenario", "urban", "--data-root", str(tiny_root), "--out", str(tmp_path / "o"), "--threads", "2"])

    assert events == ["run_info", "train"]
