"""The descriptive baseline figures: numbers must match the shared metric code; no training involved."""

import json

import numpy as np
import pytest

import make_baseline_figures as figs
from src.config import SimulationConfig
from src.phase3_dataset_builder import DatasetBuildConfig, build_phase3_dataset, save_dataset
from src.phase3_metrics import evaluate_db_metrics, fit_ar_baseline
from src.timeseries import TimeSeriesConfig


@pytest.fixture()
def tiny_root(tmp_path):
    root = tmp_path / "data"
    for scenario in ("urban", "rural"):
        split, metadata = build_phase3_dataset(
            SimulationConfig(), TimeSeriesConfig(n_slots=60),
            DatasetBuildConfig(scenario=scenario, user_counts=(4,), acir_points_db=(20.0, 30.0), runs_per_point=3),
        )
        save_dataset(split, metadata, root / scenario)
    return root


def test_baseline_errors_match_the_shared_metric_code(tiny_root) -> None:
    from src.phase3_data import load_phase3_dataset

    errors = figs.compute_baseline_errors(tiny_root)

    split = load_phase3_dataset(tiny_root / "urban", require_pre_split=True)
    reference = evaluate_db_metrics(
        split.y_val, split.y_val, split.X_val, ar_coefficients=fit_ar_baseline(split.X_train, split.y_train)
    )
    got = errors["urban"]["val_mae_db"]
    assert got["persistence"] == pytest.approx(reference.persistence_mae_db)
    assert got["moving_average_10"] == pytest.approx(reference.moving_average_mae_db)
    assert got["linear_ar_10"] == pytest.approx(reference.ar_mae_db)
    assert errors["urban"]["n_val_windows"] == len(split.y_val)


def test_all_three_figures_and_the_numbers_file_are_written_at_200_dpi(tiny_root, tmp_path) -> None:
    from PIL import Image

    out = tmp_path / "figures"
    assert figs.main(["--data-root", str(tiny_root), "--out", str(out)]) == 0

    for name in ("urban_run_time_series.png", "urban_local_mean_target.png", "baseline_errors.png"):
        path = out / name
        assert path.exists() and path.stat().st_size > 10_000
        with Image.open(path) as image:
            assert round(image.info["dpi"][0]) == 200
    numbers = json.loads((out / "baseline_errors.json").read_text(encoding="utf-8"))
    assert set(numbers["baselines"]) == {"urban", "rural"}
    assert numbers["figure_run"]["acir_db"] == 30.0 and "git_sha" in numbers["run_info"]


def test_figure_run_is_the_first_dataset_run_at_the_figure_acir(tiny_root) -> None:
    metadata = json.loads((tiny_root / "urban" / "metadata.json").read_text(encoding="utf-8"))
    run, series = figs.simulate_figure_run(metadata)

    assert run == next(r for r in metadata["runs"] if r["acir_db"] == 30.0)
    assert series.interference_dbm.shape == (run["user_count"], metadata["n_slots_per_run"])
    assert np.all(series.interference_local_dbm <= series.interference_local_dbm + 1e-9)
    with pytest.raises(ValueError, match="No run with ACIR"):
        figs.pick_run({"runs": [{"acir_db": 20.0}]})


def test_figures_never_open_the_test_files(tiny_root, tmp_path) -> None:
    for scenario in ("urban", "rural"):
        for name in ("X_test.npy", "y_test.npy"):
            (tiny_root / scenario / name).unlink()  # would raise if anything tried to read the test split

    assert figs.main(["--data-root", str(tiny_root), "--out", str(tmp_path / "figs")]) == 0


def test_figure_title_and_labels_are_the_requested_ones() -> None:
    import inspect

    source = inspect.getsource(figs.figure_baseline_bars)
    assert "Baseline errors; LSTM not shown here" in source
    assert "validation MAE (dB)" in source
