"""Tests for the Phase 3 dataset builder."""

import json

import numpy as np
import pytest

from src.config import SimulationConfig
from src.phase3_data import load_phase3_dataset
from src.phase3_dataset_builder import DatasetBuildConfig, build_phase3_dataset, make_windows, save_dataset
from src.timeseries import TimeSeriesConfig


def test_make_windows_shapes_and_alignment() -> None:
    series = np.arange(30 * 3, dtype=np.float32).reshape(30, 3)
    x, y = make_windows(series, lookback=10, horizon=1, target_index=0)

    assert x.shape == (20, 10, 3)
    assert y.shape == (20,)
    assert np.array_equal(x[0], series[0:10])
    assert y[0] == series[10, 0]  # one slot after the window
    assert np.array_equal(x[-1], series[19:29])
    assert y[-1] == series[29, 0]


def test_make_windows_horizon_shifts_target() -> None:
    series = np.arange(30 * 3, dtype=np.float32).reshape(30, 3)
    x, y = make_windows(series, lookback=10, horizon=3, target_index=0)
    assert x.shape[0] == 18
    assert y[0] == series[12, 0]


def test_make_windows_rejects_short_series() -> None:
    with pytest.raises(ValueError):
        make_windows(np.zeros((10, 3)), lookback=10, horizon=1)


def _small_build() -> tuple:
    return build_phase3_dataset(
        SimulationConfig(),
        TimeSeriesConfig(n_slots=60),
        DatasetBuildConfig(scenario="urban", user_counts=(4,), acir_points_db=(20.0, 30.0), runs_per_point=3),
    )


def test_build_dataset_shapes_and_run_wise_split() -> None:
    split, meta = _small_build()

    assert split.X_train.shape[1:] == (10, 3)
    assert split.X_val.shape[1:] == (10, 3)
    assert split.X_test.shape[1:] == (10, 3)
    assert np.all(np.isfinite(split.X_train)) and np.all(np.isfinite(split.y_train))

    runs = meta["runs"]
    assert len(runs) == 6
    assert {r["split"] for r in runs} == {"train", "val", "test"}
    for name in ("train", "val", "test"):
        assert sum(r["windows"] for r in runs if r["split"] == name) == meta["window_counts"][name]
    # 4 users * (60 - 10 - 1 + 1) windows per run
    assert all(r["windows"] == 4 * 50 for r in runs)


def test_save_and_reload_with_phase3_loader(tmp_path) -> None:
    split, meta = _small_build()
    save_dataset(split, meta, tmp_path)

    reloaded = load_phase3_dataset(tmp_path, require_pre_split=True)
    assert np.array_equal(reloaded.X_test, split.X_test)
    assert np.array_equal(reloaded.y_train, split.y_train)
    assert json.loads((tmp_path / "metadata.json").read_text())["target"] == "interference_local_dbm"  # primary target = local mean


def test_build_config_validation() -> None:
    with pytest.raises(ValueError):
        DatasetBuildConfig(scenario="urban", horizon_slots=5)
    with pytest.raises(ValueError):
        build_phase3_dataset(
            SimulationConfig(),
            TimeSeriesConfig(n_slots=60),
            DatasetBuildConfig(scenario="urban", user_counts=(4,), acir_points_db=(20.0,), runs_per_point=2),
        )


# --- audit M5 / m4 / m5 / m6 -------------------------------------------------------------------

from src.phase3_data import window_overlap_fraction  # noqa: E402
from src.phase3_dataset_builder import (  # noqa: E402
    assign_run_splits,
    dataset_dir_name,
    scenario_seed_offset,
    stratum_split_counts,
)


def _build(**overrides):
    kwargs = dict(scenario="urban", user_counts=(4,), acir_points_db=(20.0, 30.0), runs_per_point=3)
    kwargs.update(overrides)
    return build_phase3_dataset(SimulationConfig(), TimeSeriesConfig(n_slots=60), DatasetBuildConfig(**kwargs))


def test_make_windows_accepts_separate_target_series() -> None:
    series = np.arange(30 * 3, dtype=np.float32).reshape(30, 3)
    targets = np.arange(30, dtype=np.float32) * 10.0
    x, y = make_windows(series, lookback=10, horizon=2, targets=targets)

    assert y[0] == targets[11]
    assert y[-1] == targets[29]
    assert np.array_equal(x[0], series[0:10])
    with pytest.raises(ValueError):
        make_windows(series, targets=np.zeros(5))


def test_local_mean_and_instantaneous_share_inputs_but_not_targets() -> None:
    local, meta_local = _build(target="local_mean")
    inst, meta_inst = _build(target="instantaneous")

    assert np.array_equal(local.X_train, inst.X_train)  # same runs, same faded inputs
    assert np.array_equal(local.X_test, inst.X_test)
    assert not np.array_equal(local.y_test, inst.y_test)
    assert meta_local["target"] == "interference_local_dbm"
    assert meta_inst["target"] == "interference_dbm"
    assert meta_local["target_kind"] == "local_mean"
    assert "fast-fading term removed" in meta_local["target_definition"]


def test_local_mean_target_has_less_spread_than_instantaneous_target() -> None:
    local, _ = _build(target="local_mean")
    inst, _ = _build(target="instantaneous")
    # Same runs and inputs; removing fading from the target cannot add variance across the test set
    # beyond the fading-induced spread, and the two targets must genuinely differ.
    assert float(np.abs(local.y_test - inst.y_test).mean()) > 1.0
    assert np.std(local.y_test) < np.std(inst.y_test)


def test_metadata_records_channel_model_and_seeds() -> None:
    _, meta = _build()
    assert meta["slot_duration_s"] == 0.5
    assert meta["fast_fading_rho"] == pytest.approx(0.1024, abs=1e-3)
    assert meta["doppler_hz"] == pytest.approx(10.0, abs=0.1)
    assert meta["timeseries_config"]["n_slots"] == 60
    assert meta["scenario_seed_offset"] == scenario_seed_offset("urban")
    assert meta["persistence_baseline"].startswith("last observed value")


def test_sinr_mask_zeroes_only_that_channel() -> None:
    plain, _ = _build()
    masked, meta = _build(mask_features=("sinr_db",))

    assert np.all(masked.X_train[:, :, 2] == 0.0)
    assert np.array_equal(masked.X_train[:, :, :2], plain.X_train[:, :, :2])
    assert np.array_equal(masked.y_test, plain.y_test)
    assert meta["masked_features"] == ["sinr_db"]
    with pytest.raises(ValueError):
        DatasetBuildConfig(scenario="urban", mask_features=("interference_dbm",))


def test_runs_are_stratified_so_every_operating_point_hits_every_split() -> None:
    _, meta = _build(runs_per_point=3)
    strata: dict = {}
    for run in meta["runs"]:
        strata.setdefault((run["user_count"], run["acir_db"]), set()).add(run["split"])
    assert len(strata) == 2
    assert all(splits == {"train", "val", "test"} for splits in strata.values())


def test_stratum_split_counts_use_rounding() -> None:
    assert stratum_split_counts(3) == (1, 1, 1)
    assert stratum_split_counts(6) == (4, 1, 1)  # 0.9 -> 1
    assert stratum_split_counts(10) == (6, 2, 2)  # 1.5 -> 2 (half-up)
    assert stratum_split_counts(20) == (14, 3, 3)
    with pytest.raises(ValueError, match="at least 3 runs"):
        stratum_split_counts(2)


def test_assign_run_splits_is_deterministic_and_stratified() -> None:
    strata = [(4, 20.0)] * 6 + [(4, 30.0)] * 6
    first = assign_run_splits(strata, seed=1)
    assert first == assign_run_splits(strata, seed=1)
    assert first != assign_run_splits(strata, seed=2)
    for stratum_slice in (first[:6], first[6:]):
        assert sorted(stratum_slice) == ["test", "train", "train", "train", "train", "val"]


def test_scenarios_do_not_share_run_seeds() -> None:
    _, urban = _build(scenario="urban")
    _, rural = _build(scenario="rural")
    urban_seeds = {r["seed"] for r in urban["runs"]}
    rural_seeds = {r["seed"] for r in rural["runs"]}
    assert urban_seeds.isdisjoint(rural_seeds)
    assert scenario_seed_offset("urban") != scenario_seed_offset("rural")
    assert scenario_seed_offset("urban") == scenario_seed_offset("urban")


def test_built_dataset_has_no_window_overlap_between_splits() -> None:
    split, _ = _build()
    assert window_overlap_fraction(split.X_train, split.X_test) == 0.0
    assert window_overlap_fraction(split.X_train, split.X_val) == 0.0


def test_dataset_dir_names() -> None:
    assert dataset_dir_name("urban") == "urban"
    assert dataset_dir_name("rural", "instantaneous") == "rural_instantaneous"
    assert dataset_dir_name("urban", "local_mean", ("sinr_db",)) == "urban_nosinr"
    assert dataset_dir_name("urban", "local_mean", ("signal_dbm",)) == "urban_nosignal"
