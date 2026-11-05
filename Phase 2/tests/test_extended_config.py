"""Phase 4 config additions: opt-in only, existing defaults and results stay reproducible."""

import numpy as np
import pytest

from src.config import (
    DEFAULT_ANTENNA_COUNTS,
    EXTENDED_ACIR_POINTS_DB,
    ACIRSweepConfig,
    AntennaConfig,
    ExperimentConfig,
    RadioConfig,
    SimulationConfig,
    TNDeploymentConfig,
    UserConfig,
)
from src.interference import evaluate_scenario_sinr
from src.phase3_dataset_builder import DatasetBuildConfig, build_phase3_dataset
from src.simulation import generate_network_scenario, run_deterministic_baseline
from src.timeseries import TimeSeriesConfig, simulate_time_series


# ------------------------------------------------------------- defaults are exactly what they were
def test_baseline_defaults_are_unchanged() -> None:
    config = SimulationConfig()

    assert config.acir_sweep_db == (20.0, 25.0, 30.0, 35.0, 40.0)
    assert (config.radio.acir_sweep.min_db, config.radio.acir_sweep.max_db) == (20.0, 40.0)
    assert config.tn_deployment.gnb_count == 1
    assert (config.users.min_count, config.users.max_count, config.users.default_count) == (4, 5, 5)
    assert config.experiment.user_count_sweep == (4, 5) and config.experiment.enforce_baseline_realism is True
    assert config.radio.carrier_frequency_hz == 2.0e9 and config.radio.bandwidth_hz == 20.0e6
    assert (config.hap.hap_count, config.satellite.leo_altitude_km) == (1, 550.0)


def test_baseline_still_rejects_the_relaxed_settings() -> None:
    with pytest.raises(ValueError, match="gnb_count"):
        SimulationConfig(tn_deployment=TNDeploymentConfig(gnb_count=3))
    with pytest.raises(ValueError, match="user_count"):
        SimulationConfig(users=UserConfig(min_count=4, max_count=8, default_count=5))
    with pytest.raises(ValueError):
        generate_network_scenario(SimulationConfig(), scenario_name="urban", user_count=8, seed=1)


def test_golden_baseline_numbers_from_before_the_change() -> None:
    """Values captured from the code before the Phase 4 config additions."""
    run = run_deterministic_baseline(SimulationConfig(), scenario_name="urban", user_count=5, acir_db=30.0, seed=7).run
    assert run.sinr.mean_sinr_db == pytest.approx(5.116672745540272, rel=1e-9)
    assert run.sinr.mean_interference_power_dbm == pytest.approx(-77.60878936158412, rel=1e-9)
    assert run.sinr.mean_signal_power_dbm == pytest.approx(-73.26786009004388, rel=1e-9)

    series = simulate_time_series(
        SimulationConfig(), TimeSeriesConfig(n_slots=60), scenario_name="rural", user_count=4, acir_db=25.0, seed=11
    )
    assert float(series.interference_dbm.mean()) == pytest.approx(-87.38785702051476, rel=1e-6)
    assert float(series.sinr_db.std()) == pytest.approx(5.152233270696191, rel=1e-6)
    assert float(series.interference_local_dbm.mean()) == pytest.approx(-84.57806507550235, rel=1e-6)


def test_golden_dataset_numbers_from_before_the_change() -> None:
    split, metadata = build_phase3_dataset(
        SimulationConfig(), TimeSeriesConfig(n_slots=60),
        DatasetBuildConfig(scenario="urban", user_counts=(4,), acir_points_db=(20.0, 30.0), runs_per_point=3),
    )
    assert split.X_train.shape == (400, 10, 3) and metadata["window_counts"] == {"train": 400, "val": 400, "test": 400}
    assert float(split.X_train.mean()) == pytest.approx(-49.14495849609375, rel=1e-5)
    assert float(split.y_train.std()) == pytest.approx(12.976710319519043, rel=1e-5)


# ------------------------------------------------------------------------------------- ACIR grid
def test_extended_acir_grid_is_12_to_45_every_3_db() -> None:
    expected = tuple(float(v) for v in (12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45))

    assert EXTENDED_ACIR_POINTS_DB == expected and len(expected) == 12
    sweep = ACIRSweepConfig.extended()
    assert (sweep.min_db, sweep.max_db, sweep.points_db) == (12.0, 45.0, expected)
    assert np.all(np.diff(sweep.points_db) == 3.0)
    assert SimulationConfig(radio=RadioConfig(acir_sweep=sweep)).acir_sweep_db == expected
    assert ACIRSweepConfig().points_db == (20.0, 25.0, 30.0, 35.0, 40.0)  # default sweep untouched


# -------------------------------------------------------------------------------------- antennas
def test_antenna_counts_default_and_resolution() -> None:
    config = SimulationConfig()

    assert DEFAULT_ANTENNA_COUNTS == (16, 32, 64) == config.antenna.counts
    assert config.antenna.default_count == 64
    assert config.resolve_antenna_count() == 64 and config.resolve_antenna_count(16) == 16
    with pytest.raises(ValueError, match="one of"):
        config.resolve_antenna_count(48)


@pytest.mark.parametrize(
    "kwargs",
    [dict(counts=()), dict(counts=(1, 16)), dict(counts=(16, 16, 32)), dict(counts=(32, 16)),
     dict(counts=(16, 32), default_count=64), dict(counts=(16.0, 32)), dict(counts=(True, 32), default_count=32)],
)
def test_antenna_config_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        AntennaConfig(**kwargs)


# ------------------------------------------------------------------------------ extended preset
def test_extended_preset_relaxes_limits_only_when_asked() -> None:
    config = SimulationConfig.extended()

    assert config.experiment.enforce_baseline_realism is False
    assert config.tn_deployment.gnb_count == 3
    assert (config.users.min_count, config.users.max_count) == (1, 20)
    assert config.acir_sweep_db == EXTENDED_ACIR_POINTS_DB
    assert config.antenna.counts == (16, 32, 64)
    assert config.experiment.user_count_sweep == (4, 8, 12, 16)
    assert SimulationConfig().experiment.enforce_baseline_realism is True  # baseline unaffected


def test_extended_preset_generates_and_evaluates_multi_gnb_scenarios() -> None:
    config = SimulationConfig.extended(gnb_count=3, max_users=20)

    scenario = generate_network_scenario(config, scenario_name="urban", user_count=12, seed=5)
    assert (len(scenario.gnbs), len(scenario.users), len(scenario.haps)) == (3, 12, 1)
    summary = evaluate_scenario_sinr(scenario, config, acir_db=12.0)
    assert all(len(user.interferers) == 3 for user in summary.user_metrics)

    series = simulate_time_series(config, TimeSeriesConfig(n_slots=30), scenario_name="rural", user_count=8,
                                  acir_db=45.0, seed=2)
    assert series.interference_dbm.shape == (8, 30) and np.all(np.isfinite(series.sinr_db))


def test_extended_preset_is_deterministic_and_aggregates_more_interference() -> None:
    one = SimulationConfig.extended(gnb_count=1)
    three = SimulationConfig.extended(gnb_count=3)
    kwargs = dict(scenario_name="urban", user_count=6, acir_db=30.0, seed=9)

    a = run_deterministic_baseline(three, **kwargs).run.sinr
    assert a.mean_sinr_db == run_deterministic_baseline(three, **kwargs).run.sinr.mean_sinr_db
    assert a.mean_interference_power_dbm != run_deterministic_baseline(one, **kwargs).run.sinr.mean_interference_power_dbm


def test_extended_preset_parameter_validation() -> None:
    with pytest.raises(ValueError):
        SimulationConfig.extended(gnb_count=0)
    with pytest.raises(ValueError):
        SimulationConfig.extended(max_users=0)
    with pytest.raises(ValueError, match="user_count"):
        SimulationConfig.extended(max_users=10, user_count_sweep=(4, 12))
    assert ExperimentConfig().user_count_sweep == (4, 5)
