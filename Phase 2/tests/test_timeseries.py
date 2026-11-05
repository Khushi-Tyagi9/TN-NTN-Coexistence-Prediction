"""Tests for the time-stepped simulation."""

import numpy as np
import pytest

from src.config import SimulationConfig
from src.interference import evaluate_scenario_sinr
from src.simulation import generate_network_scenario
from src.timeseries import TimeSeriesConfig, simulate_time_series


def _static_config(n_slots: int = 20) -> TimeSeriesConfig:
    """Every random/dynamic term off -> must reproduce the baseline snapshot."""
    return TimeSeriesConfig(
        n_slots=n_slots,
        satellite_speed_km_s=0.0,
        user_speed_m_s=0.0,
        shadowing_sigma_db_urban=0.0,
        shadowing_sigma_db_rural=0.0,
        fast_fading_enabled=False,
        load_sigma_db=0.0,
        signal_jitter_sigma_db=0.0,
    )


@pytest.mark.parametrize("scenario_name", ["urban", "rural"])
def test_static_limit_matches_baseline_snapshot(scenario_name: str) -> None:
    config = SimulationConfig()
    run = simulate_time_series(
        config, _static_config(), scenario_name=scenario_name, user_count=5, acir_db=30.0, seed=7
    )
    scenario = generate_network_scenario(config, scenario_name=scenario_name, user_count=5, seed=7)
    baseline = evaluate_scenario_sinr(scenario, config, acir_db=30.0)

    for user_index, metrics in enumerate(baseline.user_metrics):
        assert np.allclose(run.interference_dbm[user_index], metrics.interference_power_dbm, atol=1e-6)
        assert np.allclose(run.signal_dbm[user_index], metrics.signal_power_dbm, atol=1e-6)
        assert np.allclose(run.sinr_db[user_index], metrics.sinr_db, atol=1e-6)


def test_output_shapes_and_finiteness() -> None:
    run = simulate_time_series(
        SimulationConfig(), TimeSeriesConfig(n_slots=120), scenario_name="urban", user_count=4, acir_db=25.0, seed=1
    )
    assert run.interference_dbm.shape == (4, 120)
    assert run.features().shape == (4, 120, 3)
    assert np.all(np.isfinite(run.features()))


def test_same_seed_is_deterministic_and_different_seed_differs() -> None:
    kwargs = dict(scenario_name="rural", user_count=5, acir_db=35.0)
    a = simulate_time_series(SimulationConfig(), seed=3, **kwargs)
    b = simulate_time_series(SimulationConfig(), seed=3, **kwargs)
    c = simulate_time_series(SimulationConfig(), seed=4, **kwargs)
    assert np.array_equal(a.interference_dbm, b.interference_dbm)
    assert not np.array_equal(a.interference_dbm, c.interference_dbm)


def test_interference_actually_varies_over_time() -> None:
    run = simulate_time_series(
        SimulationConfig(), scenario_name="urban", user_count=5, acir_db=30.0, seed=11
    )
    assert float(np.mean(np.std(run.interference_dbm, axis=1))) > 2.0  # dB


def test_satellite_flyover_signal_peaks_at_centre() -> None:
    ts = TimeSeriesConfig(signal_jitter_sigma_db=0.0, user_speed_m_s=0.0)
    run = simulate_time_series(SimulationConfig(), ts, scenario_name="urban", user_count=4, acir_db=30.0, seed=5)
    mean_signal = run.signal_dbm.mean(axis=0)
    centre = mean_signal[len(mean_signal) // 2]
    assert centre > mean_signal[0] + 3.0
    assert centre > mean_signal[-1] + 3.0


def test_higher_acir_lowers_interference() -> None:
    low = simulate_time_series(SimulationConfig(), scenario_name="urban", user_count=5, acir_db=20.0, seed=9)
    high = simulate_time_series(SimulationConfig(), scenario_name="urban", user_count=5, acir_db=40.0, seed=9)
    assert np.mean(low.interference_dbm - high.interference_dbm) == pytest.approx(20.0, abs=1e-6)


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        TimeSeriesConfig(n_slots=5)
    with pytest.raises(ValueError):
        TimeSeriesConfig(shadowing_correlation=1.0)


# --- audit M5: Doppler-derived fast-fading correlation and local-mean interference --------------

from math import pi  # noqa: E402

from scipy.special import j0  # noqa: E402

from src.timeseries import SPEED_OF_LIGHT_M_S, _ar1_gaussian  # noqa: E402

CARRIER_HZ = 2.0e9


def test_fast_fading_rho_is_derived_from_doppler_by_default() -> None:
    ts = TimeSeriesConfig()
    assert ts.fast_fading_correlation is None
    doppler = 1.5 * CARRIER_HZ / SPEED_OF_LIGHT_M_S
    assert ts.doppler_hz(CARRIER_HZ) == pytest.approx(doppler)
    assert ts.fast_fading_rho(CARRIER_HZ) == pytest.approx(float(j0(2 * pi * doppler * 0.5)))
    # 500 ms slots are ~12 coherence times at 2 GHz / 1.5 m/s: fading is nearly independent per slot
    assert abs(ts.fast_fading_rho(CARRIER_HZ)) < 0.2


def test_fast_fading_rho_rises_for_short_slots_and_override_wins() -> None:
    short = TimeSeriesConfig(slot_duration_s=0.01)
    assert 0.85 < short.fast_fading_rho(CARRIER_HZ) < 0.95
    assert TimeSeriesConfig(fast_fading_correlation=0.3).fast_fading_rho(CARRIER_HZ) == 0.3
    assert TimeSeriesConfig(user_speed_m_s=0.0).fast_fading_rho(CARRIER_HZ) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        TimeSeriesConfig(fast_fading_correlation=1.0)
    with pytest.raises(ValueError):
        TimeSeriesConfig().doppler_hz(0.0)


def test_ar1_generator_reproduces_requested_lag1_correlation() -> None:
    rng = np.random.default_rng(0)
    for rho in (0.1, 0.7, -0.3):
        series = _ar1_gaussian(rng, (1, 20000), rho, 1.0)[0]
        assert np.corrcoef(series[:-1], series[1:])[0, 1] == pytest.approx(rho, abs=0.03)
        assert series.std() == pytest.approx(1.0, abs=0.05)


def test_local_mean_equals_instantaneous_without_fast_fading() -> None:
    run = simulate_time_series(
        SimulationConfig(), _static_config(), scenario_name="urban", user_count=5, acir_db=30.0, seed=7
    )
    assert np.array_equal(run.interference_local_dbm, run.interference_dbm)


def test_local_mean_removes_fast_fading_but_keeps_the_mean_power() -> None:
    # Only fast fading is random here, so instantaneous/local power ratio is a unit-mean Rayleigh power.
    ts = TimeSeriesConfig(
        n_slots=600, shadowing_sigma_db_urban=0.0, load_sigma_db=0.0, user_speed_m_s=0.0,
        fast_fading_correlation=0.0,
    )
    run = simulate_time_series(SimulationConfig(), ts, scenario_name="urban", user_count=5, acir_db=30.0, seed=2)

    ratio = 10.0 ** ((run.interference_dbm - run.interference_local_dbm) / 10.0)
    assert ratio.mean() == pytest.approx(1.0, abs=0.08)  # unit-mean fading power
    assert np.std(run.interference_dbm - run.interference_local_dbm) > 4.0  # Rayleigh in dB: ~5.6 dB
    assert np.std(np.diff(run.interference_local_dbm, axis=1)) < 0.5 * np.std(np.diff(run.interference_dbm, axis=1))


def test_local_mean_series_is_deterministic_for_a_seed() -> None:
    kwargs = dict(scenario_name="rural", user_count=4, acir_db=25.0, seed=8)
    a = simulate_time_series(SimulationConfig(), **kwargs)
    b = simulate_time_series(SimulationConfig(), **kwargs)
    assert np.array_equal(a.interference_local_dbm, b.interference_local_dbm)
