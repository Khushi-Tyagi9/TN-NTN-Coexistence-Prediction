"""Phase 4 metrics: known values, boundaries, linear-vs-dB averaging, and input validation."""

import math

import numpy as np
import pytest

from src.phase4_metrics import (
    coverage_probability,
    dbm_to_mw,
    interference_free_sinr_db,
    interference_reduction_pct,
    interference_reduction_pct_from_dbm,
    throughput_bps,
    throughput_degradation_pct,
)

BANDWIDTH = 20e6


# ------------------------------------------------------------------------------------- coverage
def test_coverage_probability_known_values_and_inclusive_threshold() -> None:
    sinr = np.array([-5.0, 0.0, 10.0, 15.0, 20.0])

    assert coverage_probability(sinr, 15.0) == pytest.approx(0.4)  # 15 counts (>=), 20 counts
    assert coverage_probability(sinr, 14.999) == pytest.approx(0.4)
    assert coverage_probability(sinr, 15.001) == pytest.approx(0.2)
    assert coverage_probability(sinr, -100.0) == 1.0 and coverage_probability(sinr, 100.0) == 0.0
    assert coverage_probability(np.array([[1.0, 20.0], [30.0, -3.0]]), 10.0) == pytest.approx(0.5)  # 2-D (users, slots)
    assert coverage_probability(12.0, 10.0) == 1.0  # scalar


def test_coverage_probability_validation() -> None:
    with pytest.raises(ValueError):
        coverage_probability(np.array([]), 0.0)
    with pytest.raises(ValueError):
        coverage_probability(np.array([1.0, np.nan]), 0.0)
    with pytest.raises(ValueError):
        coverage_probability(np.array([1.0]), float("inf"))


# ----------------------------------------------------------------------------------- throughput
def test_throughput_matches_shannon_known_points() -> None:
    assert throughput_bps(0.0, BANDWIDTH) == pytest.approx(BANDWIDTH * 1.0)  # SINR 1 -> log2(2) = 1 bit/s/Hz
    assert throughput_bps(10.0 * math.log10(3.0), BANDWIDTH) == pytest.approx(BANDWIDTH * 2.0)  # SINR 3 -> 2 bit/s/Hz
    assert throughput_bps(10.0, BANDWIDTH) == pytest.approx(BANDWIDTH * math.log2(11.0))
    assert throughput_bps(-100.0, BANDWIDTH) == pytest.approx(0.0, abs=1.0)  # no rate at hopeless SINR
    assert np.allclose(throughput_bps(np.array([0.0, 10.0]), 1.0), [1.0, math.log2(11.0)])


def test_throughput_is_monotone_in_sinr_and_linear_in_bandwidth() -> None:
    sinr = np.linspace(-20, 40, 61)
    rate = throughput_bps(sinr, BANDWIDTH)

    assert np.all(np.diff(rate) > 0)
    assert np.allclose(throughput_bps(sinr, 2 * BANDWIDTH), 2 * rate)


def test_throughput_validation() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            throughput_bps(10.0, bad)
    with pytest.raises(ValueError):
        throughput_bps(np.array([float("inf")]), BANDWIDTH)


# ---------------------------------------------------------------- degradation vs interference-free
def test_interference_free_reference_is_signal_minus_noise() -> None:
    assert interference_free_sinr_db(-73.0, -94.0) == pytest.approx(21.0)
    assert np.allclose(interference_free_sinr_db(np.array([-70.0, -80.0]), -94.0), [24.0, 14.0])
    with pytest.raises(ValueError):
        interference_free_sinr_db(-73.0, float("nan"))


def test_throughput_degradation_zero_when_nothing_is_lost() -> None:
    sinr = np.array([5.0, 12.0, 20.0])
    assert throughput_degradation_pct(sinr, sinr, BANDWIDTH) == pytest.approx(0.0)


def test_throughput_degradation_known_value() -> None:
    # actual SINR 0 dB -> 1 bit/s/Hz; reference SINR 10*log10(3) dB -> 2 bit/s/Hz => 50% lost
    assert throughput_degradation_pct(0.0, 10.0 * math.log10(3.0), BANDWIDTH) == pytest.approx(50.0)
    # interference wipes the link out: ~100% lost
    assert throughput_degradation_pct(-60.0, 20.0, BANDWIDTH) == pytest.approx(100.0, abs=0.01)
    # a better-than-reference link is reported as negative degradation (not clipped)
    assert throughput_degradation_pct(20.0, 10.0, BANDWIDTH) < 0.0


def test_throughput_degradation_uses_mean_rates_not_mean_sinr() -> None:
    actual = np.array([-10.0, 30.0])
    reference = np.array([30.0, 30.0])
    expected = 100.0 * (1.0 - np.mean(throughput_bps(actual, BANDWIDTH)) / np.mean(throughput_bps(reference, BANDWIDTH)))

    assert throughput_degradation_pct(actual, reference, BANDWIDTH) == pytest.approx(expected)
    assert expected > 40.0  # one bad slot costs far more than the dB average would suggest


def test_throughput_degradation_validation() -> None:
    with pytest.raises(ValueError, match="same shape"):
        throughput_degradation_pct(np.array([1.0, 2.0]), np.array([1.0]), BANDWIDTH)
    with pytest.raises(ValueError, match="Reference throughput"):
        throughput_degradation_pct(0.0, -4000.0, BANDWIDTH)  # 10**(-400) underflows, so the reference rate is exactly 0


# -------------------------------------------------------------------------- interference reduction
def test_interference_reduction_known_values() -> None:
    assert interference_reduction_pct(1.0, 0.1) == pytest.approx(90.0)  # 10 dB -> 90%
    assert interference_reduction_pct(1.0, 0.0) == pytest.approx(100.0)
    assert interference_reduction_pct(1.0, 1.0) == pytest.approx(0.0)
    assert interference_reduction_pct(1.0, 2.0) == pytest.approx(-100.0)  # interference doubled
    assert interference_reduction_pct(np.array([2.0, 4.0]), np.array([0.2, 0.4])) == pytest.approx(90.0)


def test_interference_reduction_is_computed_in_linear_power_not_db() -> None:
    before_dbm = np.array([-60.0, -90.0])
    after_dbm = np.array([-70.0, -90.0])  # only the strong sample dropped by 10 dB

    linear = interference_reduction_pct_from_dbm(before_dbm, after_dbm)
    expected = 100.0 * (1.0 - (1e-7 + 1e-9) / (1e-6 + 1e-9))
    assert linear == pytest.approx(expected)
    assert linear == pytest.approx(89.91, abs=0.01)  # dominated by the strong sample
    assert linear > 80.0  # whereas averaging dB first (-75 vs -80 dBm = 5 dB) would claim only ~68%
    assert interference_reduction_pct_from_dbm(-60.0, -70.0) == pytest.approx(90.0)


def test_ratio_of_means_versus_mean_of_ratios() -> None:
    before = np.array([1.0, 100.0])
    after = np.array([0.5, 10.0])  # per-slot reductions: 50% and 90%

    assert interference_reduction_pct(before, after) == pytest.approx(100 * (1 - 10.5 / 101.0))  # ratio of means
    assert interference_reduction_pct(before, after, per_sample=True) == pytest.approx(70.0)  # mean of ratios


def test_interference_reduction_validation() -> None:
    with pytest.raises(ValueError, match="same shape"):
        interference_reduction_pct(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="> 0"):
        interference_reduction_pct(0.0, 0.0)
    with pytest.raises(ValueError, match=">= 0"):
        interference_reduction_pct(1.0, -0.1)
    with pytest.raises(ValueError):
        interference_reduction_pct(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        interference_reduction_pct(1.0, float("nan"))


def test_dbm_to_mw() -> None:
    assert dbm_to_mw(0.0) == pytest.approx(1.0)
    assert dbm_to_mw(30.0) == pytest.approx(1000.0)
    assert np.allclose(dbm_to_mw(np.array([-10.0, 10.0])), [0.1, 10.0])
