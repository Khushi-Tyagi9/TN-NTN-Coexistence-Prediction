"""Phase 4 system metrics: coverage, throughput, throughput degradation, interference reduction.

Conventions: SINR/SNR in dB, powers in dBm or mW as named, bandwidth in Hz, throughput in bit/s.
Throughput is the Shannon rate ``B * log2(1 + SINR_linear)``. Reductions and degradations are expressed
on LINEAR power / rate averages (ratio of means), never on dB values.
"""

from __future__ import annotations

import numpy as np


def _finite_array(values: np.ndarray | float, name: str, *, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values only.")
    return array


def dbm_to_mw(power_dbm: np.ndarray | float) -> np.ndarray:
    """dBm -> mW (element-wise)."""
    return 10.0 ** (_finite_array(power_dbm, "power_dbm") / 10.0)


def coverage_probability(sinr_db: np.ndarray | float, threshold_db: float) -> float:
    """Fraction of (user, slot) samples with ``SINR >= threshold_db`` (inclusive)."""
    if not np.isfinite(threshold_db):
        raise ValueError("threshold_db must be finite.")
    sinr = _finite_array(sinr_db, "sinr_db")
    return float(np.mean(sinr >= threshold_db))


def throughput_bps(sinr_db: np.ndarray | float, bandwidth_hz: float) -> np.ndarray:
    """Shannon throughput ``bandwidth * log2(1 + SINR_linear)`` in bit/s, element-wise."""
    if not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be finite and > 0.")
    sinr_linear = 10.0 ** (_finite_array(sinr_db, "sinr_db") / 10.0)
    return bandwidth_hz * np.log1p(sinr_linear) / np.log(2.0)


def interference_free_sinr_db(signal_dbm: np.ndarray | float, noise_dbm: float) -> np.ndarray:
    """Reference SINR with all interference removed (SNR): ``signal - noise`` in dB."""
    if not np.isfinite(noise_dbm):
        raise ValueError("noise_dbm must be finite.")
    return _finite_array(signal_dbm, "signal_dbm") - noise_dbm


def throughput_degradation_pct(
    sinr_db: np.ndarray | float,
    reference_sinr_db: np.ndarray | float,
    bandwidth_hz: float,
) -> float:
    """Percent throughput lost versus an interference-free reference.

    ``100 * (1 - mean(T_actual) / mean(T_reference))`` with ``T = B*log2(1 + SINR)``. The reference is
    typically :func:`interference_free_sinr_db`. A negative value means the actual link beats the reference.
    """
    actual = throughput_bps(sinr_db, bandwidth_hz)
    reference = throughput_bps(reference_sinr_db, bandwidth_hz)
    if actual.shape != reference.shape:
        raise ValueError("sinr_db and reference_sinr_db must have the same shape.")
    reference_mean = float(np.mean(reference))
    if reference_mean <= 0:
        raise ValueError("Reference throughput must be > 0.")
    return float(100.0 * (1.0 - np.mean(actual) / reference_mean))


def interference_reduction_pct(
    power_before_mw: np.ndarray | float,
    power_after_mw: np.ndarray | float,
    *,
    per_sample: bool = False,
) -> float:
    """``100 * (1 - P_after / P_before)`` in LINEAR power (mW).

    Default: ratio of the mean powers over all samples (energy-weighted). With ``per_sample=True`` it is
    the mean of the per-sample reductions instead (each slot weighted equally). 100 means all interference
    removed, 0 means no change, negative means interference increased.
    """
    before = _finite_array(power_before_mw, "power_before_mw")
    after = _finite_array(power_after_mw, "power_after_mw")
    if before.shape != after.shape:
        raise ValueError("power_before_mw and power_after_mw must have the same shape.")
    if np.any(before <= 0):
        raise ValueError("power_before_mw must be > 0 (no reduction is defined for zero interference).")
    if np.any(after < 0):
        raise ValueError("power_after_mw must be >= 0.")
    if per_sample:
        return float(100.0 * np.mean(1.0 - after / before))
    return float(100.0 * (1.0 - np.mean(after) / np.mean(before)))


def interference_reduction_pct_from_dbm(
    power_before_dbm: np.ndarray | float,
    power_after_dbm: np.ndarray | float,
    *,
    per_sample: bool = False,
) -> float:
    """Same as :func:`interference_reduction_pct` for inputs in dBm (converted to linear first)."""
    return interference_reduction_pct(dbm_to_mw(power_before_dbm), dbm_to_mw(power_after_dbm), per_sample=per_sample)
