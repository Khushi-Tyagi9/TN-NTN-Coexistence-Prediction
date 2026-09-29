"""Phase 3 spatial nulling and steering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, isfinite, pi, sin

import numpy as np


def _validate_antenna_count(antenna_count: int) -> None:
    if antenna_count <= 1:
        raise ValueError("antenna_count must be > 1.")


def steering_vector_ula(
    angle_rad: float,
    *,
    antenna_count: int,
    spacing_wavelength: float = 0.5,
) -> np.ndarray:
    """Compute normalized ULA steering vector."""
    _validate_antenna_count(antenna_count)
    if not isfinite(angle_rad):
        raise ValueError("angle_rad must be finite.")
    if spacing_wavelength <= 0:
        raise ValueError("spacing_wavelength must be > 0.")

    element_indices = np.arange(antenna_count, dtype=np.float64)
    phase = 2.0 * pi * spacing_wavelength * element_indices * sin(angle_rad)
    vector = np.exp(1j * phase).astype(np.complex128)
    return vector / np.sqrt(antenna_count)


def zero_forcing_weights(
    *,
    desired_angle_rad: float,
    ntn_angle_rad: float,
    antenna_count: int,
    spacing_wavelength: float = 0.5,
) -> np.ndarray:
    """Design beamforming weights with a null at the NTN direction."""
    steering_desired = steering_vector_ula(
        desired_angle_rad,
        antenna_count=antenna_count,
        spacing_wavelength=spacing_wavelength,
    )
    steering_ntn = steering_vector_ula(
        ntn_angle_rad,
        antenna_count=antenna_count,
        spacing_wavelength=spacing_wavelength,
    )

    projection = (np.vdot(steering_ntn, steering_desired) / np.vdot(steering_ntn, steering_ntn)) * steering_ntn
    candidate = steering_desired - projection
    norm = np.linalg.norm(candidate)
    if norm <= 1e-12:
        raise ValueError("Desired and NTN steering vectors are nearly colinear; cannot form stable null.")
    return candidate / norm


def array_gain(weights: np.ndarray, steering: np.ndarray) -> float:
    """Return linear array gain |w^H a|^2."""
    if weights.ndim != 1 or steering.ndim != 1:
        raise ValueError("weights and steering must be 1D vectors.")
    if weights.shape[0] != steering.shape[0]:
        raise ValueError("weights and steering must have the same length.")
    if np.linalg.norm(weights) <= 0:
        raise ValueError("weights norm must be > 0.")
    return float(np.abs(np.vdot(weights, steering)) ** 2)


def db10(value: float) -> float:
    if value < 0:
        raise ValueError("value must be >= 0.")
    if value == 0:
        return float("-inf")
    return 10.0 * np.log10(value)


@dataclass(frozen=True, slots=True)
class ZFNullingResult:
    """Summary of ZF nulling quality."""

    desired_gain_db: float
    ntn_gain_db: float
    suppression_db: float

    def __post_init__(self) -> None:
        for field_name in ("desired_gain_db", "ntn_gain_db", "suppression_db"):
            value = getattr(self, field_name)
            if value != float("-inf") and not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite or -inf.")


def evaluate_zf_nulling(
    *,
    desired_angle_rad: float,
    ntn_angle_rad: float,
    steering_ntn_angle_rad: float | None = None,
    antenna_count: int,
    spacing_wavelength: float = 0.5,
) -> ZFNullingResult:
    """Build ZF weights and measure desired gain vs NTN suppression."""
    steering_angle = ntn_angle_rad if steering_ntn_angle_rad is None else steering_ntn_angle_rad
    weights = zero_forcing_weights(
        desired_angle_rad=desired_angle_rad,
        ntn_angle_rad=steering_angle,
        antenna_count=antenna_count,
        spacing_wavelength=spacing_wavelength,
    )
    desired_steering = steering_vector_ula(
        desired_angle_rad,
        antenna_count=antenna_count,
        spacing_wavelength=spacing_wavelength,
    )
    ntn_steering = steering_vector_ula(
        ntn_angle_rad,
        antenna_count=antenna_count,
        spacing_wavelength=spacing_wavelength,
    )
    desired_gain_db = db10(array_gain(weights, desired_steering))
    ntn_gain_db = db10(array_gain(weights, ntn_steering))
    suppression_db = desired_gain_db - ntn_gain_db if ntn_gain_db != float("-inf") else float("inf")
    return ZFNullingResult(
        desired_gain_db=float(desired_gain_db),
        ntn_gain_db=float(ntn_gain_db),
        suppression_db=float(suppression_db),
    )


def map_interference_power_to_ntn_angle(
    interference_power: float,
    *,
    power_min: float,
    power_max: float,
    nominal_angle_rad: float,
    max_tracking_offset_rad: float = 0.12,
) -> float:
    """Map predicted/true interference level to an NTN angle offset around nominal pointing."""
    if not all(isfinite(value) for value in (interference_power, power_min, power_max, nominal_angle_rad)):
        raise ValueError("interference_power, power_min, power_max, and nominal_angle_rad must be finite.")
    if max_tracking_offset_rad < 0:
        raise ValueError("max_tracking_offset_rad must be >= 0.")
    if power_min > power_max:
        raise ValueError("power_min cannot exceed power_max.")

    if power_max - power_min <= 1e-12:
        normalized = 0.5
    else:
        normalized = (interference_power - power_min) / (power_max - power_min)
    normalized = float(np.clip(normalized, 0.0, 1.0))

    offset = (2.0 * normalized - 1.0) * max_tracking_offset_rad
    angle = nominal_angle_rad + offset
    return float(np.clip(angle, -pi / 2.0, pi / 2.0))


def estimate_ntn_angle_from_channel(
    predicted_channel: np.ndarray,
    *,
    spacing_wavelength: float = 0.5,
    grid_size: int = 721,
) -> float:
    """Estimate arrival angle from predicted channel via beam-scan matching."""
    channel = np.asarray(predicted_channel, dtype=np.complex128).reshape(-1)
    antenna_count = int(channel.shape[0])
    _validate_antenna_count(antenna_count)
    if spacing_wavelength <= 0:
        raise ValueError("spacing_wavelength must be > 0.")
    if grid_size < 3:
        raise ValueError("grid_size must be >= 3.")
    if not np.all(np.isfinite(channel.real)) or not np.all(np.isfinite(channel.imag)):
        raise ValueError("predicted_channel must contain finite values.")

    candidate_angles = np.linspace(-pi / 2.0, pi / 2.0, grid_size, dtype=np.float64)
    best_index = 0
    best_score = float("-inf")
    for idx, angle in enumerate(candidate_angles):
        steering = steering_vector_ula(
            angle,
            antenna_count=antenna_count,
            spacing_wavelength=spacing_wavelength,
        )
        score = float(np.abs(np.vdot(steering, channel)))
        if score > best_score:
            best_score = score
            best_index = idx
    return float(candidate_angles[best_index])


def estimate_ntn_angle_from_sin(
    sin_theta: float,
) -> float:
    """Small helper when predicted interference maps directly to sin(theta)."""
    if not isfinite(sin_theta):
        raise ValueError("sin_theta must be finite.")
    if sin_theta < -1.0 or sin_theta > 1.0:
        raise ValueError("sin_theta must be in [-1, 1].")
    return float(asin(sin_theta))
