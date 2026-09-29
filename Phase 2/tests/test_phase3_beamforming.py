"""Tests for Phase 3 beamforming and prediction-to-angle mapping."""

import numpy as np

from src.phase3_beamforming import evaluate_zf_nulling, map_interference_power_to_ntn_angle


def test_interference_power_maps_to_bounded_ntn_angle() -> None:
    nominal = 0.35
    angle_low = map_interference_power_to_ntn_angle(
        0.0,
        power_min=0.0,
        power_max=10.0,
        nominal_angle_rad=nominal,
        max_tracking_offset_rad=0.1,
    )
    angle_high = map_interference_power_to_ntn_angle(
        10.0,
        power_min=0.0,
        power_max=10.0,
        nominal_angle_rad=nominal,
        max_tracking_offset_rad=0.1,
    )

    assert np.isclose(angle_low, nominal - 0.1)
    assert np.isclose(angle_high, nominal + 0.1)


def test_zf_nulling_steering_override_models_pointing_error() -> None:
    perfect = evaluate_zf_nulling(
        desired_angle_rad=0.0,
        ntn_angle_rad=0.35,
        antenna_count=64,
    )
    mismatch = evaluate_zf_nulling(
        desired_angle_rad=0.0,
        ntn_angle_rad=0.35,
        steering_ntn_angle_rad=0.42,
        antenna_count=64,
    )

    assert np.isfinite(mismatch.suppression_db)
    assert perfect.suppression_db >= mismatch.suppression_db
