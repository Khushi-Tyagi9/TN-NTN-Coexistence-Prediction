"""Tests for deterministic baseline channel/path-loss helpers."""

import pytest

from src.channels import FreeSpaceChannelModel, baseline_path_loss, free_space_path_loss_db, total_path_loss_db


def test_free_space_path_loss_increases_with_distance() -> None:
    """FSPL should increase monotonically with distance (fixed frequency)."""
    near_loss_db = free_space_path_loss_db(distance_km=1.0, carrier_frequency_hz=2.0e9)
    far_loss_db = free_space_path_loss_db(distance_km=10.0, carrier_frequency_hz=2.0e9)
    assert far_loss_db > near_loss_db


def test_free_space_path_loss_increases_with_frequency() -> None:
    """FSPL should increase monotonically with frequency (fixed distance)."""
    low_frequency_loss_db = free_space_path_loss_db(distance_km=5.0, carrier_frequency_hz=1.0e9)
    high_frequency_loss_db = free_space_path_loss_db(distance_km=5.0, carrier_frequency_hz=2.0e9)
    assert high_frequency_loss_db > low_frequency_loss_db


def test_total_path_loss_includes_additional_loss_term() -> None:
    """Total path loss should include deterministic additional loss directly."""
    breakdown = baseline_path_loss(distance_km=2.0, carrier_frequency_hz=2.0e9, additional_loss_db=5.0)
    assert breakdown.total_loss_db == pytest.approx(breakdown.free_space_loss_db + 5.0)
    assert total_path_loss_db(2.0, 2.0e9, additional_loss_db=5.0) == pytest.approx(breakdown.total_loss_db)


def test_free_space_channel_model_evaluates_to_channel_sample() -> None:
    """Concrete channel model should satisfy the abstract channel interface."""
    model = FreeSpaceChannelModel(distance_km=3.0, carrier_frequency_hz=2.0e9, additional_loss_db=1.5, fading_db=0.2)
    sample = model.evaluate()
    assert sample.path_loss_db == pytest.approx(total_path_loss_db(3.0, 2.0e9, additional_loss_db=1.5))
    assert sample.fading_db == pytest.approx(0.2)


def test_free_space_path_loss_rejects_non_positive_inputs() -> None:
    """Distance and frequency must be positive to keep units physically valid."""
    with pytest.raises(ValueError, match="distance_km must be > 0"):
        free_space_path_loss_db(distance_km=0.0, carrier_frequency_hz=2.0e9)

    with pytest.raises(ValueError, match="carrier_frequency_hz must be > 0"):
        free_space_path_loss_db(distance_km=1.0, carrier_frequency_hz=0.0)

