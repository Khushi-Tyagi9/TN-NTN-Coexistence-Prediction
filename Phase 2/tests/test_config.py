"""Tests for typed, validated TN-NTN baseline configuration."""

import pytest

from src.config import ACIRSweepConfig, HAPConfig, RadioConfig, SimulationConfig, TNDeploymentConfig, UserConfig


def test_simulation_config_defaults() -> None:
    """SimulationConfig should expose realistic baseline defaults."""
    config = SimulationConfig()
    assert config.reproducibility.seed == 42
    assert config.hap.hap_count == 1
    assert config.tn_deployment.gnb_count == 1
    assert config.users.min_count == 4
    assert config.users.default_count == 5
    assert config.users.max_count == 5
    assert config.acir_sweep_db == (20.0, 25.0, 30.0, 35.0, 40.0)
    assert set(config.scenario.presets) == {"urban", "rural"}


def test_validation_fails_when_baseline_hap_count_is_zero() -> None:
    """Baseline realism should require at least one HAP."""
    with pytest.raises(ValueError, match="hap_count >= 1"):
        SimulationConfig(hap=HAPConfig(hap_count=0))


def test_validation_fails_when_baseline_user_bounds_exceed_range() -> None:
    """Baseline realism should keep users in [4, 5]."""
    with pytest.raises(ValueError, match="4 <= user_count <= 5"):
        SimulationConfig(users=UserConfig(min_count=4, default_count=5, max_count=6))


def test_validation_fails_when_baseline_gnb_count_is_not_one() -> None:
    """Baseline realism should enforce exactly one gNB tower."""
    with pytest.raises(ValueError, match="gnb_count == 1"):
        SimulationConfig(tn_deployment=TNDeploymentConfig(gnb_count=2))


def test_validation_fails_when_acir_bounds_are_invalid() -> None:
    """ACIR bounds must remain ordered and valid."""
    with pytest.raises(ValueError, match="ACIR bounds"):
        ACIRSweepConfig(min_db=35.0, max_db=30.0)


def test_validation_fails_when_acir_sweep_is_non_monotonic() -> None:
    """ACIR sweep points must be strictly increasing."""
    with pytest.raises(ValueError, match="strictly increasing"):
        SimulationConfig(
            radio=RadioConfig(
                acir_sweep=ACIRSweepConfig(points_db=(20.0, 30.0, 25.0), min_db=20.0, max_db=40.0)
            )
        )

