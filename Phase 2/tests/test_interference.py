"""Tests for baseline interference aggregation and SINR outputs."""

from __future__ import annotations

from math import isfinite

import pytest

from src.config import SimulationConfig
from src.entities import NTNUserNode, NetworkScenario, Position3D, SatelliteNode, TNGNBNode
from src.interference import InterferenceModel, compute_noise_power_dbm, evaluate_scenario_sinr
from src.simulation import generate_network_scenario


def _build_manual_scenario() -> NetworkScenario:
    satellite = SatelliteNode(
        entity_id="sat-1",
        position=Position3D.from_xy_km(0.0, 0.0, altitude_km=550.0),
        leo_altitude_km=550.0,
        tx_power_dbw=20.0,
        tx_gain_dbi=30.0,
    )
    gnb = TNGNBNode(entity_id="gnb-1", position=Position3D.from_xy_km(0.6, 0.0, altitude_km=0.0), tx_power_dbm=46.0)
    near_user = NTNUserNode(entity_id="ue-near", position=Position3D.from_xy_km(0.05, 0.0, altitude_km=0.0))
    far_user = NTNUserNode(entity_id="ue-far", position=Position3D.from_xy_km(80.0, 0.0, altitude_km=0.0))
    return NetworkScenario(
        satellites=(satellite,),
        gnbs=(gnb,),
        users=(near_user, far_user),
        enforce_baseline_realism=False,
    )


def test_better_acir_reduces_interference_and_improves_sinr() -> None:
    """Higher ACIR should reduce leaked TN interference and improve SINR."""
    scenario = _build_manual_scenario()
    config = SimulationConfig()

    low_acir_summary = evaluate_scenario_sinr(scenario, config, acir_db=20.0)
    high_acir_summary = evaluate_scenario_sinr(scenario, config, acir_db=40.0)

    low_acir_user = low_acir_summary.user_metrics[0]
    high_acir_user = high_acir_summary.user_metrics[0]

    assert high_acir_user.interference_power_mw < low_acir_user.interference_power_mw
    assert high_acir_user.sinr_db > low_acir_user.sinr_db


def test_farther_user_has_lower_received_satellite_power() -> None:
    """Satellite received signal should decrease with longer user distance."""
    scenario = _build_manual_scenario()
    summary = evaluate_scenario_sinr(scenario, SimulationConfig(), acir_db=30.0)
    by_user_id = {metric.user_id: metric for metric in summary.user_metrics}

    assert by_user_id["ue-near"].signal_power_dbm > by_user_id["ue-far"].signal_power_dbm
    assert by_user_id["ue-near"].signal_path_loss_db < by_user_id["ue-far"].signal_path_loss_db


def test_noise_power_matches_bandwidth_and_noise_figure_formula() -> None:
    """Thermal noise helper should follow -174 dBm/Hz + 10log10(BW) + NF."""
    noise_power_dbm = compute_noise_power_dbm(20.0e6, 7.0)
    assert noise_power_dbm == pytest.approx(-93.9897, abs=1e-3)


def test_generated_scenario_end_to_end_sinr_outputs_are_finite() -> None:
    """Scenario generator output should run through baseline SINR model end-to-end."""
    config = SimulationConfig()
    scenario = generate_network_scenario(config)
    summary = InterferenceModel(scenario=scenario, config=config, acir_db=30.0).evaluate()

    assert len(summary.user_metrics) == len(scenario.users)
    assert all(isfinite(metric.sinr_db) for metric in summary.user_metrics)
    assert summary.min_sinr_db <= summary.mean_sinr_db <= summary.max_sinr_db


def test_sinr_evaluation_rejects_invalid_acir() -> None:
    """ACIR must remain strictly positive."""
    with pytest.raises(ValueError, match="acir_db must be > 0"):
        evaluate_scenario_sinr(_build_manual_scenario(), SimulationConfig(), acir_db=0.0)

