"""Starter tests for the simulation scaffolding module."""

import pytest

from src.config import ScenarioConfig, SimulationConfig
from src.entities import HAPPositionModel, Position3D, horizontal_distance_km
from src.simulation import (
    SimulationContext,
    SimulationRunner,
    generate_network_scenario,
    run_monte_carlo_baseline,
)


def test_simulation_runner_exposes_context() -> None:
    """SimulationRunner should retain the context passed at creation."""
    context = SimulationContext(config=SimulationConfig())
    runner = SimulationRunner(context)
    assert runner.context is context


def test_deterministic_single_run_returns_expected_shape() -> None:
    """Deterministic baseline entrypoint should return typed run and aggregate payloads."""
    runner = SimulationRunner(SimulationContext(config=SimulationConfig()))
    result = runner.run_deterministic(scenario_name="urban", user_count=4, acir_db=30.0, seed=123)

    assert result.run.run_id == "urban-u4-acir30-r1"
    assert result.run.parameters.scenario_name == "urban"
    assert result.run.parameters.user_count == 4
    assert result.run.parameters.acir_db == 30.0
    assert result.run.parameters.seed == 123
    assert result.aggregate.run_count == 1
    assert len(result.run.sinr.user_metrics) == 4
    assert [record.event for record in result.metadata_records] == [
        "deterministic_run_started",
        "deterministic_run_completed",
    ]


def test_monte_carlo_run_count_respects_requested_runs() -> None:
    """Monte Carlo loop should produce the configured number of runs per sweep point."""
    result = run_monte_carlo_baseline(
        SimulationConfig(),
        monte_carlo_runs=3,
        scenario_order=("urban",),
        user_count_sweep=(4,),
        acir_sweep_db=(20.0, 30.0),
        base_seed=100,
    )

    assert len(result.sweep_results) == 2
    assert all(len(sweep.runs) == 3 for sweep in result.sweep_results)
    assert all(sweep.aggregate.run_count == 3 for sweep in result.sweep_results)


def test_acir_sweep_results_preserve_configured_order() -> None:
    """Sweep output ordering should match the supplied ACIR sweep ordering."""
    config = SimulationConfig()
    result = run_monte_carlo_baseline(
        config,
        monte_carlo_runs=1,
        scenario_order=("urban",),
        user_count_sweep=(4,),
        acir_sweep_db=config.acir_sweep_db,
    )

    assert [sweep.acir_db for sweep in result.sweep_results] == list(config.acir_sweep_db)


def test_runner_results_are_reproducible_for_same_seed_and_config() -> None:
    """Deterministic seed + config should produce identical Monte Carlo outputs."""
    config = SimulationConfig()
    first = run_monte_carlo_baseline(
        config,
        monte_carlo_runs=2,
        scenario_order=("urban",),
        user_count_sweep=(4,),
        acir_sweep_db=(25.0,),
        base_seed=77,
    )
    second = run_monte_carlo_baseline(
        config,
        monte_carlo_runs=2,
        scenario_order=("urban",),
        user_count_sweep=(4,),
        acir_sweep_db=(25.0,),
        base_seed=77,
    )
    assert first == second


def test_generate_network_scenario_is_reproducible_for_same_seed() -> None:
    """Scenario generation should be deterministic for the configured RNG seed."""
    config = SimulationConfig()
    first = generate_network_scenario(config)
    second = generate_network_scenario(config)
    assert first == second


def test_generate_network_scenario_supports_active_preset_and_override() -> None:
    """Generation should use active preset by default and allow explicit override."""
    config = SimulationConfig(scenario=ScenarioConfig(active_preset="rural"))
    rural = generate_network_scenario(config)
    urban = generate_network_scenario(config, scenario_name="urban")
    anchor = Position3D.from_xy_km(*config.hap.anchor_xy_km)

    rural_max_user_distance = max(horizontal_distance_km(user, anchor) for user in rural.users)
    urban_max_user_distance = max(horizontal_distance_km(user, anchor) for user in urban.users)
    assert rural_max_user_distance > urban_max_user_distance


def test_generate_network_scenario_satisfies_baseline_count_constraints() -> None:
    """Generated scenarios should include required baseline node counts and models."""
    scenario = generate_network_scenario(SimulationConfig())
    assert len(scenario.satellites) >= 1
    assert len(scenario.haps) >= 1
    assert len(scenario.gnbs) == 1
    assert 4 <= len(scenario.users) <= 5
    assert all(hap.position_model is HAPPositionModel.QUASI_STATIC for hap in scenario.haps)


def test_generate_network_scenario_rejects_invalid_scenario_name() -> None:
    """Unknown scenario names should fail fast."""
    with pytest.raises(ValueError, match="Unknown scenario preset"):
        generate_network_scenario(SimulationConfig(), scenario_name="suburban")

