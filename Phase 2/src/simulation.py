"""Simulation orchestration scaffolding for TN-NTN coexistence runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin, sqrt
from random import Random
from typing import Any

from .config import ScenarioPreset, SimulationConfig
from .entities import HAPNode, HAPPositionModel, NTNUserNode, NetworkScenario, Position3D, SatelliteNode, TNGNBNode
from .interference import ScenarioSINRSummary, dbm_to_mw, evaluate_scenario_sinr, mw_to_dbm


def _sample_xy_in_disk(
    rng: Random,
    *,
    radius_km: float,
    center_xy_km: tuple[float, float],
) -> tuple[float, float]:
    angle_rad = rng.uniform(0.0, 2.0 * pi)
    radial_distance_km = sqrt(rng.random()) * radius_km
    center_x_km, center_y_km = center_xy_km
    return (
        center_x_km + radial_distance_km * cos(angle_rad),
        center_y_km + radial_distance_km * sin(angle_rad),
    )


def _resolve_preset(config: SimulationConfig, scenario_name: str | None) -> ScenarioPreset:
    if scenario_name is None:
        return config.get_active_scenario()
    return config.scenario.get_preset(scenario_name)


def _resolve_counts(config: SimulationConfig, user_count: int | None) -> tuple[int, int, int]:
    hap_count = config.hap.hap_count
    if hap_count < 1:
        raise ValueError("Scenario generation requires hap_count >= 1.")

    gnb_count = config.tn_deployment.gnb_count
    if gnb_count < 1:
        raise ValueError("Scenario generation requires gnb_count >= 1.")
    if config.experiment.enforce_baseline_realism and gnb_count != 1:
        raise ValueError("Scenario generation requires gnb_count == 1 for baseline realism.")

    resolved_user_count = config.resolve_user_count(user_count)
    if config.experiment.enforce_baseline_realism and not (4 <= resolved_user_count <= 5):
        raise ValueError("Scenario generation requires user_count in [4, 5].")

    return hap_count, gnb_count, resolved_user_count


def generate_network_scenario(
    config: SimulationConfig,
    *,
    scenario_name: str | None = None,
    user_count: int | None = None,
    seed: int | None = None,
) -> NetworkScenario:
    """Generate a deterministic TN/NTN baseline scenario from typed config."""
    preset = _resolve_preset(config, scenario_name)
    hap_count, gnb_count, resolved_user_count = _resolve_counts(config, user_count)

    resolved_seed = config.reproducibility.seed if seed is None else seed
    if resolved_seed < 0:
        raise ValueError("seed must be >= 0.")
    rng = Random(resolved_seed)
    anchor_xy_km = config.hap.anchor_xy_km

    satellites = (
        SatelliteNode(
            entity_id="sat-1",
            position=Position3D.from_xy_km(
                anchor_xy_km[0],
                anchor_xy_km[1],
                altitude_km=config.satellite.leo_altitude_km,
            ),
            leo_altitude_km=config.satellite.leo_altitude_km,
            tx_power_dbw=config.satellite.tx_power_dbw,
            tx_gain_dbi=config.satellite.tx_gain_dbi,
            rx_gain_dbi=config.satellite.rx_gain_dbi,
        ),
    )

    hap_drift_radius_km = config.hap.drift_radius_m / 1000.0
    haps = tuple(
        HAPNode(
            entity_id=f"hap-{index + 1}",
            position=Position3D.from_xy_km(
                *_sample_xy_in_disk(
                    rng,
                    radius_km=hap_drift_radius_km,
                    center_xy_km=anchor_xy_km,
                ),
                altitude_km=config.hap.altitude_km,
            ),
            position_model=HAPPositionModel.QUASI_STATIC,
            anchor_xy_km=anchor_xy_km,
            drift_radius_m=config.hap.drift_radius_m,
        )
        for index in range(hap_count)
    )

    deployment_radius_km = sqrt(gnb_count / (pi * preset.gnb_density_per_km2))
    gnbs = tuple(
        TNGNBNode(
            entity_id=f"gnb-{index + 1}",
            position=Position3D.from_xy_km(
                *_sample_xy_in_disk(
                    rng,
                    radius_km=deployment_radius_km,
                    center_xy_km=anchor_xy_km,
                ),
                altitude_km=0.0,
            ),
            tx_power_dbm=config.tn_deployment.gnb_tx_power_dbm,
        )
        for index in range(gnb_count)
    )

    users = tuple(
        NTNUserNode(
            entity_id=f"ue-{index + 1}",
            position=Position3D.from_xy_km(
                *_sample_xy_in_disk(
                    rng,
                    radius_km=preset.user_spread_km,
                    center_xy_km=anchor_xy_km,
                ),
                altitude_km=0.0,
            ),
        )
        for index in range(resolved_user_count)
    )

    return NetworkScenario(
        satellites=satellites,
        haps=haps,
        gnbs=gnbs,
        users=users,
        enforce_baseline_realism=config.experiment.enforce_baseline_realism,
    )


@dataclass(frozen=True, slots=True)
class SimulationLogRecord:
    """Structured in-memory log event for simulation orchestration."""

    event: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event.strip():
            raise ValueError("event cannot be empty.")


@dataclass(frozen=True, slots=True)
class RunParameters:
    """Execution parameters for one baseline run."""

    scenario_name: str
    user_count: int
    acir_db: float
    run_index: int
    seed: int

    def __post_init__(self) -> None:
        if not self.scenario_name.strip():
            raise ValueError("scenario_name cannot be empty.")
        if self.user_count <= 0:
            raise ValueError("user_count must be > 0.")
        if self.acir_db <= 0:
            raise ValueError("acir_db must be > 0.")
        if self.run_index < 0:
            raise ValueError("run_index must be >= 0.")
        if self.seed < 0:
            raise ValueError("seed must be >= 0.")


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    """Per-run baseline result and identifiers."""

    run_id: str
    parameters: RunParameters
    sinr: ScenarioSINRSummary
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id cannot be empty.")


@dataclass(frozen=True, slots=True)
class AggregatedSINRMetrics:
    """Aggregated SINR metrics across one or more runs."""

    run_count: int
    mean_sinr_db: float
    min_sinr_db: float
    max_sinr_db: float
    mean_signal_power_dbm: float
    mean_interference_power_dbm: float

    def __post_init__(self) -> None:
        if self.run_count <= 0:
            raise ValueError("run_count must be > 0.")


@dataclass(frozen=True, slots=True)
class SweepResult:
    """All runs and aggregate metrics for one (scenario, users, ACIR) point."""

    scenario_name: str
    user_count: int
    acir_db: float
    runs: tuple[BaselineRunResult, ...]
    aggregate: AggregatedSINRMetrics

    def __post_init__(self) -> None:
        if not self.scenario_name.strip():
            raise ValueError("scenario_name cannot be empty.")
        if self.user_count <= 0:
            raise ValueError("user_count must be > 0.")
        if self.acir_db <= 0:
            raise ValueError("acir_db must be > 0.")
        if not self.runs:
            raise ValueError("runs cannot be empty.")


@dataclass(frozen=True, slots=True)
class DeterministicSimulationResult:
    """Result payload for a single deterministic baseline run."""

    run: BaselineRunResult
    aggregate: AggregatedSINRMetrics
    metadata_records: tuple[SimulationLogRecord, ...]


@dataclass(frozen=True, slots=True)
class MonteCarloSimulationResult:
    """Result payload for Monte Carlo baseline sweeps."""

    base_seed: int
    monte_carlo_runs: int
    scenario_order: tuple[str, ...]
    user_count_sweep: tuple[int, ...]
    acir_sweep_db: tuple[float, ...]
    sweep_results: tuple[SweepResult, ...]
    metadata_records: tuple[SimulationLogRecord, ...]

    def __post_init__(self) -> None:
        if self.base_seed < 0:
            raise ValueError("base_seed must be >= 0.")
        if self.monte_carlo_runs <= 0:
            raise ValueError("monte_carlo_runs must be > 0.")


def _build_run_id(parameters: RunParameters) -> str:
    return (
        f"{parameters.scenario_name}"
        f"-u{parameters.user_count}"
        f"-acir{parameters.acir_db:g}"
        f"-r{parameters.run_index + 1}"
    )


def _aggregate_runs(runs: tuple[BaselineRunResult, ...]) -> AggregatedSINRMetrics:
    if not runs:
        raise ValueError("runs cannot be empty.")

    run_count = len(runs)
    mean_sinr_db = sum(run.sinr.mean_sinr_db for run in runs) / run_count
    min_sinr_db = min(run.sinr.min_sinr_db for run in runs)
    max_sinr_db = max(run.sinr.max_sinr_db for run in runs)

    mean_signal_power_mw = sum(dbm_to_mw(run.sinr.mean_signal_power_dbm) for run in runs) / run_count
    mean_interference_power_mw = sum(dbm_to_mw(run.sinr.mean_interference_power_dbm) for run in runs) / run_count

    return AggregatedSINRMetrics(
        run_count=run_count,
        mean_sinr_db=mean_sinr_db,
        min_sinr_db=min_sinr_db,
        max_sinr_db=max_sinr_db,
        mean_signal_power_dbm=mw_to_dbm(mean_signal_power_mw),
        mean_interference_power_dbm=mw_to_dbm(mean_interference_power_mw),
    )


def _execute_single_run(
    config: SimulationConfig,
    *,
    scenario_name: str,
    user_count: int,
    acir_db: float,
    run_index: int,
    seed: int,
) -> BaselineRunResult:
    scenario = generate_network_scenario(
        config,
        scenario_name=scenario_name,
        user_count=user_count,
        seed=seed,
    )
    sinr_summary = evaluate_scenario_sinr(scenario, config, acir_db=acir_db)
    parameters = RunParameters(
        scenario_name=scenario_name,
        user_count=user_count,
        acir_db=acir_db,
        run_index=run_index,
        seed=seed,
    )
    return BaselineRunResult(
        run_id=_build_run_id(parameters),
        parameters=parameters,
        sinr=sinr_summary,
        metadata={
            "scenario": scenario_name,
            "user_count": user_count,
            "acir_db": acir_db,
            "seed": seed,
            "run_index": run_index,
            "satellite_count": len(scenario.satellites),
            "hap_count": len(scenario.haps),
            "gnb_count": len(scenario.gnbs),
            "generated_user_count": len(scenario.users),
            "enforce_baseline_realism": scenario.enforce_baseline_realism,
        },
    )


def run_deterministic_baseline(
    config: SimulationConfig,
    *,
    scenario_name: str | None = None,
    user_count: int | None = None,
    acir_db: float | None = None,
    seed: int | None = None,
) -> DeterministicSimulationResult:
    """Run one deterministic baseline simulation with typed output payloads."""
    resolved_scenario_name = _resolve_preset(config, scenario_name).name
    resolved_user_count = config.resolve_user_count(user_count)
    resolved_acir_db = config.acir_sweep_db[0] if acir_db is None else acir_db
    resolved_seed = config.reproducibility.seed if seed is None else seed

    metadata_records = [
        SimulationLogRecord(
            event="deterministic_run_started",
            metadata={
                "scenario": resolved_scenario_name,
                "user_count": resolved_user_count,
                "acir_db": resolved_acir_db,
                "seed": resolved_seed,
            },
        )
    ]

    run = _execute_single_run(
        config,
        scenario_name=resolved_scenario_name,
        user_count=resolved_user_count,
        acir_db=resolved_acir_db,
        run_index=0,
        seed=resolved_seed,
    )
    aggregate = _aggregate_runs((run,))
    metadata_records.append(
        SimulationLogRecord(
            event="deterministic_run_completed",
            metadata={
                "run_id": run.run_id,
                "mean_sinr_db": run.sinr.mean_sinr_db,
                "min_sinr_db": run.sinr.min_sinr_db,
                "max_sinr_db": run.sinr.max_sinr_db,
            },
        )
    )

    return DeterministicSimulationResult(
        run=run,
        aggregate=aggregate,
        metadata_records=tuple(metadata_records),
    )


def run_monte_carlo_baseline(
    config: SimulationConfig,
    *,
    monte_carlo_runs: int | None = None,
    scenario_order: tuple[str, ...] | None = None,
    user_count_sweep: tuple[int, ...] | None = None,
    acir_sweep_db: tuple[float, ...] | None = None,
    base_seed: int | None = None,
) -> MonteCarloSimulationResult:
    """Run Monte Carlo baseline sweeps for scenario, user-count, and ACIR points."""
    resolved_monte_carlo_runs = config.experiment.monte_carlo_runs if monte_carlo_runs is None else monte_carlo_runs
    if resolved_monte_carlo_runs <= 0:
        raise ValueError("monte_carlo_runs must be > 0.")

    resolved_scenario_order = config.experiment.scenario_order if scenario_order is None else scenario_order
    if not resolved_scenario_order:
        raise ValueError("scenario_order cannot be empty.")
    for scenario_name in resolved_scenario_order:
        config.scenario.get_preset(scenario_name)

    resolved_user_count_sweep = config.experiment.user_count_sweep if user_count_sweep is None else user_count_sweep
    if not resolved_user_count_sweep:
        raise ValueError("user_count_sweep cannot be empty.")
    for user_count in resolved_user_count_sweep:
        config.resolve_user_count(user_count)

    resolved_acir_sweep_db = config.acir_sweep_db if acir_sweep_db is None else acir_sweep_db
    if not resolved_acir_sweep_db:
        raise ValueError("acir_sweep_db cannot be empty.")

    resolved_base_seed = config.reproducibility.seed if base_seed is None else base_seed
    if resolved_base_seed < 0:
        raise ValueError("base_seed must be >= 0.")

    metadata_records = [
        SimulationLogRecord(
            event="monte_carlo_started",
            metadata={
                "base_seed": resolved_base_seed,
                "monte_carlo_runs": resolved_monte_carlo_runs,
                "scenario_points": len(resolved_scenario_order),
                "user_count_points": len(resolved_user_count_sweep),
                "acir_points": len(resolved_acir_sweep_db),
            },
        )
    ]
    sweep_results: list[SweepResult] = []

    for scenario_name in resolved_scenario_order:
        for user_count in resolved_user_count_sweep:
            for acir_db in resolved_acir_sweep_db:
                run_results = []
                for run_index in range(resolved_monte_carlo_runs):
                    run_seed = resolved_base_seed + run_index
                    run_result = _execute_single_run(
                        config,
                        scenario_name=scenario_name,
                        user_count=user_count,
                        acir_db=acir_db,
                        run_index=run_index,
                        seed=run_seed,
                    )
                    run_results.append(run_result)
                    metadata_records.append(
                        SimulationLogRecord(
                            event="run_completed",
                            metadata={
                                "run_id": run_result.run_id,
                                "scenario": scenario_name,
                                "user_count": user_count,
                                "acir_db": acir_db,
                                "seed": run_seed,
                                "mean_sinr_db": run_result.sinr.mean_sinr_db,
                            },
                        )
                    )

                run_results_tuple = tuple(run_results)
                aggregate = _aggregate_runs(run_results_tuple)
                sweep_result = SweepResult(
                    scenario_name=scenario_name,
                    user_count=user_count,
                    acir_db=acir_db,
                    runs=run_results_tuple,
                    aggregate=aggregate,
                )
                sweep_results.append(sweep_result)
                metadata_records.append(
                    SimulationLogRecord(
                        event="sweep_point_completed",
                        metadata={
                            "scenario": scenario_name,
                            "user_count": user_count,
                            "acir_db": acir_db,
                            "run_count": aggregate.run_count,
                            "mean_sinr_db": aggregate.mean_sinr_db,
                        },
                    )
                )

    metadata_records.append(
        SimulationLogRecord(
            event="monte_carlo_completed",
            metadata={
                "sweep_point_count": len(sweep_results),
                "total_run_count": len(sweep_results) * resolved_monte_carlo_runs,
            },
        )
    )

    return MonteCarloSimulationResult(
        base_seed=resolved_base_seed,
        monte_carlo_runs=resolved_monte_carlo_runs,
        scenario_order=tuple(resolved_scenario_order),
        user_count_sweep=tuple(resolved_user_count_sweep),
        acir_sweep_db=tuple(resolved_acir_sweep_db),
        sweep_results=tuple(sweep_results),
        metadata_records=tuple(metadata_records),
    )


@dataclass(frozen=True)
class SimulationContext:
    """Context wrapper passed into simulation execution."""

    config: SimulationConfig


class SimulationRunner:
    """High-level simulation runner for deterministic and Monte Carlo baselines."""

    def __init__(self, context: SimulationContext) -> None:
        self._context = context

    @property
    def context(self) -> SimulationContext:
        """Expose the immutable simulation context."""
        return self._context

    def run_deterministic(
        self,
        *,
        scenario_name: str | None = None,
        user_count: int | None = None,
        acir_db: float | None = None,
        seed: int | None = None,
    ) -> DeterministicSimulationResult:
        """Execute one deterministic baseline run."""
        return run_deterministic_baseline(
            self.context.config,
            scenario_name=scenario_name,
            user_count=user_count,
            acir_db=acir_db,
            seed=seed,
        )

    def run_monte_carlo(
        self,
        *,
        monte_carlo_runs: int | None = None,
        scenario_order: tuple[str, ...] | None = None,
        user_count_sweep: tuple[int, ...] | None = None,
        acir_sweep_db: tuple[float, ...] | None = None,
        base_seed: int | None = None,
    ) -> MonteCarloSimulationResult:
        """Execute configured Monte Carlo sweeps."""
        return run_monte_carlo_baseline(
            self.context.config,
            monte_carlo_runs=monte_carlo_runs,
            scenario_order=scenario_order,
            user_count_sweep=user_count_sweep,
            acir_sweep_db=acir_sweep_db,
            base_seed=base_seed,
        )

    def run(self) -> MonteCarloSimulationResult:
        """Execute the default baseline Monte Carlo sweep plan from config."""
        return self.run_monte_carlo()


