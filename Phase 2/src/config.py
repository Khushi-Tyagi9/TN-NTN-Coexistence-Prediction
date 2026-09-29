"""Typed, validated configuration for the TN-NTN baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

BASELINE_GNB_COUNT = 1


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Controls deterministic behavior across simulation runs."""

    seed: int = 42

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be >= 0.")


@dataclass(frozen=True)
class ACIRSweepConfig:
    """Adjacent channel interference rejection sweep definition."""

    min_db: float = 20.0
    max_db: float = 40.0
    points_db: tuple[float, ...] = (20.0, 25.0, 30.0, 35.0, 40.0)

    def __post_init__(self) -> None:
        if self.min_db >= self.max_db:
            raise ValueError("ACIR bounds must satisfy min_db < max_db.")
        if len(self.points_db) < 2:
            raise ValueError("ACIR sweep must include at least two points.")
        if self.points_db[0] < self.min_db or self.points_db[-1] > self.max_db:
            raise ValueError("ACIR sweep points must remain within configured bounds.")
        if any(next_point <= point for point, next_point in zip(self.points_db, self.points_db[1:])):
            raise ValueError("ACIR sweep points must be strictly increasing.")


@dataclass(frozen=True)
class RadioConfig:
    """Radio and waveform assumptions shared by TN/NTN links."""

    carrier_frequency_hz: float = 2.0e9
    bandwidth_hz: float = 20.0e6
    noise_figure_db: float = 7.0
    acir_sweep: ACIRSweepConfig = field(default_factory=ACIRSweepConfig)

    def __post_init__(self) -> None:
        if self.carrier_frequency_hz <= 0:
            raise ValueError("carrier_frequency_hz must be > 0.")
        if self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be > 0.")
        if self.noise_figure_db < 0:
            raise ValueError("noise_figure_db must be >= 0.")


@dataclass(frozen=True)
class SatelliteConfig:
    """Coarse NTN (LEO) assumptions for the baseline."""

    leo_altitude_km: float = 550.0
    tx_power_dbw: float = 20.0
    tx_gain_dbi: float = 30.0
    rx_gain_dbi: float = 35.0

    def __post_init__(self) -> None:
        if self.leo_altitude_km <= 0:
            raise ValueError("leo_altitude_km must be > 0.")


@dataclass(frozen=True)
class HAPConfig:
    """HAP deployment and quasi-static position controls."""

    hap_count: int = 1
    altitude_km: float = 20.0
    position_model: str = "quasi_static"
    anchor_xy_km: tuple[float, float] = (0.0, 0.0)
    drift_radius_m: float = 100.0

    def __post_init__(self) -> None:
        if self.hap_count < 0:
            raise ValueError("hap_count must be >= 0.")
        if self.altitude_km <= 0:
            raise ValueError("altitude_km must be > 0.")
        if self.position_model not in {"quasi_static"}:
            raise ValueError("position_model must be 'quasi_static' for baseline.")
        if self.drift_radius_m < 0:
            raise ValueError("drift_radius_m must be >= 0.")


@dataclass(frozen=True)
class TNDeploymentConfig:
    """Terrestrial network deployment controls."""

    gnb_count: int = 1
    gnb_density_per_km2: float = 2.0
    gnb_tx_power_dbm: float = 46.0

    def __post_init__(self) -> None:
        if self.gnb_count <= 0:
            raise ValueError("gnb_count must be > 0.")
        if self.gnb_density_per_km2 <= 0:
            raise ValueError("gnb_density_per_km2 must be > 0.")


@dataclass(frozen=True)
class UserConfig:
    """UE count controls for baseline and sweeps."""

    min_count: int = 4
    max_count: int = 5
    default_count: int = 5

    def __post_init__(self) -> None:
        if self.min_count <= 0:
            raise ValueError("min_count must be > 0.")
        if self.min_count > self.max_count:
            raise ValueError("min_count must be <= max_count.")
        if not (self.min_count <= self.default_count <= self.max_count):
            raise ValueError("default_count must be within [min_count, max_count].")


@dataclass(frozen=True)
class ScenarioPreset:
    """Single scenario profile."""

    name: str
    description: str
    gnb_density_per_km2: float
    user_spread_km: float

    def __post_init__(self) -> None:
        if self.gnb_density_per_km2 <= 0:
            raise ValueError("gnb_density_per_km2 must be > 0.")
        if self.user_spread_km <= 0:
            raise ValueError("user_spread_km must be > 0.")


@dataclass(frozen=True)
class ScenarioConfig:
    """Scenario presets and active selection."""

    active_preset: str = "urban"
    urban: ScenarioPreset = field(
        default_factory=lambda: ScenarioPreset(
            name="urban",
            description="Dense TN deployment with tighter user clusters.",
            gnb_density_per_km2=3.5,
            user_spread_km=1.0,
        )
    )
    rural: ScenarioPreset = field(
        default_factory=lambda: ScenarioPreset(
            name="rural",
            description="Sparse TN deployment with wider user spread.",
            gnb_density_per_km2=0.8,
            user_spread_km=3.0,
        )
    )

    def __post_init__(self) -> None:
        if self.active_preset not in {"urban", "rural"}:
            raise ValueError("active_preset must be either 'urban' or 'rural'.")

    @property
    def presets(self) -> dict[str, ScenarioPreset]:
        """Expose all available scenario presets."""
        return {"urban": self.urban, "rural": self.rural}

    def get_preset(self, name: str) -> ScenarioPreset:
        """Resolve a preset by name."""
        try:
            return self.presets[name]
        except KeyError as exc:
            raise ValueError(f"Unknown scenario preset: {name}.") from exc


@dataclass(frozen=True)
class ExperimentConfig:
    """Baseline experiment controls and sweeps."""

    enforce_baseline_realism: bool = True
    monte_carlo_runs: int = 20
    user_count_sweep: tuple[int, ...] = (4, 5)
    scenario_order: tuple[str, ...] = ("urban", "rural")

    def __post_init__(self) -> None:
        if self.monte_carlo_runs <= 0:
            raise ValueError("monte_carlo_runs must be > 0.")
        if not self.user_count_sweep:
            raise ValueError("user_count_sweep cannot be empty.")
        if not self.scenario_order:
            raise ValueError("scenario_order cannot be empty.")


@dataclass(frozen=True)
class SimulationConfig:
    """Centralized, typed simulation configuration root."""

    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    radio: RadioConfig = field(default_factory=RadioConfig)
    satellite: SatelliteConfig = field(default_factory=SatelliteConfig)
    hap: HAPConfig = field(default_factory=HAPConfig)
    tn_deployment: TNDeploymentConfig = field(default_factory=TNDeploymentConfig)
    users: UserConfig = field(default_factory=UserConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def acir_sweep_db(self) -> tuple[float, ...]:
        """Expose ACIR sweep points for channel/interference modules."""
        return self.radio.acir_sweep.points_db

    def resolve_user_count(self, requested_count: int | None = None) -> int:
        """Resolve a user count while preserving configured bounds."""
        user_count = self.users.default_count if requested_count is None else requested_count
        if not (self.users.min_count <= user_count <= self.users.max_count):
            raise ValueError(
                f"user_count must be within [{self.users.min_count}, {self.users.max_count}]."
            )
        return user_count

    def get_active_scenario(self) -> ScenarioPreset:
        """Return the active scenario preset."""
        return self.scenario.get_preset(self.scenario.active_preset)

    def validate(self) -> None:
        """Run baseline realism checks across sections."""
        if not self.experiment.enforce_baseline_realism:
            return

        if self.hap.hap_count < 1:
            raise ValueError("Baseline requires hap_count >= 1.")
        if self.tn_deployment.gnb_count != BASELINE_GNB_COUNT:
            raise ValueError(f"Baseline requires gnb_count == {BASELINE_GNB_COUNT}.")
        if not (4 <= self.users.min_count <= self.users.default_count <= self.users.max_count <= 5):
            raise ValueError("Baseline requires 4 <= user_count <= 5 for min/default/max.")

        acir = self.radio.acir_sweep
        if acir.min_db >= acir.max_db:
            raise ValueError("Baseline ACIR bounds must satisfy min_db < max_db.")
        if acir.points_db[0] < acir.min_db or acir.points_db[-1] > acir.max_db:
            raise ValueError("Baseline ACIR sweep points must stay within bounds.")
        if any(next_point <= point for point, next_point in zip(acir.points_db, acir.points_db[1:])):
            raise ValueError("Baseline ACIR sweep points must be monotonic increasing.")

        for user_count in self.experiment.user_count_sweep:
            self.resolve_user_count(user_count)
        for preset_name in self.experiment.scenario_order:
            self.scenario.get_preset(preset_name)

