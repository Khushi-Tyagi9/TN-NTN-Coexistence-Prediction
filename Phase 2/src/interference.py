"""Baseline SINR and ACIR-based interference calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log10

from .channels import total_path_loss_db
from .config import SimulationConfig
from .entities import NTNUserNode, NetworkScenario, SatelliteNode, TNGNBNode, distance_km

THERMAL_NOISE_DENSITY_DBM_PER_HZ = -174.0


def _validate_positive_finite(value: float, field_name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite.")
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0.")


def _validate_non_negative_finite(value: float, field_name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite.")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0.")


def dbm_to_mw(power_dbm: float) -> float:
    """Convert dBm to mW (supports -inf -> 0)."""
    if power_dbm == float("-inf"):
        return 0.0
    if not isfinite(power_dbm):
        raise ValueError("power_dbm must be finite or -inf.")
    return 10.0 ** (power_dbm / 10.0)


def mw_to_dbm(power_mw: float) -> float:
    """Convert mW to dBm (supports 0 -> -inf)."""
    if not isfinite(power_mw):
        raise ValueError("power_mw must be finite.")
    if power_mw < 0:
        raise ValueError("power_mw must be >= 0.")
    if power_mw == 0:
        return float("-inf")
    return 10.0 * log10(power_mw)


def compute_noise_power_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    """Thermal noise + receiver noise figure in dBm."""
    _validate_positive_finite(bandwidth_hz, "bandwidth_hz")
    _validate_non_negative_finite(noise_figure_db, "noise_figure_db")
    return THERMAL_NOISE_DENSITY_DBM_PER_HZ + 10.0 * log10(bandwidth_hz) + noise_figure_db


@dataclass(frozen=True, slots=True)
class InterferenceRecord:
    """Single gNB contribution at a user after ACIR leakage attenuation."""

    source_id: str
    distance_km: float
    path_loss_db: float
    in_band_power_dbm: float
    acir_db: float
    leaked_power_dbm: float

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str):
            raise TypeError("source_id must be a string.")
        if not self.source_id.strip():
            raise ValueError("source_id cannot be empty.")
        _validate_positive_finite(self.distance_km, "distance_km")
        _validate_non_negative_finite(self.path_loss_db, "path_loss_db")
        if not isfinite(self.in_band_power_dbm):
            raise ValueError("in_band_power_dbm must be finite.")
        _validate_positive_finite(self.acir_db, "acir_db")
        if not isfinite(self.leaked_power_dbm):
            raise ValueError("leaked_power_dbm must be finite.")


@dataclass(frozen=True, slots=True)
class UserSINRMetrics:
    """Per-user baseline signal/interference/noise outputs."""

    user_id: str
    serving_satellite_id: str
    signal_distance_km: float
    signal_path_loss_db: float
    signal_power_dbm: float
    signal_power_mw: float
    interference_power_dbm: float
    interference_power_mw: float
    noise_power_dbm: float
    noise_power_mw: float
    sinr_db: float
    acir_db: float
    interferers: tuple[InterferenceRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str):
            raise TypeError("user_id must be a string.")
        if not self.user_id.strip():
            raise ValueError("user_id cannot be empty.")
        if not isinstance(self.serving_satellite_id, str):
            raise TypeError("serving_satellite_id must be a string.")
        if not self.serving_satellite_id.strip():
            raise ValueError("serving_satellite_id cannot be empty.")
        _validate_positive_finite(self.signal_distance_km, "signal_distance_km")
        _validate_non_negative_finite(self.signal_path_loss_db, "signal_path_loss_db")
        if not isfinite(self.signal_power_dbm):
            raise ValueError("signal_power_dbm must be finite.")
        _validate_positive_finite(self.signal_power_mw, "signal_power_mw")
        if self.interference_power_dbm != float("-inf") and not isfinite(self.interference_power_dbm):
            raise ValueError("interference_power_dbm must be finite or -inf.")
        _validate_non_negative_finite(self.interference_power_mw, "interference_power_mw")
        if not isfinite(self.noise_power_dbm):
            raise ValueError("noise_power_dbm must be finite.")
        _validate_positive_finite(self.noise_power_mw, "noise_power_mw")
        if not isfinite(self.sinr_db):
            raise ValueError("sinr_db must be finite.")
        _validate_positive_finite(self.acir_db, "acir_db")


@dataclass(frozen=True, slots=True)
class ScenarioSINRSummary:
    """Scenario-level summary with all per-user SINR outputs."""

    acir_db: float
    user_metrics: tuple[UserSINRMetrics, ...]
    mean_sinr_db: float
    min_sinr_db: float
    max_sinr_db: float
    mean_signal_power_dbm: float
    mean_interference_power_dbm: float
    noise_power_dbm: float

    def __post_init__(self) -> None:
        _validate_positive_finite(self.acir_db, "acir_db")
        if not self.user_metrics:
            raise ValueError("user_metrics cannot be empty.")
        for field_name in ("mean_sinr_db", "min_sinr_db", "max_sinr_db", "mean_signal_power_dbm", "noise_power_dbm"):
            if not isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite.")
        if self.mean_interference_power_dbm != float("-inf") and not isfinite(self.mean_interference_power_dbm):
            raise ValueError("mean_interference_power_dbm must be finite or -inf.")


def _compute_satellite_received_power(
    satellite: SatelliteNode,
    user: NTNUserNode,
    carrier_frequency_hz: float,
    additional_path_loss_db: float,
) -> tuple[float, float, float]:
    signal_distance_km = distance_km(satellite, user)
    signal_path_loss_db = total_path_loss_db(
        signal_distance_km,
        carrier_frequency_hz,
        additional_loss_db=additional_path_loss_db,
    )
    signal_power_dbm = satellite.tx_power_dbw + 30.0 + satellite.tx_gain_dbi - signal_path_loss_db
    return signal_power_dbm, signal_path_loss_db, signal_distance_km


def _compute_interference_record(
    gnb: TNGNBNode,
    user: NTNUserNode,
    carrier_frequency_hz: float,
    acir_db: float,
    additional_path_loss_db: float,
) -> InterferenceRecord:
    interference_distance_km = distance_km(gnb, user)
    path_loss_db = total_path_loss_db(
        interference_distance_km,
        carrier_frequency_hz,
        additional_loss_db=additional_path_loss_db,
    )
    in_band_power_dbm = gnb.tx_power_dbm - path_loss_db
    leaked_power_dbm = in_band_power_dbm - acir_db
    return InterferenceRecord(
        source_id=gnb.entity_id,
        distance_km=interference_distance_km,
        path_loss_db=path_loss_db,
        in_band_power_dbm=in_band_power_dbm,
        acir_db=acir_db,
        leaked_power_dbm=leaked_power_dbm,
    )


def compute_user_sinr_metrics(
    scenario: NetworkScenario,
    config: SimulationConfig,
    user: NTNUserNode,
    *,
    acir_db: float,
    additional_path_loss_db: float = 0.0,
) -> UserSINRMetrics:
    """Compute baseline signal, ACIR-leaked interference, and SINR for one user."""
    if not isinstance(scenario, NetworkScenario):
        raise TypeError("scenario must be a NetworkScenario.")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig.")
    if not isinstance(user, NTNUserNode):
        raise TypeError("user must be an NTNUserNode.")
    _validate_positive_finite(acir_db, "acir_db")
    _validate_non_negative_finite(additional_path_loss_db, "additional_path_loss_db")
    if user not in scenario.users:
        raise ValueError(f"user '{user.entity_id}' does not exist in the supplied scenario.")
    if not scenario.satellites:
        raise ValueError("Scenario must contain at least one satellite.")

    carrier_frequency_hz = config.radio.carrier_frequency_hz
    serving_satellite: SatelliteNode | None = None
    serving_signal_power_dbm: float | None = None
    serving_path_loss_db: float | None = None
    serving_distance_km: float | None = None

    for satellite in scenario.satellites:
        signal_power_dbm, signal_path_loss_db, signal_distance_km = _compute_satellite_received_power(
            satellite,
            user,
            carrier_frequency_hz,
            additional_path_loss_db,
        )
        if serving_signal_power_dbm is None or signal_power_dbm > serving_signal_power_dbm:
            serving_satellite = satellite
            serving_signal_power_dbm = signal_power_dbm
            serving_path_loss_db = signal_path_loss_db
            serving_distance_km = signal_distance_km

    if (
        serving_satellite is None
        or serving_signal_power_dbm is None
        or serving_path_loss_db is None
        or serving_distance_km is None
    ):
        raise RuntimeError("Unable to resolve a serving satellite despite non-empty satellite list.")

    interferers = tuple(
        _compute_interference_record(
            gnb,
            user,
            carrier_frequency_hz,
            acir_db,
            additional_path_loss_db,
        )
        for gnb in scenario.gnbs
    )
    interference_power_mw = sum(dbm_to_mw(record.leaked_power_dbm) for record in interferers)
    interference_power_dbm = mw_to_dbm(interference_power_mw)

    noise_power_dbm = compute_noise_power_dbm(
        config.radio.bandwidth_hz,
        config.radio.noise_figure_db,
    )
    noise_power_mw = dbm_to_mw(noise_power_dbm)
    signal_power_mw = dbm_to_mw(serving_signal_power_dbm)

    denominator_mw = interference_power_mw + noise_power_mw
    _validate_positive_finite(denominator_mw, "denominator_mw")
    sinr_db = 10.0 * log10(signal_power_mw / denominator_mw)
    if not isfinite(sinr_db):
        raise ValueError("Computed SINR is non-finite; check scenario and inputs.")

    return UserSINRMetrics(
        user_id=user.entity_id,
        serving_satellite_id=serving_satellite.entity_id,
        signal_distance_km=serving_distance_km,
        signal_path_loss_db=serving_path_loss_db,
        signal_power_dbm=serving_signal_power_dbm,
        signal_power_mw=signal_power_mw,
        interference_power_dbm=interference_power_dbm,
        interference_power_mw=interference_power_mw,
        noise_power_dbm=noise_power_dbm,
        noise_power_mw=noise_power_mw,
        sinr_db=sinr_db,
        acir_db=acir_db,
        interferers=interferers,
    )


def evaluate_scenario_sinr(
    scenario: NetworkScenario,
    config: SimulationConfig,
    *,
    acir_db: float,
    additional_path_loss_db: float = 0.0,
) -> ScenarioSINRSummary:
    """Compute per-user SINR and summary metrics for a full scenario."""
    if not isinstance(scenario, NetworkScenario):
        raise TypeError("scenario must be a NetworkScenario.")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig.")
    _validate_positive_finite(acir_db, "acir_db")
    _validate_non_negative_finite(additional_path_loss_db, "additional_path_loss_db")
    if not scenario.users:
        raise ValueError("Scenario must contain at least one user.")

    user_metrics = tuple(
        compute_user_sinr_metrics(
            scenario,
            config,
            user,
            acir_db=acir_db,
            additional_path_loss_db=additional_path_loss_db,
        )
        for user in scenario.users
    )

    sinr_values_db = [metric.sinr_db for metric in user_metrics]
    mean_sinr_db = sum(sinr_values_db) / len(sinr_values_db)
    min_sinr_db = min(sinr_values_db)
    max_sinr_db = max(sinr_values_db)

    mean_signal_power_mw = sum(metric.signal_power_mw for metric in user_metrics) / len(user_metrics)
    mean_interference_power_mw = sum(metric.interference_power_mw for metric in user_metrics) / len(user_metrics)

    return ScenarioSINRSummary(
        acir_db=acir_db,
        user_metrics=user_metrics,
        mean_sinr_db=mean_sinr_db,
        min_sinr_db=min_sinr_db,
        max_sinr_db=max_sinr_db,
        mean_signal_power_dbm=mw_to_dbm(mean_signal_power_mw),
        mean_interference_power_dbm=mw_to_dbm(mean_interference_power_mw),
        noise_power_dbm=user_metrics[0].noise_power_dbm,
    )


@dataclass(frozen=True, slots=True)
class InterferenceModel:
    """Thin model wrapper around baseline scenario SINR evaluation."""

    scenario: NetworkScenario
    config: SimulationConfig
    acir_db: float
    additional_path_loss_db: float = 0.0

    def evaluate(self) -> ScenarioSINRSummary:
        """Run baseline per-user SINR evaluation for this scenario."""
        return evaluate_scenario_sinr(
            self.scenario,
            self.config,
            acir_db=self.acir_db,
            additional_path_loss_db=self.additional_path_loss_db,
        )

