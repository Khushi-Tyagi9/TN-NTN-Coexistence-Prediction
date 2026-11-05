"""Time-stepped TN-NTN simulation: the data source for Phase 3 prediction.

The baseline simulator (``simulation.py``) evaluates ONE static snapshot per run.
An LSTM needs sequences, so this module adds a time axis on top of the existing
scenario generator and path-loss model:

* the LEO satellite flies over the anchor point (slow-varying signal power),
* users random-walk around their start position,
* the gNB -> user interference link gets log-normal shadowing, fast (Rayleigh)
  fading and a slowly varying gNB traffic-load term.

With every random term switched off and zero motion, the output equals the
baseline ``evaluate_scenario_sinr`` (see tests/test_timeseries.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np
from scipy.special import j0

from .channels import FSPL_CONSTANT_DB, HZ_PER_MHZ
from .config import SimulationConfig
from .interference import compute_noise_power_dbm
from .simulation import generate_network_scenario

FEATURE_NAMES = ("interference_dbm", "signal_dbm", "sinr_db")
MIN_LINK_DISTANCE_KM = 0.01  # 10 m guard so path loss never goes non-physical
MIN_FAST_FADING_POWER = 1e-3  # clip deep fades at -30 dB
SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class TimeSeriesConfig:
    """Time axis and channel-dynamics parameters."""

    n_slots: int = 600
    slot_duration_s: float = 0.5
    satellite_speed_km_s: float = 7.5
    user_speed_m_s: float = 1.5
    shadowing_sigma_db_urban: float = 8.0
    shadowing_sigma_db_rural: float = 5.0
    shadowing_correlation: float = 0.95
    fast_fading_enabled: bool = True
    # None -> derived from Doppler: rho = J0(2*pi*f_d*dt), f_d = v*f_c/c (Clarke/Jakes lag-1
    # autocorrelation). A float in [0, 1) overrides it (e.g. for tests).
    fast_fading_correlation: float | None = None
    load_sigma_db: float = 2.0
    load_correlation: float = 0.98
    signal_jitter_sigma_db: float = 0.3

    def __post_init__(self) -> None:
        if self.n_slots < 12:
            raise ValueError("n_slots must be >= 12 (10-slot lookback + 1 target + margin).")
        if self.slot_duration_s <= 0:
            raise ValueError("slot_duration_s must be > 0.")
        for name in ("satellite_speed_km_s", "user_speed_m_s", "load_sigma_db", "signal_jitter_sigma_db",
                     "shadowing_sigma_db_urban", "shadowing_sigma_db_rural"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0.")
        for name in ("shadowing_correlation", "load_correlation"):
            if not (0.0 <= getattr(self, name) < 1.0):
                raise ValueError(f"{name} must be in [0, 1).")
        if self.fast_fading_correlation is not None and not (0.0 <= self.fast_fading_correlation < 1.0):
            raise ValueError("fast_fading_correlation must be None (Doppler-derived) or in [0, 1).")

    def doppler_hz(self, carrier_frequency_hz: float) -> float:
        """Maximum Doppler shift of the user's motion: f_d = v * f_c / c."""
        if carrier_frequency_hz <= 0:
            raise ValueError("carrier_frequency_hz must be > 0.")
        return self.user_speed_m_s * carrier_frequency_hz / SPEED_OF_LIGHT_M_S

    def fast_fading_rho(self, carrier_frequency_hz: float) -> float:
        """Slot-to-slot correlation of the complex fading coefficient.

        Derived from Doppler as J0(2*pi*f_d*dt) unless overridden. It can be slightly negative
        (J0 oscillates); the AR(1) generator handles that. At 2 GHz, 1.5 m/s and 0.5 s slots
        this is ~0.10, i.e. fast fading is essentially independent from slot to slot.
        """
        if self.fast_fading_correlation is not None:
            return float(self.fast_fading_correlation)
        return float(j0(2.0 * pi * self.doppler_hz(carrier_frequency_hz) * self.slot_duration_s))

    def shadowing_sigma_db(self, scenario_name: str) -> float:
        return self.shadowing_sigma_db_urban if scenario_name == "urban" else self.shadowing_sigma_db_rural


@dataclass(frozen=True, eq=False)
class TimeSeriesRun:
    """Per-slot series for every user of one simulated run. Arrays are (users, slots)."""

    scenario_name: str
    user_count: int
    acir_db: float
    seed: int
    slot_duration_s: float
    noise_power_dbm: float
    interference_dbm: np.ndarray  # instantaneous: includes fast fading
    signal_dbm: np.ndarray
    sinr_db: np.ndarray
    interference_local_dbm: np.ndarray  # local mean: fast-fading term removed (E over unit-mean Rayleigh power)

    def features(self) -> np.ndarray:
        """Stack to (users, slots, 3) in FEATURE_NAMES order."""
        return np.stack([self.interference_dbm, self.signal_dbm, self.sinr_db], axis=-1)


def _fspl_db(distance_km: np.ndarray, carrier_frequency_hz: float) -> np.ndarray:
    """Vectorised twin of channels.free_space_path_loss_db."""
    carrier_mhz = carrier_frequency_hz / HZ_PER_MHZ
    return FSPL_CONSTANT_DB + 20.0 * np.log10(distance_km) + 20.0 * np.log10(carrier_mhz)


def _ar1_gaussian(rng: np.random.Generator, shape: tuple[int, ...], rho: float, sigma: float) -> np.ndarray:
    """Stationary AR(1) Gaussian series along the last axis with std ``sigma``."""
    out = np.empty(shape)
    noise = rng.normal(0.0, sigma * sqrt(1.0 - rho**2), shape)
    out[..., 0] = rng.normal(0.0, sigma, shape[:-1])
    for t in range(1, shape[-1]):
        out[..., t] = rho * out[..., t - 1] + noise[..., t]
    return out


def _fast_fading_db(rng: np.random.Generator, shape: tuple[int, ...], rho: float) -> np.ndarray:
    """Rayleigh fading power in dB (unit mean power) with AR(1) time correlation."""
    real = _ar1_gaussian(rng, shape, rho, sqrt(0.5))
    imag = _ar1_gaussian(rng, shape, rho, sqrt(0.5))
    return 10.0 * np.log10(np.maximum(real**2 + imag**2, MIN_FAST_FADING_POWER))


def simulate_time_series(
    config: SimulationConfig,
    ts_config: TimeSeriesConfig | None = None,
    *,
    scenario_name: str,
    user_count: int,
    acir_db: float,
    seed: int,
) -> TimeSeriesRun:
    """Simulate one run over ``n_slots`` slots and return per-user series."""
    ts = TimeSeriesConfig() if ts_config is None else ts_config
    if acir_db <= 0:
        raise ValueError("acir_db must be > 0.")

    scenario = generate_network_scenario(config, scenario_name=scenario_name, user_count=user_count, seed=seed)
    rng = np.random.default_rng(seed)
    n_slots, dt = ts.n_slots, ts.slot_duration_s
    n_users, n_gnbs = len(scenario.users), len(scenario.gnbs)
    carrier_hz = config.radio.carrier_frequency_hz
    t_s = np.arange(n_slots) * dt

    # --- satellite pass: symmetric fly-over along x, centred on the scenario satellite
    sat = scenario.satellites[0]
    half_span_km = 0.5 * ts.satellite_speed_km_s * n_slots * dt
    sat_x = sat.position.x_km - half_span_km + ts.satellite_speed_km_s * t_s

    # --- user random walk (first slot = generated start position)
    start_xy = np.array([[u.position.x_km, u.position.y_km] for u in scenario.users])
    user_z = np.array([u.position.z_km for u in scenario.users])
    step_km = ts.user_speed_m_s * dt / 1000.0
    steps = rng.normal(0.0, step_km, (n_users, 2, n_slots))
    steps[:, :, 0] = 0.0
    user_xy = start_xy[:, :, None] + np.cumsum(steps, axis=2)  # (U, 2, T)

    # --- wanted signal: satellite downlink
    sat_dist = np.sqrt(
        (user_xy[:, 0, :] - sat_x[None, :]) ** 2
        + (user_xy[:, 1, :] - sat.position.y_km) ** 2
        + (sat.position.z_km - user_z[:, None]) ** 2
    )
    signal_dbm = (
        sat.tx_power_dbw + 30.0 + sat.tx_gain_dbi
        - _fspl_db(sat_dist, carrier_hz)
        + rng.normal(0.0, ts.signal_jitter_sigma_db, (n_users, n_slots))
    )

    # --- interference: leaked gNB power with shadowing + fast fading + load
    gnb_xy = np.array([[g.position.x_km, g.position.y_km] for g in scenario.gnbs])
    gnb_z = np.array([g.position.z_km for g in scenario.gnbs])
    gnb_tx = np.array([g.tx_power_dbm for g in scenario.gnbs])[:, None, None]
    gnb_dist = np.sqrt(
        (user_xy[None, :, 0, :] - gnb_xy[:, 0, None, None]) ** 2
        + (user_xy[None, :, 1, :] - gnb_xy[:, 1, None, None]) ** 2
        + (gnb_z[:, None, None] - user_z[None, :, None]) ** 2
    )
    gnb_dist = np.maximum(gnb_dist, MIN_LINK_DISTANCE_KM)
    shape_gu = (n_gnbs, n_users, n_slots)
    link_dbm = gnb_tx - _fspl_db(gnb_dist, carrier_hz) - acir_db
    link_dbm = link_dbm + _ar1_gaussian(rng, shape_gu, ts.shadowing_correlation, ts.shadowing_sigma_db(scenario_name))
    fading_db = (
        _fast_fading_db(rng, shape_gu, ts.fast_fading_rho(carrier_hz))
        if ts.fast_fading_enabled
        else np.zeros(shape_gu)
    )
    load_db = _ar1_gaussian(rng, (n_gnbs, 1, n_slots), ts.load_correlation, ts.load_sigma_db)
    local_link_dbm = link_dbm + load_db  # path loss + ACIR + shadowing + load (no fast fading)
    interference_local_mw = np.sum(10.0 ** (local_link_dbm / 10.0), axis=0)  # (U, T)
    interference_mw = np.sum(10.0 ** ((local_link_dbm + fading_db) / 10.0), axis=0)  # (U, T)

    # --- SINR
    noise_dbm = compute_noise_power_dbm(config.radio.bandwidth_hz, config.radio.noise_figure_db)
    noise_mw = 10.0 ** (noise_dbm / 10.0)
    sinr_db = signal_dbm - 10.0 * np.log10(interference_mw + noise_mw)

    return TimeSeriesRun(
        scenario_name=scenario_name,
        user_count=n_users,
        acir_db=float(acir_db),
        seed=seed,
        slot_duration_s=dt,
        noise_power_dbm=float(noise_dbm),
        interference_dbm=10.0 * np.log10(interference_mw),
        signal_dbm=signal_dbm,
        sinr_db=sinr_db,
        interference_local_dbm=10.0 * np.log10(interference_local_mw),
    )