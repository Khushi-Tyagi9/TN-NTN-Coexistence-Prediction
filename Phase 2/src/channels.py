"""Channel/path-loss helpers and model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite, log10

FSPL_CONSTANT_DB = 32.44
HZ_PER_MHZ = 1.0e6


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


@dataclass(frozen=True, slots=True)
class ChannelSample:
    """Minimal channel snapshot for a single link evaluation."""

    path_loss_db: float
    fading_db: float = 0.0

    def __post_init__(self) -> None:
        _validate_non_negative_finite(self.path_loss_db, "path_loss_db")
        if not isfinite(self.fading_db):
            raise ValueError("fading_db must be finite.")


@dataclass(frozen=True, slots=True)
class PathLossBreakdown:
    """Deterministic path-loss breakdown for one link."""

    distance_km: float
    carrier_frequency_hz: float
    free_space_loss_db: float
    additional_loss_db: float = 0.0

    def __post_init__(self) -> None:
        _validate_positive_finite(self.distance_km, "distance_km")
        _validate_positive_finite(self.carrier_frequency_hz, "carrier_frequency_hz")
        _validate_non_negative_finite(self.free_space_loss_db, "free_space_loss_db")
        _validate_non_negative_finite(self.additional_loss_db, "additional_loss_db")

    @property
    def total_loss_db(self) -> float:
        """Total deterministic path loss in dB."""
        return self.free_space_loss_db + self.additional_loss_db


class BaseChannelModel(ABC):
    """Abstract contract for channel implementations."""

    @abstractmethod
    def evaluate(self) -> ChannelSample:
        """Return a channel sample for the current context."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class FreeSpaceChannelModel(BaseChannelModel):
    """Simple deterministic channel model using FSPL + optional excess loss."""

    distance_km: float
    carrier_frequency_hz: float
    additional_loss_db: float = 0.0
    fading_db: float = 0.0

    def __post_init__(self) -> None:
        _validate_positive_finite(self.distance_km, "distance_km")
        _validate_positive_finite(self.carrier_frequency_hz, "carrier_frequency_hz")
        _validate_non_negative_finite(self.additional_loss_db, "additional_loss_db")
        if not isfinite(self.fading_db):
            raise ValueError("fading_db must be finite.")

    def evaluate(self) -> ChannelSample:
        return ChannelSample(
            path_loss_db=total_path_loss_db(
                self.distance_km,
                self.carrier_frequency_hz,
                additional_loss_db=self.additional_loss_db,
            ),
            fading_db=self.fading_db,
        )


def free_space_path_loss_db(distance_km: float, carrier_frequency_hz: float) -> float:
    """Compute free-space path loss (dB) for distance in km and frequency in Hz."""
    _validate_positive_finite(distance_km, "distance_km")
    _validate_positive_finite(carrier_frequency_hz, "carrier_frequency_hz")
    carrier_frequency_mhz = carrier_frequency_hz / HZ_PER_MHZ
    return FSPL_CONSTANT_DB + 20.0 * log10(distance_km) + 20.0 * log10(carrier_frequency_mhz)


def baseline_path_loss(
    distance_km: float,
    carrier_frequency_hz: float,
    *,
    additional_loss_db: float = 0.0,
) -> PathLossBreakdown:
    """Build a deterministic baseline path-loss breakdown."""
    _validate_non_negative_finite(additional_loss_db, "additional_loss_db")
    free_space_loss_db = free_space_path_loss_db(distance_km, carrier_frequency_hz)
    return PathLossBreakdown(
        distance_km=distance_km,
        carrier_frequency_hz=carrier_frequency_hz,
        free_space_loss_db=free_space_loss_db,
        additional_loss_db=additional_loss_db,
    )


def total_path_loss_db(
    distance_km: float,
    carrier_frequency_hz: float,
    *,
    additional_loss_db: float = 0.0,
) -> float:
    """Return total baseline path loss in dB."""
    return baseline_path_loss(
        distance_km,
        carrier_frequency_hz,
        additional_loss_db=additional_loss_db,
    ).total_loss_db

