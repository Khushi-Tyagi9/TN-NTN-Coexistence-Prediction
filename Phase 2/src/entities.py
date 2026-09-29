"""Entity definitions and geometry helpers for TN/NTN scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import dist, isfinite
from typing import Iterable, TypeVar


class EntityType(str, Enum):
    """Supported high-level entity categories."""

    TN = "tn"
    NTN = "ntn"


class NodeRole(str, Enum):
    """Role-specific node categories."""

    SATELLITE = "satellite"
    HAP = "hap"
    GNB = "gnb"
    USER = "user"


class HAPPositionModel(str, Enum):
    """Supported HAP placement models."""

    QUASI_STATIC = "quasi_static"


def _validate_finite(value: float, field_name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite.")


@dataclass(frozen=True, slots=True)
class Position3D:
    """Simple position representation in kilometers."""

    x_km: float
    y_km: float
    z_km: float = 0.0

    def __post_init__(self) -> None:
        _validate_finite(self.x_km, "x_km")
        _validate_finite(self.y_km, "y_km")
        _validate_finite(self.z_km, "z_km")

    @classmethod
    def from_xy_km(cls, x_km: float, y_km: float, *, altitude_km: float = 0.0) -> Position3D:
        """Build a 3D position from a 2D point plus altitude."""
        return cls(x_km=x_km, y_km=y_km, z_km=altitude_km)

    @property
    def altitude_km(self) -> float:
        """Altitude component in kilometers."""
        return self.z_km


@dataclass(frozen=True, slots=True)
class NetworkEntity:
    """Common immutable network node representation."""

    entity_id: str
    position: Position3D
    entity_type: EntityType
    role: NodeRole

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str):
            raise TypeError("entity_id must be a string.")
        if not self.entity_id.strip():
            raise ValueError("entity_id cannot be empty.")
        if not isinstance(self.position, Position3D):
            raise TypeError("position must be a Position3D.")
        if not isinstance(self.entity_type, EntityType):
            raise TypeError("entity_type must be an EntityType.")
        if not isinstance(self.role, NodeRole):
            raise TypeError("role must be a NodeRole.")

    @property
    def altitude_km(self) -> float:
        """Altitude derived from the shared position representation."""
        return self.position.altitude_km


@dataclass(frozen=True, slots=True)
class SatelliteNode(NetworkEntity):
    """LEO satellite node."""

    entity_type: EntityType = EntityType.NTN
    role: NodeRole = NodeRole.SATELLITE
    leo_altitude_km: float = 550.0
    tx_power_dbw: float = 20.0
    tx_gain_dbi: float = 30.0
    rx_gain_dbi: float = 35.0

    def __post_init__(self) -> None:
        super(SatelliteNode, self).__post_init__()
        if self.entity_type is not EntityType.NTN:
            raise ValueError("SatelliteNode must use EntityType.NTN.")
        if self.role is not NodeRole.SATELLITE:
            raise ValueError("SatelliteNode must use NodeRole.SATELLITE.")
        if self.leo_altitude_km <= 0:
            raise ValueError("leo_altitude_km must be > 0.")
        if self.altitude_km <= 0:
            raise ValueError("Satellite altitude must be > 0.")
        if abs(self.altitude_km - self.leo_altitude_km) > 1e-6:
            raise ValueError("Satellite altitude must match leo_altitude_km.")


@dataclass(frozen=True, slots=True)
class HAPNode(NetworkEntity):
    """High-altitude platform node with quasi-static controls."""

    entity_type: EntityType = EntityType.NTN
    role: NodeRole = NodeRole.HAP
    position_model: HAPPositionModel | str = HAPPositionModel.QUASI_STATIC
    anchor_xy_km: tuple[float, float] = (0.0, 0.0)
    drift_radius_m: float = 100.0

    def __post_init__(self) -> None:
        super(HAPNode, self).__post_init__()
        if self.entity_type is not EntityType.NTN:
            raise ValueError("HAPNode must use EntityType.NTN.")
        if self.role is not NodeRole.HAP:
            raise ValueError("HAPNode must use NodeRole.HAP.")
        if isinstance(self.position_model, str):
            try:
                object.__setattr__(self, "position_model", HAPPositionModel(self.position_model))
            except ValueError as exc:
                raise ValueError("position_model must be 'quasi_static'.") from exc
        if not isinstance(self.position_model, HAPPositionModel):
            raise TypeError("position_model must be a HAPPositionModel.")
        if self.position_model is not HAPPositionModel.QUASI_STATIC:
            raise ValueError("position_model must be 'quasi_static'.")
        if len(self.anchor_xy_km) != 2:
            raise ValueError("anchor_xy_km must contain exactly two coordinates.")
        _validate_finite(self.anchor_xy_km[0], "anchor_xy_km[0]")
        _validate_finite(self.anchor_xy_km[1], "anchor_xy_km[1]")
        if self.drift_radius_m < 0:
            raise ValueError("drift_radius_m must be >= 0.")
        if self.altitude_km <= 0:
            raise ValueError("HAP altitude must be > 0.")


@dataclass(frozen=True, slots=True)
class TNGNBNode(NetworkEntity):
    """Terrestrial gNB node."""

    entity_type: EntityType = EntityType.TN
    role: NodeRole = NodeRole.GNB
    tx_power_dbm: float = 46.0

    def __post_init__(self) -> None:
        super(TNGNBNode, self).__post_init__()
        if self.entity_type is not EntityType.TN:
            raise ValueError("TNGNBNode must use EntityType.TN.")
        if self.role is not NodeRole.GNB:
            raise ValueError("TNGNBNode must use NodeRole.GNB.")
        if self.altitude_km < 0:
            raise ValueError("gNB altitude must be >= 0.")


@dataclass(frozen=True, slots=True)
class NTNUserNode(NetworkEntity):
    """NTN user node."""

    entity_type: EntityType = EntityType.NTN
    role: NodeRole = NodeRole.USER

    def __post_init__(self) -> None:
        super(NTNUserNode, self).__post_init__()
        if self.entity_type is not EntityType.NTN:
            raise ValueError("NTNUserNode must use EntityType.NTN.")
        if self.role is not NodeRole.USER:
            raise ValueError("NTNUserNode must use NodeRole.USER.")
        if self.altitude_km < 0:
            raise ValueError("User altitude must be >= 0.")


PositionLike = Position3D | NetworkEntity


def _resolve_position(value: PositionLike) -> Position3D:
    if isinstance(value, Position3D):
        return value
    if isinstance(value, NetworkEntity):
        return value.position
    raise TypeError("Distance helpers require Position3D or NetworkEntity inputs.")


def horizontal_distance_km(first: PositionLike, second: PositionLike) -> float:
    """Calculate 2D distance in the horizontal plane."""
    first_position = _resolve_position(first)
    second_position = _resolve_position(second)
    return dist((first_position.x_km, first_position.y_km), (second_position.x_km, second_position.y_km))


def altitude_difference_km(first: PositionLike, second: PositionLike) -> float:
    """Calculate absolute altitude separation."""
    first_position = _resolve_position(first)
    second_position = _resolve_position(second)
    return abs(first_position.altitude_km - second_position.altitude_km)


def distance_km(first: PositionLike, second: PositionLike) -> float:
    """Calculate 3D Euclidean distance."""
    first_position = _resolve_position(first)
    second_position = _resolve_position(second)
    return dist(
        (first_position.x_km, first_position.y_km, first_position.altitude_km),
        (second_position.x_km, second_position.y_km, second_position.altitude_km),
    )


TNode = TypeVar("TNode", bound=NetworkEntity)


def _validate_nodes(nodes: Iterable[NetworkEntity], expected_type: type[TNode], collection_name: str) -> tuple[TNode, ...]:
    node_tuple = tuple(nodes)
    for node in node_tuple:
        if not isinstance(node, expected_type):
            raise TypeError(f"{collection_name} must contain only {expected_type.__name__}.")
    return node_tuple


@dataclass(frozen=True, slots=True)
class NetworkScenario:
    """Scenario-level node container for generator handoff."""

    satellites: tuple[SatelliteNode, ...] = ()
    haps: tuple[HAPNode, ...] = ()
    gnbs: tuple[TNGNBNode, ...] = ()
    users: tuple[NTNUserNode, ...] = ()
    enforce_baseline_realism: bool = True

    def __post_init__(self) -> None:
        satellites = _validate_nodes(self.satellites, SatelliteNode, "satellites")
        haps = _validate_nodes(self.haps, HAPNode, "haps")
        gnbs = _validate_nodes(self.gnbs, TNGNBNode, "gnbs")
        users = _validate_nodes(self.users, NTNUserNode, "users")

        object.__setattr__(self, "satellites", satellites)
        object.__setattr__(self, "haps", haps)
        object.__setattr__(self, "gnbs", gnbs)
        object.__setattr__(self, "users", users)

        all_ids = [node.entity_id for node in self.all_nodes]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Scenario entity_id values must be unique.")

        if self.enforce_baseline_realism:
            if len(self.haps) < 1:
                raise ValueError("Baseline requires at least one HAP node.")
            if len(self.gnbs) != 1:
                raise ValueError("Baseline requires exactly 1 gNB node.")
            if not (4 <= len(self.users) <= 5):
                raise ValueError("Baseline requires user count in [4, 5].")

    @property
    def all_nodes(self) -> tuple[NetworkEntity, ...]:
        """Flatten all node collections in a deterministic order."""
        return (*self.satellites, *self.haps, *self.gnbs, *self.users)

