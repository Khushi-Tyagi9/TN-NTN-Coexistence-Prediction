"""Tests for typed entities and geometry helpers."""

import pytest

from src.entities import (
    EntityType,
    HAPNode,
    NetworkScenario,
    NTNUserNode,
    Position3D,
    SatelliteNode,
    TNGNBNode,
    altitude_difference_km,
    distance_km,
    horizontal_distance_km,
)


def test_entity_creation_for_all_node_types() -> None:
    """All node dataclasses should validate and expose typed defaults."""
    satellite = SatelliteNode(entity_id="sat-1", position=Position3D.from_xy_km(0.0, 0.0, altitude_km=550.0))
    hap = HAPNode(
        entity_id="hap-1",
        position=Position3D.from_xy_km(0.0, 1.0, altitude_km=20.0),
        position_model="quasi_static",
    )
    gnbs = (
        TNGNBNode(entity_id="gnb-1", position=Position3D.from_xy_km(1.0, 0.0, altitude_km=0.05)),
    )
    users = tuple(
        NTNUserNode(entity_id=f"ue-{index}", position=Position3D.from_xy_km(index * 0.1, 0.0))
        for index in range(1, 5)
    )
    scenario = NetworkScenario(satellites=(satellite,), haps=(hap,), gnbs=gnbs, users=users)

    assert satellite.entity_type is EntityType.NTN
    assert hap.position_model.value == "quasi_static"
    assert all(gnb.entity_type is EntityType.TN for gnb in gnbs)
    assert len(scenario.all_nodes) == 7


def test_validation_fails_for_invalid_positions_and_types() -> None:
    """Entity constructors should reject invalid position/type combinations."""
    with pytest.raises(ValueError, match="finite"):
        Position3D(x_km=float("nan"), y_km=0.0, z_km=0.0)

    with pytest.raises(ValueError, match="EntityType.NTN"):
        SatelliteNode(
            entity_id="sat-invalid",
            position=Position3D.from_xy_km(0.0, 0.0, altitude_km=550.0),
            entity_type=EntityType.TN,
        )


def test_network_scenario_validation_fails_for_invalid_counts_and_collections() -> None:
    """Scenario container should enforce count bounds and typed collections."""
    hap = HAPNode(entity_id="hap-1", position=Position3D.from_xy_km(0.0, 0.0, altitude_km=20.0))
    gnbs = (
        TNGNBNode(entity_id="gnb-1", position=Position3D.from_xy_km(1.0, 0.0)),
    )
    user = NTNUserNode(entity_id="ue-1", position=Position3D.from_xy_km(0.1, 0.1))

    with pytest.raises(ValueError, match="user count in \\[4, 5\\]"):
        NetworkScenario(haps=(hap,), gnbs=gnbs, users=(user,))

    with pytest.raises(TypeError, match="users must contain only NTNUserNode"):
        NetworkScenario(haps=(hap,), gnbs=gnbs, users=(gnbs[0],), enforce_baseline_realism=False)  # type: ignore[arg-type]


def test_distance_helpers_return_expected_values() -> None:
    """Distance helpers should provide consistent horizontal/vertical/3D results."""
    origin = Position3D.from_xy_km(0.0, 0.0, altitude_km=0.0)
    point = Position3D.from_xy_km(3.0, 4.0, altitude_km=12.0)

    assert horizontal_distance_km(origin, point) == pytest.approx(5.0)
    assert altitude_difference_km(origin, point) == pytest.approx(12.0)
    assert distance_km(origin, point) == pytest.approx(13.0)

