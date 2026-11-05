"""Turn time-stepped simulation runs into LSTM-ready (N, 10, 3) windows.

Splitting is done per RUN, never per window: neighbouring windows of one run
overlap heavily, so a random window split would leak test data into training
and inflate accuracy. Runs are assigned to train/val/test *within each*
(user_count, ACIR) stratum so every split sees every operating point.

Prediction target (``DatasetBuildConfig.target``):

* ``local_mean`` (primary): interference ``horizon`` slots ahead with the fast-fading term
  removed (``interference_local_dbm``). The inputs still contain the faded interference.
* ``instantaneous``: the faded interference itself (``interference_dbm``), reported alongside
  because it contains an irreducible fast-fading component.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import SimulationConfig
from .phase3_data import (
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_LOOKBACK_SLOTS,
    Phase3DatasetSplit,
    save_phase3_dataset_split,
)
from .reproducibility import collect_run_info
from .timeseries import FEATURE_NAMES, TimeSeriesConfig, simulate_time_series

# target key -> TimeSeriesRun attribute / metadata name
TARGET_ATTRIBUTES = {"local_mean": "interference_local_dbm", "instantaneous": "interference_dbm"}
TARGET_NAME = "interference_dbm"  # legacy name of the instantaneous target / persistence feature
TARGET_INDEX = FEATURE_NAMES.index(TARGET_NAME)
MASKABLE_FEATURES = ("signal_dbm", "sinr_db")  # interference_dbm is the persistence baseline input
SEED_BLOCK = 100_000  # max simulated runs per scenario before seed blocks could collide


def scenario_seed_offset(scenario: str) -> int:
    """Stable per-scenario seed block so urban and rural never share run seeds."""
    return (zlib.crc32(scenario.encode("utf-8")) % 10_000) * SEED_BLOCK


def dataset_dir_name(scenario: str, target: str = "local_mean", mask_features: tuple[str, ...] = ()) -> str:
    """Folder name for one dataset variant: ``urban``, ``urban_instantaneous``, ``urban_nosinr`` ..."""
    name = scenario
    if target != "local_mean":
        name += f"_{target}"
    if mask_features:
        name += "_no" + "_no".join(feature.replace("_dbm", "").replace("_db", "") for feature in mask_features)
    return name


@dataclass(frozen=True)
class DatasetBuildConfig:
    """What to simulate and how to window it."""

    scenario: str
    user_counts: tuple[int, ...] = (4, 5)
    acir_points_db: tuple[float, ...] | None = None  # None -> SimulationConfig.acir_sweep_db
    runs_per_point: int = 6
    horizon_slots: int = 1
    base_seed: int = 1000
    split_seed: int = 42
    target: str = "local_mean"  # "local_mean" | "instantaneous"
    # Feature channels zero-filled to keep the (N, 10, 3) shape (ablation, e.g. ("sinr_db",)).
    mask_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.runs_per_point < 1:
            raise ValueError("runs_per_point must be >= 1.")
        if not (1 <= self.horizon_slots <= 3):
            raise ValueError("horizon_slots must be in [1, 3] (project plan: 1-3 slots ahead).")
        if not self.user_counts:
            raise ValueError("user_counts cannot be empty.")
        if self.target not in TARGET_ATTRIBUTES:
            raise ValueError(f"target must be one of {sorted(TARGET_ATTRIBUTES)}.")
        for feature in self.mask_features:
            if feature not in MASKABLE_FEATURES:
                raise ValueError(f"mask_features entries must be in {MASKABLE_FEATURES}, got {feature!r}.")


def make_windows(
    series: np.ndarray,
    *,
    lookback: int = EXPECTED_LOOKBACK_SLOTS,
    horizon: int = 1,
    target_index: int = TARGET_INDEX,
    targets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over one (T, F) series -> X (N, lookback, F), y (N,).

    X[i] covers slots i .. i+lookback-1; y[i] is the target at slot i+lookback-1+horizon
    (i.e. ``horizon`` slots after the window ends). The target is column ``target_index`` of
    ``series`` unless a separate (T,) ``targets`` series is given (e.g. the local-mean series).
    """
    arr = np.asarray(series, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("series must have shape (T, F).")
    n_windows = arr.shape[0] - lookback - horizon + 1
    if n_windows <= 0:
        raise ValueError("series too short for the requested lookback and horizon.")
    windows = sliding_window_view(arr, lookback, axis=0)  # (T-lookback+1, F, lookback)
    x = windows[:n_windows].transpose(0, 2, 1).copy()
    first_target = lookback + horizon - 1
    if targets is None:
        y = arr[first_target : first_target + n_windows, target_index].copy()
    else:
        target_series = np.asarray(targets, dtype=np.float32)
        if target_series.shape != (arr.shape[0],):
            raise ValueError("targets must have shape (T,) matching series.")
        y = target_series[first_target : first_target + n_windows].copy()
    return x, y


def _half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def stratum_split_counts(run_count: int) -> tuple[int, int, int]:
    """(train, val, test) run counts for one stratum: val/test via round-half-up, >= 1 each."""
    n_val = max(1, _half_up(run_count * DEFAULT_VAL_RATIO))
    n_test = max(1, _half_up(run_count * DEFAULT_TEST_RATIO))
    n_train = run_count - n_val - n_test
    if n_train < 1:
        raise ValueError(
            "Need at least 3 runs per (user_count, ACIR) stratum to fill train/val/test; "
            "increase runs_per_point."
        )
    return n_train, n_val, n_test


def assign_run_splits(strata: list[tuple[Any, ...]], seed: int) -> list[str]:
    """Assign each run to train/val/test, stratified by the key of each run.

    ``strata[i]`` identifies run ``i``'s stratum (e.g. ``(user_count, acir_db)``). Within every
    stratum runs are shuffled with a seeded generator, then split by :func:`stratum_split_counts`.
    """
    rng = np.random.default_rng(seed)
    assigned = [""] * len(strata)
    for key in sorted(set(strata)):
        members = [i for i, s in enumerate(strata) if s == key]
        n_train, n_val, _ = stratum_split_counts(len(members))
        labels = ["train"] * n_train + ["val"] * n_val + ["test"] * (len(members) - n_train - n_val)
        for position, member_offset in enumerate(rng.permutation(len(members))):
            assigned[members[int(member_offset)]] = labels[position]
    return assigned


def build_phase3_dataset(
    sim_config: SimulationConfig,
    ts_config: TimeSeriesConfig,
    build: DatasetBuildConfig,
) -> tuple[Phase3DatasetSplit, dict[str, Any]]:
    """Simulate all runs, window them, and return a run-wise train/val/test split."""
    acir_points = build.acir_points_db if build.acir_points_db is not None else sim_config.acir_sweep_db
    plan = [
        (user_count, float(acir_db), run_index)
        for user_count in build.user_counts
        for acir_db in acir_points
        for run_index in range(build.runs_per_point)
    ]
    split_of_run = assign_run_splits([(user_count, acir_db) for user_count, acir_db, _ in plan], build.split_seed)
    seed_offset = scenario_seed_offset(build.scenario)
    target_attribute = TARGET_ATTRIBUTES[build.target]
    masked_columns = [FEATURE_NAMES.index(name) for name in build.mask_features]

    parts: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {s: ([], []) for s in ("train", "val", "test")}
    runs_meta: list[dict[str, Any]] = []
    for index, ((user_count, acir_db, run_index), split_name) in enumerate(zip(plan, split_of_run)):
        seed = build.base_seed + seed_offset + index
        run = simulate_time_series(
            sim_config, ts_config, scenario_name=build.scenario, user_count=user_count, acir_db=acir_db, seed=seed
        )
        features = run.features()  # (users, slots, 3)
        if masked_columns:
            features = features.copy()
            features[:, :, masked_columns] = 0.0
        target_series = getattr(run, target_attribute)  # (users, slots)
        window_count = 0
        for user_series, user_target in zip(features, target_series):
            x, y = make_windows(user_series, horizon=build.horizon_slots, targets=user_target)
            parts[split_name][0].append(x)
            parts[split_name][1].append(y)
            window_count += x.shape[0]
        runs_meta.append(
            {"run_id": f"{build.scenario}-u{user_count}-acir{acir_db:g}-r{run_index}", "user_count": user_count,
             "acir_db": acir_db, "seed": seed, "split": split_name, "windows": window_count}
        )

    def stacked(name: str) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = parts[name]
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    x_train, y_train = stacked("train")
    x_val, y_val = stacked("val")
    x_test, y_test = stacked("test")
    split = Phase3DatasetSplit(x_train, y_train, x_val, y_val, x_test, y_test)

    metadata: dict[str, Any] = {
        "scenario": build.scenario,
        "feature_names": list(FEATURE_NAMES),
        "masked_features": list(build.mask_features),
        "target_kind": build.target,
        "target": target_attribute,
        "target_definition": (
            "interference (dBm) horizon_slots ahead with the fast-fading term removed"
            if build.target == "local_mean"
            else "instantaneous interference (dBm) horizon_slots ahead, including fast fading"
        ),
        "persistence_baseline": "last observed value of feature interference_dbm (faded)",
        "lookback_slots": EXPECTED_LOOKBACK_SLOTS,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "horizon_slots": build.horizon_slots,
        "slot_duration_s": ts_config.slot_duration_s,
        "n_slots_per_run": ts_config.n_slots,
        "carrier_frequency_hz": sim_config.radio.carrier_frequency_hz,
        "doppler_hz": ts_config.doppler_hz(sim_config.radio.carrier_frequency_hz),
        "fast_fading_rho": ts_config.fast_fading_rho(sim_config.radio.carrier_frequency_hz),
        "timeseries_config": asdict(ts_config),
        "acir_points_db": [float(a) for a in acir_points],
        "user_counts": list(build.user_counts),
        "runs_per_point": build.runs_per_point,
        "base_seed": build.base_seed,
        "scenario_seed_offset": seed_offset,
        "split_seed": build.split_seed,
        "split_unit": "run, stratified by (user_count, acir_db); no window overlap between splits",
        "window_counts": {"train": int(x_train.shape[0]), "val": int(x_val.shape[0]), "test": int(x_test.shape[0])},
        "runs": runs_meta,
        "provenance": collect_run_info(build.base_seed),
    }
    return split, metadata


def save_dataset(split: Phase3DatasetSplit, metadata: dict[str, Any], out_dir: str | Path) -> Path:
    """Write X_train.npy ... y_test.npy (names the Phase 3 loader expects) + metadata.json."""
    root = Path(out_dir)
    save_phase3_dataset_split(split, root)  # no prefix: loader only reads unprefixed names
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return root
