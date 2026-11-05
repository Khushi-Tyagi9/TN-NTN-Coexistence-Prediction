"""Phase 3 dataset loading, validation, and split helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXPECTED_LOOKBACK_SLOTS = 10
EXPECTED_FEATURE_COUNT = 3
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15


def _as_float_array(array: np.ndarray, *, field_name: str) -> np.ndarray:
    casted = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(casted)):
        raise ValueError(f"{field_name} must contain finite values only.")
    return casted


def _validate_feature_tensor(features: np.ndarray, *, field_name: str) -> np.ndarray:
    validated = _as_float_array(features, field_name=field_name)
    if validated.ndim != 3:
        raise ValueError(f"{field_name} must have shape (N, 10, 3).")
    if validated.shape[1] != EXPECTED_LOOKBACK_SLOTS:
        raise ValueError(
            f"{field_name} lookback dimension must be {EXPECTED_LOOKBACK_SLOTS}, got {validated.shape[1]}."
        )
    if validated.shape[2] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"{field_name} feature dimension must be {EXPECTED_FEATURE_COUNT}, got {validated.shape[2]}."
        )
    return validated


def _validate_target_vector(targets: np.ndarray, *, field_name: str) -> np.ndarray:
    validated = _as_float_array(targets, field_name=field_name)
    if validated.ndim == 2 and validated.shape[1] == 1:
        validated = validated[:, 0]
    if validated.ndim != 1:
        raise ValueError(f"{field_name} must have shape (N,) or (N, 1).")
    return validated


def _validate_pair(features: np.ndarray, targets: np.ndarray, *, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    validated_features = _validate_feature_tensor(features, field_name=f"{split_name}.features")
    validated_targets = _validate_target_vector(targets, field_name=f"{split_name}.targets")
    if validated_features.shape[0] != validated_targets.shape[0]:
        raise ValueError(
            f"{split_name} sample mismatch: features={validated_features.shape[0]} vs targets={validated_targets.shape[0]}."
        )
    return validated_features, validated_targets


def _load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return np.load(path)


@dataclass(frozen=True, slots=True)
class Phase3DatasetSplit:
    """Train/validation/test split for Phase 3 prediction."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def __post_init__(self) -> None:
        x_train, y_train = _validate_pair(self.X_train, self.y_train, split_name="train")
        x_val, y_val = _validate_pair(self.X_val, self.y_val, split_name="validation")
        x_test, y_test = _validate_pair(self.X_test, self.y_test, split_name="test")
        object.__setattr__(self, "X_train", x_train)
        object.__setattr__(self, "y_train", y_train)
        object.__setattr__(self, "X_val", x_val)
        object.__setattr__(self, "y_val", y_val)
        object.__setattr__(self, "X_test", x_test)
        object.__setattr__(self, "y_test", y_test)

    @property
    def sample_count(self) -> int:
        return int(self.X_train.shape[0] + self.X_val.shape[0] + self.X_test.shape[0])


def _group_split_counts(group_count: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """(train, val, test) group counts; val/test use round() and are at least 1 each."""
    if group_count < 3:
        raise ValueError("At least 3 groups (runs) are required for train/val/test splitting.")
    n_val = max(1, int(round(group_count * val_ratio)))
    n_test = max(1, int(round(group_count * test_ratio)))
    n_train = group_count - n_val - n_test
    if n_train < 1:
        raise ValueError("Split ratios produced an empty train split; provide more groups.")
    return n_train, n_val, n_test


def _shuffled_window_indices(
    sample_count: int, train_ratio: float, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_count = int(np.floor(sample_count * train_ratio))
    val_count = int(np.floor(sample_count * val_ratio))
    test_count = sample_count - train_count - val_count
    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError("Split ratios produced an empty split; provide a larger dataset.")
    shuffled = np.random.default_rng(seed).permutation(sample_count)
    return shuffled[:train_count], shuffled[train_count : train_count + val_count], shuffled[train_count + val_count :]


def _grouped_window_indices(
    group_ids: np.ndarray, val_ratio: float, test_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = np.unique(group_ids)  # sorted -> deterministic given seed
    n_train, n_val, _ = _group_split_counts(unique_groups.shape[0], val_ratio, test_ratio)
    order = np.random.default_rng(seed).permutation(unique_groups.shape[0])
    shuffled_groups = unique_groups[order]
    train_groups = shuffled_groups[:n_train]
    val_groups = shuffled_groups[n_train : n_train + n_val]
    test_groups = shuffled_groups[n_train + n_val :]
    return (
        np.flatnonzero(np.isin(group_ids, train_groups)),
        np.flatnonzero(np.isin(group_ids, val_groups)),
        np.flatnonzero(np.isin(group_ids, test_groups)),
    )


def split_phase3_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    group_ids: np.ndarray | None = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
    allow_window_shuffle: bool = False,
) -> Phase3DatasetSplit:
    """Deterministically split a full Phase 3 dataset WITHOUT leaking overlapping windows.

    Windows are slid with stride 1, so window i+1 contains window i's target as an input.
    Shuffling individual windows therefore puts test labels into the training inputs.
    The split is done per *group* (``group_ids``: one id per window, e.g. the simulation run
    and user it came from); every group lands entirely in one of train/val/test.

    ``allow_window_shuffle=True`` restores the old per-window shuffle (leaky; emits a warning).
    It exists only for data that is genuinely i.i.d. per row (e.g. unit-test fixtures).
    """
    if seed < 0:
        raise ValueError("seed must be >= 0.")
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must be > 0.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")

    validated_features, validated_targets = _validate_pair(features, targets, split_name="full")
    sample_count = validated_features.shape[0]
    if sample_count < 3:
        raise ValueError("At least 3 samples are required for train/val/test splitting.")

    if group_ids is not None:
        groups = np.asarray(group_ids)
        if groups.ndim != 1 or groups.shape[0] != sample_count:
            raise ValueError("group_ids must be a 1-D array with one id per window.")
        train_idx, val_idx, test_idx = _grouped_window_indices(groups, val_ratio, test_ratio, seed)
    elif allow_window_shuffle:
        warnings.warn(
            "Splitting overlapping time-series windows at random leaks test targets into training "
            "inputs; pass group_ids (per-run ids) instead of allow_window_shuffle=True.",
            UserWarning,
            stacklevel=2,
        )
        train_idx, val_idx, test_idx = _shuffled_window_indices(sample_count, train_ratio, val_ratio, seed)
    else:
        raise ValueError(
            "group_ids are required: windows overlap in time, so a random window split leaks test "
            "targets into training. Provide per-window run ids (group_ids), or pass "
            "allow_window_shuffle=True only for data whose rows are independent."
        )

    return Phase3DatasetSplit(
        X_train=validated_features[train_idx],
        y_train=validated_targets[train_idx],
        X_val=validated_features[val_idx],
        y_val=validated_targets[val_idx],
        X_test=validated_features[test_idx],
        y_test=validated_targets[test_idx],
    )


def window_overlap_fraction(
    reference_windows: np.ndarray,
    probe_windows: np.ndarray,
    *,
    min_shared_slots: int = 5,
) -> float:
    """Fraction of ``probe_windows`` that overlap in time with some ``reference_windows`` window.

    Two stride-1 windows overlap when the last k slots of one equal the first k slots of the
    other (k >= ``min_shared_slots``), checked in both directions. 0.0 means no leakage of this
    kind between the two sets; use it on (train, test) to verify a split.
    """
    reference = _validate_feature_tensor(reference_windows, field_name="reference_windows")
    probe = _validate_feature_tensor(probe_windows, field_name="probe_windows")
    lookback = reference.shape[1]
    if not (1 <= min_shared_slots < lookback):
        raise ValueError("min_shared_slots must be in [1, lookback).")
    if probe.shape[0] == 0:
        return 0.0

    hit = np.zeros(probe.shape[0], dtype=bool)
    for shared in range(min_shared_slots, lookback):
        ref_heads = {reference[i, :shared].tobytes() for i in range(reference.shape[0])}
        ref_tails = {reference[i, lookback - shared :].tobytes() for i in range(reference.shape[0])}
        for i in range(probe.shape[0]):
            if hit[i]:
                continue
            if probe[i, lookback - shared :].tobytes() in ref_heads or probe[i, :shared].tobytes() in ref_tails:
                hit[i] = True
    return float(np.mean(hit))


_SPLIT_ORDER = ("train", "val", "test")


def _split_file_names(scenario: str | None, *, legacy: bool = False) -> tuple[str, ...]:
    """(X_train, y_train, X_val, y_val, X_test, y_test) file names.

    Canonical: ``X_<scenario>_<split>.npy`` (or ``X_<split>.npy`` without a scenario).
    Legacy (written by older ``save_phase3_dataset_split(prefix=...)``): ``<scenario>_X_<split>.npy``.
    """
    names: list[str] = []
    for split_name in _SPLIT_ORDER:
        for array_name in ("X", "y"):
            if scenario is None:
                names.append(f"{array_name}_{split_name}.npy")
            elif legacy:
                names.append(f"{scenario}_{array_name}_{split_name}.npy")
            else:
                names.append(f"{array_name}_{scenario}_{split_name}.npy")
    return tuple(names)


def _load_split_files(root: Path, names: tuple[str, ...]) -> Phase3DatasetSplit | None:
    paths = [root / name for name in names]
    if not all(path.exists() for path in paths):
        return None
    return Phase3DatasetSplit(
        X_train=_load_array(paths[0]),
        y_train=_load_array(paths[1]),
        X_val=_load_array(paths[2]),
        y_val=_load_array(paths[3]),
        X_test=_load_array(paths[4]),
        y_test=_load_array(paths[5]),
    )


def _try_load_presplit(root: Path, *, scenario: str | None = None) -> Phase3DatasetSplit | None:
    """Pre-split files. With a scenario, its own files win over the unprefixed ``X_train.npy`` set,
    so a shared folder can never silently serve another scenario's data."""
    if scenario:
        loaded = _load_split_files(root, _split_file_names(scenario))
        if loaded is not None:
            return loaded
        loaded = _load_split_files(root, _split_file_names(scenario, legacy=True))
        if loaded is not None:
            warnings.warn(
                f"Loaded legacy file names '{scenario}_X_<split>.npy'; re-save with "
                f"save_phase3_dataset_split(..., scenario='{scenario}') to get 'X_{scenario}_<split>.npy'.",
                DeprecationWarning,
                stacklevel=3,
            )
            return loaded
    return _load_split_files(root, _split_file_names(None))


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        if (root / name).exists():
            return root / name
    return None


def load_phase3_dataset(
    data_root: str | Path,
    *,
    scenario: str | None = None,
    seed: int = 42,
    require_pre_split: bool = False,
    allow_window_shuffle: bool = False,
) -> Phase3DatasetSplit:
    """Load Phase 3 dataset from pre-split files or from X/y arrays.

    Unsplit ``X.npy``/``y.npy`` are split per group using ``groups.npy`` (or
    ``groups_<scenario>.npy``), one run id per window. Without group ids the load fails
    unless ``allow_window_shuffle=True`` (leaky; see ``split_phase3_dataset``).
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    pre_split = _try_load_presplit(root, scenario=scenario)
    if pre_split is not None:
        return pre_split
    if require_pre_split:
        raise FileNotFoundError("Pre-split files were required but not found under the supplied directory.")

    unsplit_name_candidates: list[tuple[str, str]] = []
    group_name_candidates: list[str] = []
    if scenario:
        unsplit_name_candidates.append((f"X_{scenario}.npy", f"y_{scenario}.npy"))
        group_name_candidates.append(f"groups_{scenario}.npy")
    unsplit_name_candidates.append(("X.npy", "y.npy"))
    group_name_candidates.append("groups.npy")

    for x_name, y_name in unsplit_name_candidates:
        x_path = root / x_name
        y_path = root / y_name
        if x_path.exists() and y_path.exists():
            groups_path = _first_existing(root, group_name_candidates)
            group_ids = None if groups_path is None else np.load(groups_path)
            return split_phase3_dataset(
                np.load(x_path),
                np.load(y_path),
                group_ids=group_ids,
                seed=seed,
                allow_window_shuffle=allow_window_shuffle,
            )

    supported = ", ".join(
        [
            "X_train.npy/y_train.npy + X_val.npy/y_val.npy + X_test.npy/y_test.npy",
            "X_<scenario>_<split>.npy + y_<scenario>_<split>.npy",
            "X.npy + y.npy + groups.npy",
            "X_<scenario>.npy + y_<scenario>.npy + groups_<scenario>.npy",
        ]
    )
    raise FileNotFoundError(f"No compatible Phase 3 dataset files found under {root}. Supported patterns: {supported}.")


def save_phase3_dataset_split(
    split: Phase3DatasetSplit,
    output_root: str | Path,
    *,
    scenario: str | None = None,
    prefix: str = "",
) -> None:
    """Persist a validated split as .npy arrays that ``load_phase3_dataset`` can read back.

    Without ``scenario``: ``X_train.npy`` ... ``y_test.npy``. With ``scenario``:
    ``X_<scenario>_train.npy`` ... (several scenarios can share one folder).
    ``prefix`` is a deprecated alias of ``scenario``; it used to write ``<prefix>_X_train.npy``,
    which the loader never found, and now writes the canonical scenario names.
    """
    if prefix:
        if scenario is not None and scenario != prefix:
            raise ValueError("Pass either scenario or the deprecated prefix, not both.")
        warnings.warn("'prefix' is deprecated; use 'scenario'.", DeprecationWarning, stacklevel=2)
        scenario = prefix

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    names = _split_file_names(scenario)
    arrays = (
        split.X_train, split.y_train, split.X_val, split.y_val, split.X_test, split.y_test,
    )
    for name, array in zip(names, arrays):
        np.save(root / name, array)


def load_train_val_arrays(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ONLY ``X_train, y_train, X_val, y_val`` from a pre-split folder.

    The test files are never opened, so exploratory work (preliminary runs, figures) cannot touch the
    test split, which is reserved for the final protocol.
    """
    root = Path(data_dir)
    names = ("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy")
    x_train, y_train = _load_array(root / names[0]), _load_array(root / names[1])
    x_val, y_val = _load_array(root / names[2]), _load_array(root / names[3])
    x_train = _validate_feature_tensor(x_train, field_name="train.features")
    x_val = _validate_feature_tensor(x_val, field_name="validation.features")
    y_train = _validate_target_vector(y_train, field_name="train.targets")
    y_val = _validate_target_vector(y_val, field_name="validation.targets")
    if x_train.shape[0] != y_train.shape[0] or x_val.shape[0] != y_val.shape[0]:
        raise ValueError("features and targets must have the same number of samples.")
    return x_train, y_train, x_val, y_val
