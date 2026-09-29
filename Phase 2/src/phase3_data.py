"""Phase 3 dataset loading, validation, and split helpers."""

from __future__ import annotations

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


def split_phase3_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
) -> Phase3DatasetSplit:
    """Deterministically split a full Phase 3 dataset."""
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

    train_count = int(np.floor(sample_count * train_ratio))
    val_count = int(np.floor(sample_count * val_ratio))
    test_count = sample_count - train_count - val_count
    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError("Split ratios produced an empty split; provide a larger dataset.")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(sample_count)
    train_idx = shuffled_indices[:train_count]
    val_idx = shuffled_indices[train_count : train_count + val_count]
    test_idx = shuffled_indices[train_count + val_count :]

    return Phase3DatasetSplit(
        X_train=validated_features[train_idx],
        y_train=validated_targets[train_idx],
        X_val=validated_features[val_idx],
        y_val=validated_targets[val_idx],
        X_test=validated_features[test_idx],
        y_test=validated_targets[test_idx],
    )


def _try_load_presplit(root: Path, *, scenario: str | None = None) -> Phase3DatasetSplit | None:
    naming_variants: list[tuple[str, str, str, str, str, str]] = [
        ("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy")
    ]
    if scenario:
        naming_variants.append(
            (
                f"X_{scenario}_train.npy",
                f"y_{scenario}_train.npy",
                f"X_{scenario}_val.npy",
                f"y_{scenario}_val.npy",
                f"X_{scenario}_test.npy",
                f"y_{scenario}_test.npy",
            )
        )

    for names in naming_variants:
        paths = [root / name for name in names]
        if all(path.exists() for path in paths):
            return Phase3DatasetSplit(
                X_train=_load_array(paths[0]),
                y_train=_load_array(paths[1]),
                X_val=_load_array(paths[2]),
                y_val=_load_array(paths[3]),
                X_test=_load_array(paths[4]),
                y_test=_load_array(paths[5]),
            )
    return None


def load_phase3_dataset(
    data_root: str | Path,
    *,
    scenario: str | None = None,
    seed: int = 42,
    require_pre_split: bool = False,
) -> Phase3DatasetSplit:
    """Load Phase 3 dataset from pre-split files or from X/y arrays."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    pre_split = _try_load_presplit(root, scenario=scenario)
    if pre_split is not None:
        return pre_split
    if require_pre_split:
        raise FileNotFoundError("Pre-split files were required but not found under the supplied directory.")

    unsplit_name_candidates: list[tuple[str, str]] = []
    if scenario:
        unsplit_name_candidates.append((f"X_{scenario}.npy", f"y_{scenario}.npy"))
    unsplit_name_candidates.append(("X.npy", "y.npy"))

    for x_name, y_name in unsplit_name_candidates:
        x_path = root / x_name
        y_path = root / y_name
        if x_path.exists() and y_path.exists():
            return split_phase3_dataset(np.load(x_path), np.load(y_path), seed=seed)

    supported = ", ".join(
        [
            "X_train.npy/y_train.npy + X_val.npy/y_val.npy + X_test.npy/y_test.npy",
            "X_<scenario>_<split>.npy + y_<scenario>_<split>.npy",
            "X.npy + y.npy",
            "X_<scenario>.npy + y_<scenario>.npy",
        ]
    )
    raise FileNotFoundError(f"No compatible Phase 3 dataset files found under {root}. Supported patterns: {supported}.")


def save_phase3_dataset_split(
    split: Phase3DatasetSplit,
    output_root: str | Path,
    *,
    prefix: str = "",
) -> None:
    """Persist a validated split as .npy arrays."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    normalized_prefix = f"{prefix}_" if prefix else ""

    np.save(root / f"{normalized_prefix}X_train.npy", split.X_train)
    np.save(root / f"{normalized_prefix}y_train.npy", split.y_train)
    np.save(root / f"{normalized_prefix}X_val.npy", split.X_val)
    np.save(root / f"{normalized_prefix}y_val.npy", split.y_val)
    np.save(root / f"{normalized_prefix}X_test.npy", split.X_test)
    np.save(root / f"{normalized_prefix}y_test.npy", split.y_test)
