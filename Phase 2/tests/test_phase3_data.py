"""Tests for Phase 3 dataset loading and deterministic split behavior."""

import numpy as np

from src.phase3_data import load_phase3_dataset, split_phase3_dataset


def test_split_phase3_dataset_uses_expected_70_15_15_counts() -> None:
    sample_count = 1189
    x = np.ones((sample_count, 10, 3), dtype=np.float32)
    y = np.linspace(1.0, 2.0, sample_count, dtype=np.float32)

    split = split_phase3_dataset(x, y, seed=42)

    assert split.X_train.shape[0] == 832
    assert split.X_val.shape[0] == 178
    assert split.X_test.shape[0] == 179


def test_load_phase3_dataset_from_unsplit_files(tmp_path) -> None:
    sample_count = 1189
    x = np.random.default_rng(7).normal(size=(sample_count, 10, 3)).astype(np.float32)
    y = np.random.default_rng(11).normal(loc=1.0, scale=0.1, size=(sample_count,)).astype(np.float32)
    np.save(tmp_path / "X.npy", x)
    np.save(tmp_path / "y.npy", y)

    split = load_phase3_dataset(tmp_path, seed=17)

    assert split.sample_count == sample_count
    assert split.X_train.shape[1:] == (10, 3)
    assert split.X_val.shape[1:] == (10, 3)
    assert split.X_test.shape[1:] == (10, 3)
