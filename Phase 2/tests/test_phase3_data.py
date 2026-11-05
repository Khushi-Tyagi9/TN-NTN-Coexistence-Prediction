"""Tests for Phase 3 dataset loading and deterministic split behavior."""

import numpy as np

from src.phase3_data import load_phase3_dataset, split_phase3_dataset


def test_split_phase3_dataset_uses_expected_70_15_15_counts() -> None:
    sample_count = 1189
    x = np.ones((sample_count, 10, 3), dtype=np.float32)
    y = np.linspace(1.0, 2.0, sample_count, dtype=np.float32)

    split = split_phase3_dataset(x, y, seed=42, allow_window_shuffle=True)  # i.i.d. fixture rows

    assert split.X_train.shape[0] == 832
    assert split.X_val.shape[0] == 178
    assert split.X_test.shape[0] == 179


def test_load_phase3_dataset_from_unsplit_files(tmp_path) -> None:
    sample_count = 1189
    x = np.random.default_rng(7).normal(size=(sample_count, 10, 3)).astype(np.float32)
    y = np.random.default_rng(11).normal(loc=1.0, scale=0.1, size=(sample_count,)).astype(np.float32)
    np.save(tmp_path / "X.npy", x)
    np.save(tmp_path / "y.npy", y)

    split = load_phase3_dataset(tmp_path, seed=17, allow_window_shuffle=True)  # i.i.d. fixture rows

    assert split.sample_count == sample_count
    assert split.X_train.shape[1:] == (10, 3)
    assert split.X_val.shape[1:] == (10, 3)
    assert split.X_test.shape[1:] == (10, 3)


# --- audit C2: overlapping windows must not straddle splits --------------------------------------

import pytest  # noqa: E402

from src.phase3_data import window_overlap_fraction  # noqa: E402


def _stride1_windows(n_runs: int = 20, slots: int = 60, lookback: int = 10):
    """Stride-1 windows from independent random-walk runs, plus a run id per window."""
    xs, ys, groups = [], [], []
    for run in range(n_runs):
        rng = np.random.default_rng(run)
        series = np.cumsum(rng.normal(size=(slots, 3)), axis=0).astype(np.float32)
        for i in range(slots - lookback):
            xs.append(series[i : i + lookback])
            ys.append(series[i + lookback, 0])
            groups.append(run)
    return np.stack(xs), np.array(ys, dtype=np.float32), np.array(groups)


def test_group_split_has_no_window_overlap_between_splits() -> None:
    x, y, groups = _stride1_windows()

    split = split_phase3_dataset(x, y, group_ids=groups, seed=3)

    assert window_overlap_fraction(split.X_train, split.X_test) == 0.0
    assert window_overlap_fraction(split.X_train, split.X_val) == 0.0
    assert window_overlap_fraction(split.X_val, split.X_test) == 0.0
    assert split.sample_count == len(y)


def test_window_shuffle_leaks_and_the_leak_check_detects_it() -> None:
    x, y, _ = _stride1_windows()

    with pytest.warns(UserWarning, match="leaks"):
        leaky = split_phase3_dataset(x, y, seed=3, allow_window_shuffle=True)

    assert window_overlap_fraction(leaky.X_train, leaky.X_test) > 0.5


def test_split_without_groups_is_refused_by_default() -> None:
    x, y, _ = _stride1_windows(n_runs=4)
    with pytest.raises(ValueError, match="group_ids are required"):
        split_phase3_dataset(x, y)


def test_group_split_keeps_each_group_in_one_split_and_is_deterministic() -> None:
    x, y, groups = _stride1_windows(n_runs=10)

    first = split_phase3_dataset(x, y, group_ids=groups, seed=5)
    second = split_phase3_dataset(x, y, group_ids=groups, seed=5)
    other = split_phase3_dataset(x, y, group_ids=groups, seed=6)

    assert np.array_equal(first.X_test, second.X_test)
    assert not np.array_equal(first.X_test, other.X_test)
    # 10 runs -> round(1.5)=2 val, 2 test, 6 train; windows per run are equal (50)
    assert (first.X_train.shape[0], first.X_val.shape[0], first.X_test.shape[0]) == (300, 100, 100)


def test_group_split_input_validation() -> None:
    x, y, groups = _stride1_windows(n_runs=4)
    with pytest.raises(ValueError, match="one id per window"):
        split_phase3_dataset(x, y, group_ids=groups[:-1])
    with pytest.raises(ValueError, match="At least 3 groups"):
        split_phase3_dataset(x[groups < 2], y[groups < 2], group_ids=groups[groups < 2])


def test_window_overlap_fraction_validation_and_shift_detection() -> None:
    x, _, _ = _stride1_windows(n_runs=1, slots=30)
    # window i+1 is window i shifted by one slot -> 9 shared slots
    assert window_overlap_fraction(x[:1], x[1:2]) == 1.0
    assert window_overlap_fraction(x[:1], x[15:16]) == 0.0
    with pytest.raises(ValueError):
        window_overlap_fraction(x, x, min_shared_slots=10)


def test_loader_uses_groups_file_and_refuses_unsplit_without_it(tmp_path) -> None:
    x, y, groups = _stride1_windows(n_runs=12)
    np.save(tmp_path / "X.npy", x)
    np.save(tmp_path / "y.npy", y)

    with pytest.raises(ValueError, match="group_ids are required"):
        load_phase3_dataset(tmp_path)

    np.save(tmp_path / "groups.npy", groups)
    split = load_phase3_dataset(tmp_path, seed=1)
    assert window_overlap_fraction(split.X_train, split.X_test) == 0.0
    assert split.sample_count == len(y)


def test_loader_prefers_scenario_specific_groups_file(tmp_path) -> None:
    x, y, groups = _stride1_windows(n_runs=12)
    np.save(tmp_path / "X_urban.npy", x)
    np.save(tmp_path / "y_urban.npy", y)
    np.save(tmp_path / "groups_urban.npy", groups)

    split = load_phase3_dataset(tmp_path, scenario="urban", seed=1)

    assert window_overlap_fraction(split.X_train, split.X_test) == 0.0


# --- audit M1: saver/loader file naming must round-trip -------------------------------------------

from src.phase3_data import Phase3DatasetSplit, save_phase3_dataset_split  # noqa: E402


def _tiny_split(offset: float = 0.0) -> Phase3DatasetSplit:
    rng = np.random.default_rng(int(offset) + 1)

    def part(n: int):
        return rng.normal(size=(n, 10, 3)).astype(np.float32) + offset, rng.normal(size=n).astype(np.float32) + offset

    (xt, yt), (xv, yv), (xs, ys) = part(8), part(3), part(3)
    return Phase3DatasetSplit(xt, yt, xv, yv, xs, ys)


def _assert_same(a: Phase3DatasetSplit, b: Phase3DatasetSplit) -> None:
    for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


def test_round_trip_without_scenario(tmp_path) -> None:
    split = _tiny_split()
    save_phase3_dataset_split(split, tmp_path)

    assert (tmp_path / "X_train.npy").exists()
    _assert_same(load_phase3_dataset(tmp_path, require_pre_split=True), split)


def test_round_trip_with_scenario_uses_canonical_names(tmp_path) -> None:
    split = _tiny_split()
    save_phase3_dataset_split(split, tmp_path, scenario="urban")

    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f"{a}_urban_{s}.npy" for a in ("X", "y") for s in ("train", "val", "test")
    )
    _assert_same(load_phase3_dataset(tmp_path, scenario="urban", require_pre_split=True), split)


def test_two_scenarios_share_one_folder_without_mixing(tmp_path) -> None:
    urban, rural = _tiny_split(0.0), _tiny_split(100.0)
    save_phase3_dataset_split(urban, tmp_path, scenario="urban")
    save_phase3_dataset_split(rural, tmp_path, scenario="rural")

    _assert_same(load_phase3_dataset(tmp_path, scenario="urban"), urban)
    _assert_same(load_phase3_dataset(tmp_path, scenario="rural"), rural)


def test_scenario_files_win_over_unprefixed_files(tmp_path) -> None:
    plain, urban = _tiny_split(0.0), _tiny_split(50.0)
    save_phase3_dataset_split(plain, tmp_path)
    save_phase3_dataset_split(urban, tmp_path, scenario="urban")

    _assert_same(load_phase3_dataset(tmp_path, scenario="urban"), urban)
    _assert_same(load_phase3_dataset(tmp_path), plain)
    _assert_same(load_phase3_dataset(tmp_path, scenario="rural"), plain)  # no rural files -> plain fallback


def test_deprecated_prefix_now_writes_loadable_names(tmp_path) -> None:
    split = _tiny_split()
    with pytest.warns(DeprecationWarning, match="prefix"):
        save_phase3_dataset_split(split, tmp_path, prefix="urban")

    assert (tmp_path / "X_urban_train.npy").exists() and not (tmp_path / "urban_X_train.npy").exists()
    _assert_same(load_phase3_dataset(tmp_path, scenario="urban"), split)  # the original bug: this raised
    with pytest.raises(ValueError, match="not both"):
        save_phase3_dataset_split(split, tmp_path, scenario="rural", prefix="urban")


def test_legacy_prefixed_files_still_load_with_a_warning(tmp_path) -> None:
    split = _tiny_split()
    for name, array in zip(
        ("urban_X_train", "urban_y_train", "urban_X_val", "urban_y_val", "urban_X_test", "urban_y_test"),
        (split.X_train, split.y_train, split.X_val, split.y_val, split.X_test, split.y_test),
    ):
        np.save(tmp_path / f"{name}.npy", array)

    with pytest.warns(DeprecationWarning, match="legacy file names"):
        loaded = load_phase3_dataset(tmp_path, scenario="urban")
    _assert_same(loaded, split)


def test_missing_split_file_is_not_silently_ignored(tmp_path) -> None:
    save_phase3_dataset_split(_tiny_split(), tmp_path, scenario="urban")
    (tmp_path / "y_urban_test.npy").unlink()

    with pytest.raises(FileNotFoundError):
        load_phase3_dataset(tmp_path, scenario="urban")


def test_load_train_val_arrays_never_opens_test_files(tmp_path) -> None:
    from src.phase3_data import load_train_val_arrays

    split = _tiny_split()
    save_phase3_dataset_split(split, tmp_path)
    (tmp_path / "X_test.npy").unlink()
    (tmp_path / "y_test.npy").unlink()

    x_train, y_train, x_val, y_val = load_train_val_arrays(tmp_path)

    assert np.array_equal(x_train, split.X_train) and np.array_equal(y_val, split.y_val)
    assert x_val.shape == split.X_val.shape and y_train.shape == split.y_train.shape


def test_load_train_val_arrays_validates(tmp_path) -> None:
    from src.phase3_data import load_train_val_arrays

    with pytest.raises(FileNotFoundError):
        load_train_val_arrays(tmp_path)
    split = _tiny_split()
    save_phase3_dataset_split(split, tmp_path)
    np.save(tmp_path / "y_val.npy", split.y_val[:-1])
    with pytest.raises(ValueError, match="same number of samples"):
        load_train_val_arrays(tmp_path)
