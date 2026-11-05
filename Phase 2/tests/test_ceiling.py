"""Interference-free coverage ceiling measurement (Phase 4, first implementation task)."""

import json
import shutil

import numpy as np
import pytest

import measure_ceiling as mc
from src.config import SimulationConfig
from src.interference import compute_noise_power_dbm
from src.phase3_dataset_builder import DatasetBuildConfig, build_phase3_dataset, save_dataset
from src.timeseries import TimeSeriesConfig, simulate_time_series


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    """Two tiny datasets with the real 600-slot pass (the geometry matters), a few runs each."""
    root = tmp_path_factory.mktemp("data")
    for scenario in mc.SCENARIOS:
        split, metadata = build_phase3_dataset(
            SimulationConfig(), TimeSeriesConfig(),
            DatasetBuildConfig(scenario=scenario, user_counts=(4,), acir_points_db=(20.0, 30.0), runs_per_point=3),
        )
        save_dataset(split, metadata, root / scenario)
    return root


def test_analytic_share_confirms_the_hand_arithmetic() -> None:
    share = mc.analytic_nadir_line_share(SimulationConfig(), TimeSeriesConfig())

    assert share == pytest.approx(mc.HAND_ESTIMATE_SHARE, abs=0.005)  # |x| <= 909 km of 1125 km -> 0.808
    assert 0.80 < share < 0.82


def test_only_test_runs_are_measured_and_they_reproduce_x_test(data_root) -> None:
    metadata = json.loads((data_root / "urban" / "metadata.json").read_text(encoding="utf-8"))
    test_ids = [r["run_id"] for r in metadata["runs"] if r["split"] == "test"]

    result = mc.measure_scenario(data_root / "urban")

    assert [r["run_id"] for r in result["runs"]] == test_ids and len(test_ids) == 2
    assert result["n_test_runs"] == 2 and result["n_user_series"] == 8 and result["n_user_slots"] == 8 * 600
    assert result["reproduces_saved_X_test"] is True


def test_a_modified_x_test_is_detected(data_root, tmp_path) -> None:
    copy = tmp_path / "urban_copy"
    shutil.copytree(data_root / "urban", copy)
    x_test = np.load(copy / "X_test.npy")
    x_test[0, 0, 0] += 1.0
    np.save(copy / "X_test.npy", x_test)

    assert mc.measure_scenario(copy)["reproduces_saved_X_test"] is False


def test_per_run_share_matches_an_independent_computation(data_root) -> None:
    metadata = json.loads((data_root / "rural" / "metadata.json").read_text(encoding="utf-8"))
    result = mc.measure_scenario(data_root / "rural")
    run = result["runs"][0]
    series = simulate_time_series(
        SimulationConfig(), TimeSeriesConfig(**metadata["timeseries_config"]), scenario_name="rural",
        user_count=run["user_count"], acir_db=run["acir_db"], seed=run["seed"],
    )
    noise = compute_noise_power_dbm(20e6, 7.0)

    expected = float(np.mean((series.signal_dbm - noise) >= 15.0))

    assert run["share_snr_ge_threshold"] == pytest.approx(expected, abs=1e-12)
    assert series.noise_power_dbm == pytest.approx(noise)


def test_the_ceiling_does_not_depend_on_acir_or_interference() -> None:
    kwargs = dict(scenario_name="urban", user_count=4, seed=123)
    low = simulate_time_series(SimulationConfig(), TimeSeriesConfig(), acir_db=12.0, **kwargs)
    high = simulate_time_series(SimulationConfig(), TimeSeriesConfig(), acir_db=45.0, **kwargs)

    assert np.array_equal(low.signal_dbm, high.signal_dbm)
    assert not np.array_equal(low.interference_dbm, high.interference_dbm)  # interference differs, the ceiling cannot


def test_pass_edges_are_below_threshold_and_mid_pass_above(data_root) -> None:
    metadata = json.loads((data_root / "urban" / "metadata.json").read_text(encoding="utf-8"))
    run = next(r for r in metadata["runs"] if r["split"] == "test")
    series = simulate_time_series(SimulationConfig(), TimeSeriesConfig(), scenario_name="urban",
                                  user_count=run["user_count"], acir_db=run["acir_db"], seed=run["seed"])
    snr = series.signal_dbm - series.noise_power_dbm

    assert np.all(snr[:, 0] < 15.0) and np.all(snr[:, -1] < 15.0)  # edges of the 300 s pass
    assert np.all(snr[:, 300] > 15.0)  # mid-pass
    share = mc.measure_scenario(data_root / "urban")["pooled_share"]
    assert 0.75 < share < 0.90


def test_spread_statistics_are_consistent_with_the_run_list(data_root) -> None:
    r = mc.measure_scenario(data_root / "urban")
    shares = np.array([run["share_snr_ge_threshold"] for run in r["runs"]])

    assert r["over_runs"]["mean"] == pytest.approx(shares.mean())
    assert r["over_runs"]["std"] == pytest.approx(shares.std(ddof=1))  # sample std
    assert (r["over_runs"]["min"], r["over_runs"]["max"], r["over_runs"]["n"]) == (shares.min(), shares.max(), len(shares))
    pooled = sum(round(run["share_snr_ge_threshold"] * run["n_user_slots"]) for run in r["runs"]) / r["n_user_slots"]
    assert r["pooled_share"] == pytest.approx(pooled)
    assert r["difference_vs_hand_estimate_pp"] == pytest.approx(100 * (r["pooled_share"] - mc.HAND_ESTIMATE_SHARE))


def test_a_dataset_without_test_runs_is_rejected(data_root, tmp_path) -> None:
    copy = tmp_path / "no_test"
    shutil.copytree(data_root / "urban", copy)
    metadata = json.loads((copy / "metadata.json").read_text(encoding="utf-8"))
    for run in metadata["runs"]:
        run["split"] = "train"
    (copy / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="No test runs"):
        mc.measure_scenario(copy)


def test_main_writes_numbers_readme_and_run_info_and_refuses_to_overwrite(data_root, tmp_path, capsys) -> None:
    out = tmp_path / "ceiling"

    assert mc.main(["--data-root", str(data_root), "--out", str(out)]) == 0

    result = json.loads((out / "ceiling.json").read_text(encoding="utf-8"))
    assert set(result["scenarios"]) == {"urban", "rural"} and result["snr_threshold_db"] == 15.0
    assert result["noise_power_dbm"] == pytest.approx(compute_noise_power_dbm(20e6, 7.0))
    assert {"git_sha", "git_dirty", "git_dirty_files", "python", "numpy", "argv"} <= result["run_info"].keys()
    assert all(s["reproduces_saved_X_test"] for s in result["scenarios"].values())
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "hand estimate" in readme and "reproduces X_test" in readme and "{" not in readme and "}" not in readme
    assert "[urban] ceiling mean over 2 runs" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        mc.main(["--data-root", str(data_root), "--out", str(out)])
