"""Stage B tooling: grid/selection rules, seed aggregation + gates, plots, and the driver end to end."""

import inspect
import json

import numpy as np
import pytest

import run_stage_b
from src.phase3_aggregate import aggregate_seed_runs
from src.phase3_metrics import DbMetrics
from src.phase3_search import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_MAX_EPOCHS,
    SEARCH_GRID,
    SEARCH_SCENARIO,
    grid_configs,
    run_trial,
    run_validation_search,
    select_best,
)


def _metrics(mae: float, *, persistence: float = 5.0, moving_average: float = 4.0, ar: float = 3.5) -> DbMetrics:
    best = min(persistence, moving_average, ar)
    return DbMetrics(
        mae_db=mae, rmse_db=mae * 1.3, within_1db=0.2, within_3db=0.6,
        persistence_mae_db=persistence, persistence_rmse_db=persistence * 1.3,
        skill_vs_persistence=1 - mae / persistence,
        moving_average_mae_db=moving_average, ar_mae_db=ar, best_baseline="linear_ar_10",
        best_baseline_mae_db=best, skill_vs_best_baseline=1 - mae / best,
    )


# ---------------------------------------------------------------------------- grid + selection
def test_amended_grid_is_five_one_layer_configs_with_the_fixed_budget() -> None:
    configs = grid_configs()

    assert len(configs) == 5 == len(SEARCH_GRID) and SEARCH_SCENARIO == "urban"
    assert [(c.hidden_size, c.learning_rate, c.num_layers) for c in configs] == [
        (96, 1e-3, 1),  # repo default first
        (64, 1e-3, 1), (64, 3e-4, 1), (128, 1e-3, 1), (128, 3e-4, 1),
    ]
    assert {c.seed for c in configs} == {42}
    assert {(c.batch_size, c.max_epochs, c.early_stopping_patience, c.dropout, c.weight_decay) for c in configs} == {
        (64, 40, 6, 0.0, 0.0)
    }
    assert (DEFAULT_MAX_EPOCHS, DEFAULT_EARLY_STOPPING_PATIENCE) == (40, 6)


def _trial(val_loss: float, hidden: int = 96, layers: int = 1, lr: float = 1e-3, val_mae: float = 9.9) -> dict:
    return {"best_val_loss_normalized_mse": val_loss, "val_mae_db": val_mae,
            "config": {"hidden_size": hidden, "num_layers": layers, "learning_rate": lr}}


def test_selection_is_lowest_validation_loss_with_documented_tie_breaks() -> None:
    assert select_best([_trial(0.30), _trial(0.25), _trial(0.27)]) == 1
    # validation LOSS decides, not validation MAE
    assert select_best([_trial(0.30, val_mae=1.0), _trial(0.20, val_mae=5.0)]) == 1
    # exact ties: smaller hidden size first, then fewer layers, then larger learning rate
    assert select_best([_trial(0.2, hidden=128), _trial(0.2, hidden=64)]) == 1
    assert select_best([_trial(0.2, layers=2), _trial(0.2, layers=1)]) == 1
    assert select_best([_trial(0.2, lr=3e-4), _trial(0.2, lr=1e-3)]) == 1
    with pytest.raises(ValueError):
        select_best([])


def test_search_functions_cannot_receive_the_test_split() -> None:
    for function in (run_trial, run_validation_search):
        names = set(inspect.signature(function).parameters)
        assert not any("test" in name for name in names), function.__name__


# ---------------------------------------------------------------------------------- aggregation
def test_aggregate_reports_every_seed_mean_std_and_baselines() -> None:
    out = aggregate_seed_runs([42, 43, 44], [_metrics(3.0), _metrics(3.2), _metrics(3.4)])

    assert [row["seed"] for row in out["per_seed"]] == [42, 43, 44]
    assert out["mae_db"]["mean"] == pytest.approx(3.2)
    assert out["mae_db"]["std"] == pytest.approx(0.2)  # sample std (ddof=1)
    assert out["baselines_mae_db"] == {"persistence": 5.0, "moving_average_10": 4.0, "linear_ar_10": 3.5}
    assert out["best_baseline"] == "linear_ar_10"
    assert out["gate_basis"] == "mean test MAE over seeds [42, 43, 44]"
    json.dumps(out)


def test_gates_are_evaluated_on_the_mean_mae_not_the_best_seed() -> None:
    # persistence 5.0 -> 15% gate needs mean MAE <= 4.25. Best seed (4.0) passes, the mean (4.3) does not.
    out = aggregate_seed_runs([42, 43, 44], [_metrics(4.0), _metrics(4.3), _metrics(4.6)])

    assert out["per_seed"][0]["skill_vs_persistence"] >= 0.15
    assert out["skill_vs_persistence_on_mean_mae"] == pytest.approx(1 - 4.3 / 5.0)
    assert out["meets_prototype_target"] is False
    assert out["transformer_required"] is True


def test_both_gates_pass_fail_independently() -> None:
    both = aggregate_seed_runs([1], [_metrics(3.0)])  # 40% vs persistence, 14.3% vs AR (3.5)
    assert both["meets_prototype_target"] and both["meets_final_target"] and not both["transformer_required"]

    only_prototype = aggregate_seed_runs([1], [_metrics(3.4)])  # 32% vs persistence, 2.9% vs AR
    assert only_prototype["meets_prototype_target"] is True
    assert only_prototype["meets_final_target"] is False
    assert only_prototype["skill_vs_best_baseline_on_mean_mae"] == pytest.approx(1 - 3.4 / 3.5)


def test_final_gate_boundary_at_exactly_five_percent_on_the_mean() -> None:
    at = aggregate_seed_runs([1, 2], [_metrics(3.325), _metrics(3.325)])  # 3.5 * 0.95
    below = aggregate_seed_runs([1, 2], [_metrics(3.326), _metrics(3.326)])
    assert at["skill_vs_best_baseline_on_mean_mae"] == pytest.approx(0.05)
    assert below["meets_final_target"] is False


def test_aggregate_rejects_inconsistent_inputs() -> None:
    with pytest.raises(ValueError, match="differs between seeds"):
        aggregate_seed_runs([1, 2], [_metrics(3.0), _metrics(3.0, persistence=5.1)])
    with pytest.raises(ValueError, match="unique"):
        aggregate_seed_runs([1, 1], [_metrics(3.0), _metrics(3.0)])
    with pytest.raises(ValueError):
        aggregate_seed_runs([1, 2], [_metrics(3.0)])
    with pytest.raises(ValueError, match="all baselines"):
        no_ar = DbMetrics(mae_db=1.0, rmse_db=1.0, within_1db=0.5, within_3db=0.5, persistence_mae_db=2.0,
                          persistence_rmse_db=2.0, skill_vs_persistence=0.5)
        aggregate_seed_runs([1], [no_ar])


def test_single_seed_aggregate_has_zero_std() -> None:
    assert aggregate_seed_runs([42], [_metrics(3.0)])["mae_db"]["std"] == 0.0


# ------------------------------------------------------------------------------------------ plots
def test_plots_are_written(tmp_path) -> None:
    from src.phase3_plots import plot_actual_vs_predicted, plot_training_curves

    rng = np.random.default_rng(0)
    truth = rng.normal(-80, 6, 400)
    preds = {"lstm": truth + rng.normal(0, 1, 400), "persistence": truth + rng.normal(0, 4, 400),
             "ar10": truth + rng.normal(0, 2, 400)}
    history = {"train_loss": [1.0, 0.8, 0.7], "val_loss": [1.1, 0.9, 0.85]}

    curves = plot_training_curves({42: history, 43: history}, {42: 3, 43: 2}, tmp_path / "c" / "curves.png", "t")
    scatter = plot_actual_vs_predicted(truth, preds, tmp_path / "avp.png", "t", snippet_slots=100)

    for path in (curves, scatter):
        assert path.exists() and path.stat().st_size > 5_000


# ---------------------------------------------------------------------------------- driver guards
def test_driver_refuses_to_overwrite_existing_results(tmp_path) -> None:
    run_stage_b._require_clean_output(tmp_path / "missing")  # no error
    existing = tmp_path / "final"
    existing.mkdir()
    run_stage_b._require_clean_output(existing)  # empty dir is fine
    (existing / "x.json").write_text("{}")
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        run_stage_b._require_clean_output(existing)


def test_driver_parser_and_variant_rules() -> None:
    parser = run_stage_b.build_parser()
    args = parser.parse_args(["supplementary", "--scenario", "rural", "--variant", "nosinr"])
    assert (args.phase, args.scenario, args.variant, args.threads) == ("supplementary", "rural", "nosinr", 8)
    assert (args.resume, args.accept_eta, args.eta_limit_hours) == (False, False, 2.0)
    with pytest.raises(SystemExit):
        parser.parse_args(["search", "--scenario", "suburban"])
    with pytest.raises(SystemExit, match="only valid"):
        run_stage_b.main(["search", "--scenario", "urban", "--variant", "nosinr"])
    assert run_stage_b.FINAL_SEEDS == (42, 43, 44) and run_stage_b.PRIMARY_SEED == 42
    assert run_stage_b.FOLLOW_UP_RUNS == 6  # final trainings: 3 seeds x 2 scenarios


def test_search_is_urban_only() -> None:
    with pytest.raises(SystemExit, match="urban only"):
        run_stage_b.main(["search", "--scenario", "rural"])


def test_amendment_record_states_the_time_limit_reduction() -> None:
    amendment = run_stage_b._amendment()

    assert amendment["search_reduced_due_to_time_limit"] is True
    assert "time limit" in amendment["note"] and "urban" in amendment["note"]
    assert amendment["training_budget"] == {"max_epochs": 40, "early_stopping_patience": 6, "batch_size": 64}
    assert len(amendment["grid"]) == 5 and "12 configs" in amendment["original_grid_not_run"]


def test_dataset_kind_guard_rejects_a_mismatched_dataset() -> None:
    info = {"directory": "data/phase3/urban", "target_kind": "instantaneous", "masked_features": []}
    run_stage_b._check_dataset_kind(info, target_kind="instantaneous", masked=[])  # matches: no error
    with pytest.raises(SystemExit, match="expected target=local_mean"):
        run_stage_b._check_dataset_kind(info, target_kind="local_mean", masked=[])
    with pytest.raises(SystemExit, match="masked"):
        run_stage_b._check_dataset_kind(info, target_kind="instantaneous", masked=["sinr_db"])


# ------------------------------------------------------------ real search -> final -> supplementary
@pytest.fixture()
def tiny_data_root(tmp_path):
    from src.config import SimulationConfig
    from src.phase3_dataset_builder import DatasetBuildConfig, build_phase3_dataset, save_dataset
    from src.timeseries import TimeSeriesConfig

    root = tmp_path / "data"
    for scenario, target, folder in (
        ("urban", "local_mean", "urban"),
        ("urban", "instantaneous", "urban_instantaneous"),
        ("rural", "local_mean", "rural"),
    ):
        split, metadata = build_phase3_dataset(
            SimulationConfig(), TimeSeriesConfig(n_slots=60),
            DatasetBuildConfig(scenario=scenario, user_counts=(4,), acir_points_db=(20.0, 30.0), runs_per_point=3,
                               target=target),
        )
        save_dataset(split, metadata, root / folder)
    return root


@pytest.fixture()
def tiny_grid(monkeypatch):
    """Two tiny configs instead of the real 5, so the driver can be run end to end in seconds."""
    from src.phase3_prediction import LSTMTrainingConfig

    monkeypatch.setattr(
        run_stage_b, "grid_configs",
        lambda **kw: [LSTMTrainingConfig(hidden_size=h, max_epochs=2, device="cpu", seed=kw["seed"]) for h in (64, 96)],
    )


def _common(tiny_data_root, results, scenario: str) -> list[str]:
    return ["--scenario", scenario, "--data-root", str(tiny_data_root), "--results-root", str(results), "--threads", "2"]


def test_search_final_supplementary_end_to_end(tiny_data_root, tmp_path, tiny_grid) -> None:
    pytest.importorskip("torch", reason="PyTorch not installed: Stage B driver is NOT exercised")
    results = tmp_path / "results"
    urban, rural = _common(tiny_data_root, results, "urban"), _common(tiny_data_root, results, "rural")

    assert run_stage_b.main(["search", *urban]) == 0
    trials = json.loads((results / "urban" / "search" / "trials.json").read_text(encoding="utf-8"))
    assert trials["n_trials"] == 2 and trials["search_seed"] == 42
    assert trials["selection_criterion"].startswith("validation loss")
    assert trials["protocol_amendment"]["search_reduced_due_to_time_limit"] is True
    assert (results / "urban" / "search" / "trials.jsonl").read_text().count("\n") == 2
    assert all(not any("test" in key for key in trial) for trial in trials["trials"])  # no test info in the log
    winner = min(trials["trials"], key=lambda t: (t["best_val_loss_normalized_mse"], t["config"]["hidden_size"]))
    assert trials["selected_config"] == winner["config"]
    with pytest.raises(SystemExit, match="Refusing to overwrite"):  # a finished search is closed, even with --resume
        run_stage_b.main(["search", "--resume", *urban])

    assert run_stage_b.main(["final", *urban]) == 0
    summary = json.loads((results / "urban" / "summary.json").read_text(encoding="utf-8"))
    aggregate = summary["aggregate"]
    assert [row["seed"] for row in aggregate["per_seed"]] == [42, 43, 44]
    assert aggregate["gate_basis"] == "mean test MAE over seeds [42, 43, 44]"
    assert summary["selection"]["search_reduced_due_to_time_limit"] is True
    assert summary["selection"]["training_budget"]["max_epochs"] == 40
    assert summary["selection"]["config_source_scenario"] == "urban"
    assert summary["dataset"]["sha256"]["X_test.npy"] and summary["run_info"]["torch_threads"] == 2
    for seed in (42, 43, 44):
        seed_dir = results / "urban" / "final" / f"seed_{seed}"
        seed_summary = json.loads((seed_dir / "summary.json").read_text(encoding="utf-8"))
        assert seed_summary["config"]["seed"] == seed and seed_summary["history"]["val_loss"]
        assert (seed_dir / "lstm_best.pt").exists() and (seed_dir / "predictions_test.npz").exists()
    assert (results / "urban" / "training_curves.png").exists() and (results / "urban" / "actual_vs_predicted.png").exists()
    with pytest.raises(SystemExit, match="Refusing to overwrite"):  # grid is closed
        run_stage_b.main(["final", *urban])

    # rural: no search of its own, the urban-selected config is reused unchanged (only the seed varies)
    assert not (results / "rural" / "search").exists()
    assert run_stage_b.main(["final", *rural]) == 0
    rural_summary = json.loads((results / "rural" / "summary.json").read_text(encoding="utf-8"))
    assert rural_summary["selection"]["config_source_scenario"] == "urban"
    assert rural_summary["config"] == summary["config"] == trials["selected_config"]
    assert rural_summary["dataset"]["scenario"] == "rural"

    assert run_stage_b.main(["supplementary", "--variant", "instantaneous", *urban]) == 0
    sup = json.loads(
        (results / "urban" / "supplementary" / "instantaneous" / "summary_aggregate.json").read_text(encoding="utf-8")
    )
    assert sup["aggregate"]["n_seeds"] == 1 and "single seed 42" in sup["aggregate"]["gate_basis"]
    assert sup["selection"]["search_reduced_due_to_time_limit"] is True
    with pytest.raises(SystemExit, match="not found"):  # no nosinr dataset was generated in this fixture
        run_stage_b.main(["supplementary", "--variant", "nosinr", *urban])


def test_eta_over_the_limit_pauses_after_the_first_trial_and_resumes_identically(
    tiny_data_root, tmp_path, tiny_grid, capsys
) -> None:
    pytest.importorskip("torch", reason="PyTorch not installed: Stage B driver is NOT exercised")
    results = tmp_path / "results"
    urban = _common(tiny_data_root, results, "urban")
    search_dir = results / "urban" / "search"

    # A negative limit: ANY ETA exceeds it, even 0.0 s. The driver rounds a trial's time to 0.1 s, so a very fast
    # warm trial (< 0.05 s) gives ETA = 0 and would not exceed a limit of 0 h (timing-dependent, hence -1).
    assert run_stage_b.main(["search", "--eta-limit-hours", "-1", *urban]) == 3
    out = capsys.readouterr().out
    assert "[eta] first trial took" in out and "wall-clock" in out and "ETA" in out
    assert (search_dir / "ETA_EXCEEDED.txt").exists() and not (search_dir / "trials.json").exists()
    first_row = (search_dir / "trials.jsonl").read_text().splitlines()
    assert len(first_row) == 1

    with pytest.raises(SystemExit, match="Refusing to overwrite"):  # no silent restart without --resume
        run_stage_b.main(["search", *urban])

    assert run_stage_b.main(["search", "--resume", "--accept-eta", *urban]) == 0
    trials = json.loads((search_dir / "trials.json").read_text(encoding="utf-8"))
    assert trials["n_trials"] == 2 and not (search_dir / "ETA_EXCEEDED.txt").exists()
    kept = json.loads(first_row[0])
    assert trials["trials"][0]["best_val_loss_normalized_mse"] == kept["best_val_loss_normalized_mse"]  # trial 1 not redone


def test_eta_under_the_limit_does_not_pause(tiny_data_root, tmp_path, tiny_grid) -> None:
    pytest.importorskip("torch", reason="PyTorch not installed: Stage B driver is NOT exercised")
    results = tmp_path / "results"
    assert run_stage_b.main(["search", *_common(tiny_data_root, results, "urban")]) == 0
    assert not (results / "urban" / "search" / "ETA_EXCEEDED.txt").exists()


def test_resume_refuses_a_log_that_does_not_match_the_grid(tiny_data_root, tmp_path, tiny_grid) -> None:
    pytest.importorskip("torch", reason="PyTorch not installed: Stage B driver is NOT exercised")
    results = tmp_path / "results"
    search_dir = results / "urban" / "search"
    search_dir.mkdir(parents=True)
    (search_dir / "trials.jsonl").write_text(json.dumps({"index": 0, "config": {"hidden_size": 999}}) + "\n")

    with pytest.raises(SystemExit, match="does not match the declared grid"):
        run_stage_b.main(["search", "--resume", *_common(tiny_data_root, results, "urban")])


def test_final_requires_the_urban_search_first(tiny_data_root, tmp_path) -> None:
    pytest.importorskip("torch", reason="PyTorch not installed: Stage B driver is NOT exercised")
    with pytest.raises(SystemExit, match="validation search first"):
        run_stage_b.main(["final", *_common(tiny_data_root, tmp_path / "r", "rural")])
