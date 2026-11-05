"""Stage B driver: validation-only search, 3-seed final training, supplementary variants.

    python run_stage_b.py search        --scenario urban      # 5 configs, urban only, seed 42, validation only
    python run_stage_b.py final         --scenario urban      # urban-selected config x seeds 42,43,44, test once each
    python run_stage_b.py final         --scenario rural      # the SAME urban-selected config
    python run_stage_b.py supplementary --scenario urban --variant instantaneous   # chosen config, seed 42

Everything is written under results/phase3/<scenario>/ with run_info. The three phases refuse to overwrite
existing output: once the test split has been evaluated for a config/seed, the grid is closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.phase3_aggregate import aggregate_seed_runs
from src.phase3_data import load_phase3_dataset
from src.phase3_metrics import ar_prediction, fit_ar_baseline, linear_moving_average_prediction, persistence_prediction
from src.phase3_pipeline import Phase3RunResult, to_serializable_summary
from src.phase3_plots import plot_actual_vs_predicted, plot_training_curves
from src.phase3_prediction import LSTMTrainingConfig, PredictionStageResult, train_lstm_forecaster
from src.phase3_search import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_MAX_EPOCHS,
    ORIGINAL_GRID_NOT_RUN,
    REDUCTION_NOTE,
    SEARCH_GRID,
    SEARCH_SCENARIO,
    SEARCH_SELECTION_CRITERION,
    grid_configs,
    run_validation_search,
    select_best,
)
from src.reproducibility import collect_run_info

PRIMARY_SEED = 42
FINAL_SEEDS = (42, 43, 44)
VARIANTS = {"instantaneous": "instantaneous", "nosinr": "nosinr"}
DEFAULT_THREADS = 8  # torch CPU threads for the single running process (recorded in run_info)
FOLLOW_UP_RUNS = len(FINAL_SEEDS) * 2  # final trainings: 3 seeds x {urban, rural}


class _EtaExceeded(Exception):
    """Raised after the first trial when the projected remaining time exceeds the limit."""


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_info(data_dir: Path) -> dict[str, Any]:
    if not (data_dir / "metadata.json").exists():
        sys.exit(f"Dataset {data_dir} not found; generate it with build_phase3_dataset.py first.")
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    return {
        "directory": str(data_dir).replace("\\", "/"),
        "scenario": metadata["scenario"],
        "target_kind": metadata["target_kind"],
        "target": metadata["target"],
        "target_definition": metadata["target_definition"],
        "masked_features": metadata["masked_features"],
        "window_counts": metadata["window_counts"],
        "fast_fading_rho": metadata["fast_fading_rho"],
        "slot_duration_s": metadata["slot_duration_s"],
        "provenance": metadata["provenance"],
        "sha256": {name: _sha256(data_dir / name) for name in sorted(p.name for p in data_dir.glob("*.npy"))},
    }


def _run_info(seed: int, args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    return collect_run_info(
        seed, extra={"argv": sys.argv[1:], "torch_threads": args.threads, "phase": args.phase, **extra}
    )


def _require_clean_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        sys.exit(f"Refusing to overwrite {path}: results already exist and the grid is closed.")


def _setup(args: argparse.Namespace) -> tuple[Path, Path]:
    import torch

    torch.set_num_threads(args.threads)
    data_dir = Path(args.data_root) / (args.scenario if args.variant is None else f"{args.scenario}_{VARIANTS[args.variant]}")
    out_dir = Path(args.results_root) / args.scenario
    return data_dir, out_dir


def _amendment() -> dict[str, Any]:
    return {
        "search_reduced_due_to_time_limit": True,
        "note": REDUCTION_NOTE,
        "original_grid_not_run": ORIGINAL_GRID_NOT_RUN,
        "search_scenario": SEARCH_SCENARIO,
        "training_budget": {"max_epochs": DEFAULT_MAX_EPOCHS, "early_stopping_patience": DEFAULT_EARLY_STOPPING_PATIENCE,
                            "batch_size": 64},
        "grid": [{"hidden_size": h, "learning_rate": lr, "num_layers": n} for h, lr, n in SEARCH_GRID],
    }


def _load_selected_config(results_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Config chosen by the urban search (used for every scenario) and its selection record."""
    path = results_root / SEARCH_SCENARIO / "search" / "trials.json"
    if not path.exists():
        sys.exit(f"Run the validation search first (no {path.as_posix()}).")
    search = json.loads(path.read_text(encoding="utf-8"))
    selection = {
        "criterion": search["selection_criterion"],
        "trials_file": f"{SEARCH_SCENARIO}/search/trials.json",
        "config_source_scenario": SEARCH_SCENARIO,
        "search_seed": search["search_seed"],
        "n_trials": search["n_trials"],
        "selected_index": search["selected_index"],
        **search["protocol_amendment"],
    }
    return search["selected_config"], selection


def _check_dataset_kind(info: dict[str, Any], *, target_kind: str, masked: list[str]) -> None:
    if info["target_kind"] != target_kind or info["masked_features"] != masked:
        sys.exit(f"Dataset {info['directory']} is target={info['target_kind']} masked={info['masked_features']}; "
                 f"expected target={target_kind} masked={masked}.")


# ------------------------------------------------------------------------------------------ search
def _fmt_duration(seconds: float) -> str:
    return f"{seconds / 3600:.2f} h ({seconds / 60:.0f} min)"


def phase_search(args: argparse.Namespace) -> int:
    if args.scenario != SEARCH_SCENARIO:
        sys.exit(f"The search runs on {SEARCH_SCENARIO} only (reduced protocol); rural reuses the {SEARCH_SCENARIO}-selected config.")
    data_dir, out_dir = _setup(args)
    search_dir = out_dir / "search"
    log_path = search_dir / "trials.jsonl"
    if (search_dir / "trials.json").exists():
        sys.exit(f"Refusing to overwrite {search_dir}: the search is complete and the grid is closed.")
    done: list[dict[str, Any]] = []
    if args.resume:
        if log_path.exists():
            done = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        _require_clean_output(search_dir)
    info = _dataset_info(data_dir)
    _check_dataset_kind(info, target_kind="local_mean", masked=[])
    split = load_phase3_dataset(data_dir, require_pre_split=True)

    configs = grid_configs(seed=PRIMARY_SEED, device="cpu")
    if [row["config"] for row in done] != [asdict(c) for c in configs[: len(done)]]:
        sys.exit("Existing trials.jsonl does not match the declared grid order; cannot resume.")
    search_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = [{k: v for k, v in row.items() if k != "index"} for row in done]
    offset = len(done)

    def on_trial(local_index: int, trial: dict[str, Any]) -> None:
        index = offset + local_index
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"index": index, **trial}) + "\n")
        c = trial["config"]
        print(f"[{args.scenario}] trial {index + 1}/{len(configs)} hidden={c['hidden_size']} layers={c['num_layers']} "
              f"lr={c['learning_rate']:g}: val loss {trial['best_val_loss_normalized_mse']:.5f} "
              f"(val MAE {trial['val_mae_db']:.4f} dB), best epoch {trial['best_epoch']}/{trial['epochs_run']}, "
              f"wall-clock {trial['seconds']:.0f}s", flush=True)
        if index == 0:
            remaining_runs = (len(configs) - 1) + FOLLOW_UP_RUNS
            eta = remaining_runs * trial["seconds"]
            print(f"[eta] first trial took {trial['seconds']:.0f}s. Crude ETA for the {remaining_runs} remaining runs "
                  f"(search {len(configs) - 1} + {len(FINAL_SEEDS)} seeds x 2 scenarios, "
                  f"assuming each run costs about as much as this one): {_fmt_duration(eta)}", flush=True)
            if eta > args.eta_limit_hours * 3600 and not args.accept_eta:
                raise _EtaExceeded(f"ETA {_fmt_duration(eta)} exceeds {args.eta_limit_hours:g} h")

    try:
        new_trials = run_validation_search(
            split.X_train, split.y_train, split.X_val, split.y_val, configs[offset:], on_trial=on_trial
        )
    except _EtaExceeded as stop:
        (search_dir / "ETA_EXCEEDED.txt").write_text(f"{stop}\nRe-run with --resume --accept-eta to continue.\n", encoding="utf-8")
        print(f"[eta] STOPPING after the first trial: {stop}. Re-run with --resume --accept-eta to continue.", flush=True)
        return 3
    # Only train/validation arrays were passed: the test split cannot influence selection.
    trials += new_trials
    selected = select_best(trials)
    _write_json(
        search_dir / "trials.json",
        {
            "scenario": args.scenario,
            "selection_criterion": SEARCH_SELECTION_CRITERION,
            "search_seed": PRIMARY_SEED,
            "n_trials": len(trials),
            "selected_index": selected,
            "selected_config": trials[selected]["config"],
            "protocol_amendment": _amendment(),
            "trials": trials,
            "dataset": info,
            "run_info": _run_info(PRIMARY_SEED, args),
        },
    )
    (search_dir / "ETA_EXCEEDED.txt").unlink(missing_ok=True)
    print(f"[{args.scenario}] selected trial {selected + 1}: {trials[selected]['config']}")
    return 0


# ------------------------------------------------------------------------------------------- final
def _train_one(split, config: LSTMTrainingConfig, seed_dir: Path, info: dict[str, Any], args: argparse.Namespace,
               note: str | None = None):
    result = train_lstm_forecaster(split, config=config, checkpoint_path=seed_dir / "lstm_best.pt")
    stage = PredictionStageResult(lstm=result, transformer=None, transformer_executed=False)
    payload = to_serializable_summary(
        Phase3RunResult(prediction=stage, mitigation=None, mitigation_skipped_reason="Stage B: prediction only")
    )
    payload["config"] = asdict(config)
    payload["history"] = result.history
    payload["dataset"] = info
    payload["run_info"] = _run_info(config.seed, args, note=note)
    _write_json(seed_dir / "summary.json", payload)

    ar = fit_ar_baseline(split.X_train, split.y_train)
    np.savez_compressed(
        seed_dir / "predictions_test.npz",
        y_true=split.y_test,
        lstm=result.test_predictions,
        persistence=persistence_prediction(split.X_test),
        moving_average_10=linear_moving_average_prediction(split.X_test),
        linear_ar_10=ar_prediction(split.X_test, ar),
    )
    return result


def _baseline_predictions(split, lstm_pred: np.ndarray) -> dict[str, np.ndarray]:
    ar = fit_ar_baseline(split.X_train, split.y_train)
    return {
        "lstm": lstm_pred,
        "persistence": persistence_prediction(split.X_test),
        "ar10": ar_prediction(split.X_test, ar),
    }


def phase_final(args: argparse.Namespace) -> int:
    data_dir, out_dir = _setup(args)
    final_dir = out_dir / "final"
    _require_clean_output(final_dir)
    base, selection = _load_selected_config(Path(args.results_root))
    info = _dataset_info(data_dir)
    _check_dataset_kind(info, target_kind="local_mean", masked=[])
    split = load_phase3_dataset(data_dir, require_pre_split=True)

    results = {}
    for seed in FINAL_SEEDS:
        config = LSTMTrainingConfig(**{**base, "seed": seed})
        results[seed] = _train_one(split, config, final_dir / f"seed_{seed}", info, args)
        r = results[seed]
        print(f"[{args.scenario}] seed {seed}: test MAE {r.test_db_metrics.mae_db:.4f} dB "
              f"(best epoch {r.best_epoch}, {len(r.history['val_loss'])} epochs)", flush=True)

    aggregate = aggregate_seed_runs(list(FINAL_SEEDS), [results[s].test_db_metrics for s in FINAL_SEEDS])
    _write_json(
        out_dir / "summary.json",
        {
            "scenario": args.scenario,
            "stage": "B - LSTM, local-mean target, primary result",
            "config": base,
            "selection": selection,
            "aggregate": aggregate,
            "legacy_within_10pct_accuracy_informational": {
                str(s): results[s].test_metrics.within_10pct_accuracy for s in FINAL_SEEDS
            },
            "dataset": info,
            "run_info": _run_info(PRIMARY_SEED, args, seeds=list(FINAL_SEEDS)),
        },
    )
    plot_training_curves(
        {s: results[s].history for s in FINAL_SEEDS},
        {s: results[s].best_epoch for s in FINAL_SEEDS},
        out_dir / "training_curves.png",
        f"{args.scenario}: LSTM training curves (hidden {base['hidden_size']}, layers {base['num_layers']}, lr {base['learning_rate']:g})",
    )
    plot_actual_vs_predicted(
        split.y_test,
        _baseline_predictions(split, results[PRIMARY_SEED].test_predictions),
        out_dir / "actual_vs_predicted.png",
        f"{args.scenario}: local-mean interference, 1 slot ahead (test split, seed {PRIMARY_SEED})",
    )
    _print_aggregate(args.scenario, aggregate)
    return 0


def _print_aggregate(label: str, aggregate: dict[str, Any]) -> None:
    mae = aggregate["mae_db"]
    base = aggregate["baselines_mae_db"]
    print(f"[{label}] MEAN test MAE {mae['mean']:.4f} dB (std {mae['std']:.4f}, n={aggregate['n_seeds']}) | "
          f"persistence {base['persistence']:.4f} | MA10 {base['moving_average_10']:.4f} | AR10 {base['linear_ar_10']:.4f}")
    print(f"[{label}] prototype gate: skill {aggregate['skill_vs_persistence_on_mean_mae']:+.4f} "
          f"(>= {aggregate['min_skill_required']}) -> {aggregate['meets_prototype_target']}")
    print(f"[{label}] final gate: skill {aggregate['skill_vs_best_baseline_on_mean_mae']:+.4f} vs {aggregate['best_baseline']} "
          f"(>= {aggregate['min_final_skill_required']}) -> {aggregate['meets_final_target']}")
    if aggregate["transformer_required"]:
        print(f"[{label}] PROTOTYPE GATE FAILED: the plan calls for the Transformer; not run automatically.")


# ---------------------------------------------------------------------------------- supplementary
def phase_supplementary(args: argparse.Namespace) -> int:
    if args.variant is None:
        sys.exit("--variant is required for the supplementary phase.")
    data_dir, out_dir = _setup(args)
    variant_dir = out_dir / "supplementary" / args.variant
    _require_clean_output(variant_dir)
    base, selection = _load_selected_config(Path(args.results_root))
    info = _dataset_info(data_dir)
    if args.variant == "instantaneous":
        _check_dataset_kind(info, target_kind="instantaneous", masked=[])
    else:
        _check_dataset_kind(info, target_kind="local_mean", masked=["sinr_db"])
    split = load_phase3_dataset(data_dir, require_pre_split=True)

    config = LSTMTrainingConfig(**{**base, "seed": PRIMARY_SEED})
    note = "supplementary: chosen config, single seed 42, no separate search; gates are informational"
    result = _train_one(split, config, variant_dir, info, args, note=note)
    aggregate = aggregate_seed_runs([PRIMARY_SEED], [result.test_db_metrics])
    aggregate["gate_basis"] = f"single seed {PRIMARY_SEED} (supplementary run; gates informational)"
    _write_json(
        variant_dir / "summary_aggregate.json",
        {"scenario": args.scenario, "variant": args.variant, "note": note, "config": base, "selection": selection,
         "aggregate": aggregate,
         "dataset": info, "run_info": _run_info(PRIMARY_SEED, args)},
    )
    plot_training_curves(
        {PRIMARY_SEED: result.history}, {PRIMARY_SEED: result.best_epoch}, variant_dir / "training_curves.png",
        f"{args.scenario} / {args.variant}: LSTM training curves (seed {PRIMARY_SEED})",
    )
    plot_actual_vs_predicted(
        split.y_test, _baseline_predictions(split, result.test_predictions), variant_dir / "actual_vs_predicted.png",
        f"{args.scenario} / {args.variant}: {info['target']} (test split, seed {PRIMARY_SEED})",
    )
    _print_aggregate(f"{args.scenario}/{args.variant}", aggregate)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["search", "final", "supplementary"])
    parser.add_argument("--scenario", required=True, choices=["urban", "rural"])
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=None, help="supplementary phase only")
    parser.add_argument("--data-root", default="data/phase3")
    parser.add_argument("--results-root", default="results/phase3")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="torch CPU threads (recorded in run_info)")
    parser.add_argument("--resume", action="store_true", help="search only: continue an interrupted/paused search")
    parser.add_argument("--eta-limit-hours", type=float, default=2.0,
                        help="search only: stop after the first trial if the projected remaining time exceeds this")
    parser.add_argument("--accept-eta", action="store_true", help="search only: continue despite an ETA above the limit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase != "supplementary" and args.variant is not None:
        sys.exit("--variant is only valid for the supplementary phase.")
    return {"search": phase_search, "final": phase_final, "supplementary": phase_supplementary}[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
