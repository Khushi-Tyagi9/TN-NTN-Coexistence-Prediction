"""PRELIMINARY (exploratory) LSTM run, scored on the VALIDATION split only. Not the final protocol.

    python run_preliminary.py --scenario urban
    python run_preliminary.py --scenario rural
    python run_preliminary.py --scenario urban --variant instantaneous

Fixed config: hidden 96, 1 layer, lr 1e-3, seed 42, at most 40 epochs, patience 6. The test split is never
read (only X/y train and validation files are opened). After epoch 1 the script prints seconds per epoch and an
ETA, and stops without writing results if that ETA exceeds --eta-limit-minutes.
Output: results/phase3/preliminary/<scenario>[_<variant>]/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.phase3_data import load_train_val_arrays
from src.phase3_metrics import ar_prediction, fit_ar_baseline, persistence_prediction
from src.phase3_plots import plot_actual_vs_predicted, plot_training_curves
from src.phase3_preliminary import PRELIMINARY_CONFIG, train_validation_only
from src.reproducibility import collect_run_info

PHYSICAL_CORES = 12


class _EtaExceeded(Exception):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_info(data_dir: Path) -> dict[str, Any]:
    """Metadata plus hashes of the four files actually used (the test files are not touched)."""
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    used = ("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy")
    return {
        "directory": str(data_dir).replace("\\", "/"),
        "scenario": metadata["scenario"],
        "target_kind": metadata["target_kind"],
        "target": metadata["target"],
        "masked_features": metadata["masked_features"],
        "window_counts": {k: v for k, v in metadata["window_counts"].items() if k != "test"},
        "fast_fading_rho": metadata["fast_fading_rho"],
        "slot_duration_s": metadata["slot_duration_s"],
        "provenance": metadata["provenance"],
        "sha256": {name: _sha256(data_dir / name) for name in used},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, choices=["urban", "rural"])
    parser.add_argument("--variant", choices=["instantaneous"], default=None)
    parser.add_argument("--data-root", default="data/phase3")
    parser.add_argument("--out", default="results/phase3/preliminary")
    parser.add_argument("--threads", type=int, default=PHYSICAL_CORES)
    parser.add_argument("--eta-limit-minutes", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    name = args.scenario if args.variant is None else f"{args.scenario}_{args.variant}"
    data_dir = Path(args.data_root) / name
    out_dir = Path(args.out) / name
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"Refusing to overwrite {out_dir}.")
    if not (data_dir / "metadata.json").exists():
        sys.exit(f"Dataset {data_dir} not found; generate it with build_phase3_dataset.py first.")
    info = _dataset_info(data_dir)
    x_train, y_train, x_val, y_val = load_train_val_arrays(data_dir)
    config = PRELIMINARY_CONFIG

    epoch_seconds: list[float] = []
    clock = {"last": time.perf_counter()}

    def on_epoch(epoch: int, train_loss: float, val_loss: float) -> None:
        now = time.perf_counter()
        epoch_seconds.append(now - clock["last"])
        clock["last"] = now
        if epoch == 1:
            eta = epoch_seconds[0] * config.max_epochs
            print(f"[{name}] epoch 1: {epoch_seconds[0]:.1f} s/epoch. ETA if all {config.max_epochs} epochs run: "
                  f"{eta / 60:.1f} min (upper bound; early stopping may end sooner)", flush=True)
            if eta > args.eta_limit_minutes * 60:
                raise _EtaExceeded(f"ETA {eta / 60:.1f} min exceeds {args.eta_limit_minutes:g} min")
        elif epoch % 5 == 0:
            print(f"[{name}] epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f} ({epoch_seconds[-1]:.1f} s/epoch)", flush=True)

    try:
        result = train_validation_only(x_train, y_train, x_val, y_val, config, epoch_callback=on_epoch)
    except _EtaExceeded as stop:
        print(f"[{name}] STOPPED: {stop}. Nothing was written.", flush=True)
        return 3

    m = result["metrics"]
    summary = {
        "status": "PRELIMINARY / exploratory - NOT the final protocol; no gate is evaluated",
        "split_evaluated": "validation only; the test split was not read",
        "caveat": "validation is also used for early stopping and best-epoch selection, so these numbers are mildly optimistic",
        "scenario": args.scenario,
        "variant": args.variant,
        "config": asdict(config),
        "best_epoch": result["best_epoch"],
        "epochs_run": result["epochs_run"],
        "seconds_per_epoch": epoch_seconds,
        "wall_clock_seconds": result["seconds"],
        "validation": {
            "mae_db": m.mae_db,
            "rmse_db": m.rmse_db,
            "within_1db": m.within_1db,
            "within_3db": m.within_3db,
            "baselines_mae_db": {
                "persistence": m.persistence_mae_db,
                "moving_average_10": m.moving_average_mae_db,
                "linear_ar_10": m.ar_mae_db,
            },
            "best_baseline": m.best_baseline,
            "skill_vs_persistence": m.skill_vs_persistence,
            "skill_vs_best_baseline": m.skill_vs_best_baseline,
        },
        "history": result["history"],
        "dataset": info,
        "run_info": collect_run_info(config.seed, extra={"argv": argv if argv is not None else sys.argv[1:],
                                                         "torch_threads": args.threads}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_training_curves({config.seed: result["history"]}, {config.seed: result["best_epoch"]},
                         out_dir / "training_curves.png",
                         f"{name}: preliminary LSTM training curve (hidden 96, 1 layer, lr 1e-3, seed 42)")
    ar = fit_ar_baseline(x_train, y_train)
    plot_actual_vs_predicted(
        y_val,
        {"lstm": result["val_predictions"], "persistence": persistence_prediction(x_val), "ar10": ar_prediction(x_val, ar)},
        out_dir / "actual_vs_predicted_val.png",
        f"{name}: preliminary LSTM on the validation split ({info['target']}), 200-slot window",
        snippet_slots=200,
        split_name="validation",
    )

    b = summary["validation"]["baselines_mae_db"]
    print(f"[{name}] VALIDATION: MAE {m.mae_db:.3f} dB | RMSE {m.rmse_db:.3f} dB | within 1 dB {m.within_1db:.3f} | "
          f"within 3 dB {m.within_3db:.3f}")
    print(f"[{name}] baselines MAE (dB): persistence {b['persistence']:.3f} | 10-slot avg {b['moving_average_10']:.3f} | "
          f"AR(10) {b['linear_ar_10']:.3f} (best: {m.best_baseline})")
    print(f"[{name}] skill vs persistence {m.skill_vs_persistence:+.3f} | skill vs best baseline {m.skill_vs_best_baseline:+.3f} "
          f"| best epoch {result['best_epoch']}/{result['epochs_run']} | {result['seconds']:.0f} s total")
    print(f"[{name}] written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
