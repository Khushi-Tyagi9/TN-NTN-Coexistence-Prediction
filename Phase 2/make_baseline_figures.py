"""Three small descriptive figures (no model training) -> results/figures/.

1. one urban run, one user: interference, signal and SINR over time;
2. the same run: faded interference with the local-mean target over it;
3. VALIDATION-split MAE of the three baselines (repeat last value, 10-slot average, AR(10)) for urban and rural.

    python make_baseline_figures.py [--data-root data/phase3] [--out results/figures]

The run for figures 1-2 is the first urban run of the dataset's metadata.json with ACIR 30 dB (same
user count, seed and ACIR the dataset builder used); it is re-simulated here, not trained on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import SimulationConfig  # noqa: E402
from src.phase3_data import load_train_val_arrays  # noqa: E402
from src.phase3_metrics import (  # noqa: E402
    ar_prediction,
    fit_ar_baseline,
    linear_moving_average_prediction,
    mae_db,
    persistence_prediction,
)
from src.phase3_plots import AQUA, BLUE, INK, INK_SECONDARY, ORANGE, SURFACE, _style  # noqa: E402
from src.reproducibility import collect_run_info  # noqa: E402
from src.timeseries import TimeSeriesConfig, simulate_time_series  # noqa: E402

DPI = 200
YELLOW = "#eda100"
FIGURE_ACIR_DB = 30.0
BASELINE_LABELS = (
    ("persistence", "repeat last value"),
    ("moving_average_10", "10-slot average (linear power)"),
    ("linear_ar_10", "AR(10) (least squares, fitted on train)"),
)
BASELINE_COLORS = {"persistence": ORANGE, "moving_average_10": YELLOW, "linear_ar_10": AQUA}
BASELINE_HATCHES = {"persistence": "", "moving_average_10": "///", "linear_ar_10": "..."}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def pick_run(metadata: dict[str, Any]) -> dict[str, Any]:
    """First run in the dataset metadata at the figure ACIR."""
    for run in metadata["runs"]:
        if run["acir_db"] == FIGURE_ACIR_DB:
            return run
    raise ValueError(f"No run with ACIR {FIGURE_ACIR_DB} dB in the dataset metadata.")


def simulate_figure_run(metadata: dict[str, Any]):
    run = pick_run(metadata)
    series = simulate_time_series(
        SimulationConfig(),
        TimeSeriesConfig(n_slots=metadata["n_slots_per_run"]),
        scenario_name=metadata["scenario"],
        user_count=run["user_count"],
        acir_db=run["acir_db"],
        seed=run["seed"],
    )
    return run, series


def figure_time_series(series, run: dict[str, Any], path: Path, user: int = 0) -> Path:
    t = np.arange(series.interference_dbm.shape[1]) * series.slot_duration_s
    panels = (
        (series.interference_dbm[user], "interference (dBm)", ORANGE, "shadowing (slow) + fast fading (jagged)"),
        (series.signal_dbm[user], "satellite signal (dBm)", BLUE, "satellite pass: smooth rise and fall"),
        (series.sinr_db[user], "SINR (dB)", AQUA, "SINR = signal - (interference + noise)"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, (values, label, color, note) in zip(axes, panels):
        _style(ax)
        ax.plot(t, values, color=color, linewidth=1.2)
        ax.set_ylabel(label)
        ax.set_title(note, fontsize=9, loc="left", color=INK_SECONDARY)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"Urban run, user {user + 1}: {run['user_count']} users, ACIR {run['acir_db']:g} dB, seed {run['seed']} "
        f"(dataset split: {run['split']})",
        fontsize=11, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)


def figure_local_mean(series, run: dict[str, Any], path: Path, user: int = 0) -> Path:
    t = np.arange(series.interference_dbm.shape[1]) * series.slot_duration_s
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _style(ax)
    ax.plot(t, series.interference_dbm[user], color=ORANGE, linewidth=1.0, alpha=0.9, label="raw faded interference (model input)")
    ax.plot(t, series.interference_local_dbm[user], color=BLUE, linewidth=2.2, label="local mean (prediction target: fading removed)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("interference (dBm)")
    ax.set_title(
        f"Urban run, user {user + 1}: faded series and local-mean target (ACIR {run['acir_db']:g} dB, seed {run['seed']})",
        fontsize=11, loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def compute_baseline_errors(data_root: Path, scenarios: tuple[str, ...] = ("urban", "rural")) -> dict[str, Any]:
    """Validation-split MAE (dB) of the three baselines. Only train/validation files are read (never test);
    nothing is trained except the 11 least-squares AR coefficients, fitted on train."""
    out: dict[str, Any] = {}
    for scenario in scenarios:
        x_train, y_train, x_val, y_val = load_train_val_arrays(data_root / scenario)
        ar = fit_ar_baseline(x_train, y_train)
        predictions = {
            "persistence": persistence_prediction(x_val),
            "moving_average_10": linear_moving_average_prediction(x_val),
            "linear_ar_10": ar_prediction(x_val, ar),
        }
        out[scenario] = {
            "n_val_windows": int(y_val.shape[0]),
            "val_mae_db": {name: mae_db(y_val, pred) for name, pred in predictions.items()},
        }
    return out


def figure_baseline_bars(errors: dict[str, Any], path: Path) -> Path:
    scenarios = list(errors)
    width = 0.26
    positions = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(8, 5))
    _style(ax)
    for offset, (key, label) in zip((-1, 0, 1), BASELINE_LABELS):
        values = [errors[s]["val_mae_db"][key] for s in scenarios]
        bars = ax.bar(positions + offset * width, values, width * 0.92, color=BASELINE_COLORS[key],
                      hatch=BASELINE_HATCHES[key], edgecolor=SURFACE, linewidth=1.2, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
    ax.set_xticks(positions, [s.capitalize() for s in scenarios])
    ax.set_ylabel("validation MAE (dB)")
    ax.set_xlabel("scenario (validation split, next-slot local-mean interference)")
    ax.set_ylim(0, max(v["val_mae_db"]["persistence"] for v in errors.values()) * 1.18)
    ax.set_title("Baseline errors; LSTM not shown here", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/phase3")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args(argv)
    data_root, out = Path(args.data_root), Path(args.out)

    metadata = json.loads((data_root / "urban" / "metadata.json").read_text(encoding="utf-8"))
    run, series = simulate_figure_run(metadata)
    paths = [
        figure_time_series(series, run, out / "urban_run_time_series.png"),
        figure_local_mean(series, run, out / "urban_local_mean_target.png"),
    ]
    errors = compute_baseline_errors(data_root)
    paths.append(figure_baseline_bars(errors, out / "baseline_errors.png"))

    (out / "baseline_errors.json").write_text(
        json.dumps(
            {
                "description": "Validation-split MAE (dB) of baselines for the next-slot local-mean target; no LSTM; test split not read.",
                "figure_run": {k: run[k] for k in ("run_id", "user_count", "acir_db", "seed", "split")},
                "baselines": errors,
                "run_info": collect_run_info(0, extra={"data_root": str(data_root)}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
