"""Static figures for the prediction stage (training curves, actual vs predicted).

Colours are slots 1-3 of the validated categorical palette (blue, orange, aqua) on a light surface;
the ground truth is neutral ink. Every multi-series panel has a legend, so identity is never colour-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEED_COLORS = (BLUE, ORANGE, AQUA)


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def _finish(fig: plt.Figure, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_training_curves(
    histories: Mapping[int, Mapping[str, list[float]]],
    best_epochs: Mapping[int, int],
    path: str | Path,
    title: str,
) -> Path:
    """Train (dashed) and validation (solid) loss per epoch for every seed; the best epoch is marked."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _style(ax)
    for color, (seed, history) in zip(SEED_COLORS, sorted(histories.items())):
        epochs = np.arange(1, len(history["val_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], color=color, linewidth=1.5, linestyle="--", alpha=0.9)
        ax.plot(epochs, history["val_loss"], color=color, linewidth=2.0, label=f"seed {seed}")
        best = best_epochs[seed]
        ax.plot([best], [history["val_loss"][best - 1]], marker="o", markersize=7, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.plot([], [], color=INK_SECONDARY, linestyle="--", linewidth=1.5, label="train (dashed)")
    ax.plot([], [], color=INK_SECONDARY, linewidth=2.0, label="validation (solid); dot = best epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalised MSE")
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _finish(fig, path)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    path: str | Path,
    title: str,
    *,
    snippet_slots: int = 200,
    max_scatter_points: int = 6000,
    seed: int = 0,
    split_name: str = "test",
) -> Path:
    """Three panels: LSTM scatter vs truth, a contiguous test snippet, and the absolute-error CDF.

    ``predictions`` must contain "lstm", "persistence" and "ar10". Consecutive test windows come from
    one user's run, so the first ``snippet_slots`` windows form a real time series.
    """
    truth = np.asarray(y_true, dtype=np.float64)
    series = {name: np.asarray(values, dtype=np.float64) for name, values in predictions.items()}
    labels = {"lstm": "LSTM", "persistence": "persistence", "ar10": "linear AR(10)"}
    colors = {"lstm": BLUE, "persistence": ORANGE, "ar10": AQUA}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax in axes:
        _style(ax)

    rng = np.random.default_rng(seed)
    pick = rng.choice(truth.size, size=min(max_scatter_points, truth.size), replace=False)
    ax = axes[0]
    ax.scatter(truth[pick], series["lstm"][pick], s=5, color=BLUE, alpha=0.35, linewidths=0)
    low, high = float(truth.min()), float(truth.max())
    ax.plot([low, high], [low, high], color=INK, linewidth=1.2, label="ideal (y = x)")
    ax.set_xlabel("actual next-slot target (dBm)")
    ax.set_ylabel("LSTM prediction (dBm)")
    ax.set_title(f"LSTM vs actual (random subset of {split_name} windows)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    count = min(snippet_slots, truth.size)
    slots = np.arange(count)
    ax.plot(slots, truth[:count], color=INK, linewidth=2.0, label="actual")
    for key in ("persistence", "lstm"):
        ax.plot(slots, series[key][:count], color=colors[key], linewidth=1.5, label=labels[key])
    ax.set_xlabel(f"consecutive {split_name} windows (one user, 0.5 s slots)")
    ax.set_ylabel("interference (dBm)")
    ax.set_title(f"First {count} {split_name} windows", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    for key in ("lstm", "persistence", "ar10"):
        errors = np.sort(np.abs(series[key] - truth))
        ax.plot(errors, np.arange(1, errors.size + 1) / errors.size, color=colors[key], linewidth=2.0, label=labels[key])
    for tolerance in (1.0, 3.0):
        ax.axvline(tolerance, color=INK_SECONDARY, linestyle=":", linewidth=1.0)
        ax.text(tolerance, 0.02, f" {tolerance:g} dB", color=INK_SECONDARY, fontsize=8, va="bottom")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)
    ax.set_xlabel("absolute error (dB)")
    ax.set_ylabel(f"share of {split_name} windows with error <= x")
    ax.set_title("Absolute-error CDF", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, path)
