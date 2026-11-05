"""Aggregate repeated-seed runs and evaluate the gates on the MEAN test MAE (Stage B).

Declared before training: the prototype and final gates are evaluated on the mean test MAE over the
seeds (not on any single seed). Baseline MAEs depend only on the test set and the train split, so they
must agree across seeds; a mismatch is an error, not something to average away.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .phase3_metrics import (
    DEFAULT_MIN_FINAL_SKILL,
    DEFAULT_MIN_SKILL_VS_PERSISTENCE,
    DbMetrics,
    skill_vs_baseline,
)

BASELINE_AGREEMENT_RTOL = 1e-9


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,  # sample std
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_seed_runs(
    seeds: list[int],
    metrics: list[DbMetrics],
    *,
    min_skill: float = DEFAULT_MIN_SKILL_VS_PERSISTENCE,
    min_final_skill: float = DEFAULT_MIN_FINAL_SKILL,
) -> dict[str, Any]:
    """Per-seed rows, mean/std, baselines, and both gates evaluated on the mean test MAE."""
    if not metrics or len(seeds) != len(metrics):
        raise ValueError("Need one DbMetrics per seed (and at least one).")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique.")
    for item in metrics:
        if item.ar_mae_db is None or item.moving_average_mae_db is None:
            raise ValueError("Metrics must be evaluated with all baselines (AR coefficients supplied).")

    reference = metrics[0]
    for name in ("persistence_mae_db", "moving_average_mae_db", "ar_mae_db"):
        for item in metrics[1:]:
            if not np.isclose(getattr(item, name), getattr(reference, name), rtol=BASELINE_AGREEMENT_RTOL, atol=0.0):
                raise ValueError(f"{name} differs between seeds; baselines must be seed-independent.")

    baselines = {
        "persistence": reference.persistence_mae_db,
        "moving_average_10": reference.moving_average_mae_db,
        "linear_ar_10": reference.ar_mae_db,
    }
    best_name = min(baselines, key=lambda name: baselines[name])
    best_mae = baselines[best_name]

    mae_stats = _mean_std([m.mae_db for m in metrics])
    mean_mae = mae_stats["mean"]
    skill_persistence = skill_vs_baseline(mean_mae, baselines["persistence"])
    skill_best = skill_vs_baseline(mean_mae, best_mae)

    return {
        "gate_basis": f"mean test MAE over seeds {list(seeds)}",
        "n_seeds": len(seeds),
        "per_seed": [
            {
                "seed": int(seed),
                "mae_db": m.mae_db,
                "rmse_db": m.rmse_db,
                "within_1db": m.within_1db,
                "within_3db": m.within_3db,
                "skill_vs_persistence": m.skill_vs_persistence,
                "skill_vs_best_baseline": m.skill_vs_best_baseline,
            }
            for seed, m in zip(seeds, metrics)
        ],
        "mae_db": mae_stats,
        "rmse_db": _mean_std([m.rmse_db for m in metrics]),
        "within_1db": _mean_std([m.within_1db for m in metrics]),
        "within_3db": _mean_std([m.within_3db for m in metrics]),
        "baselines_mae_db": baselines,
        "best_baseline": best_name,
        "best_baseline_mae_db": best_mae,
        "skill_vs_persistence_on_mean_mae": skill_persistence,
        "skill_vs_best_baseline_on_mean_mae": skill_best,
        "min_skill_required": min_skill,
        "min_final_skill_required": min_final_skill,
        "meets_prototype_target": bool(skill_persistence >= min_skill),
        "meets_final_target": bool(skill_best >= min_final_skill),
        "transformer_required": bool(skill_persistence < min_skill),
    }
