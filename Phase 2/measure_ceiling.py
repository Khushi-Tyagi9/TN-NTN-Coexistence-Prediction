"""Measure the interference-free coverage ceiling (docs/PHASE4_DESIGN.md, section 9 item 1).

The ceiling is the share of user-slots whose interference-free SNR (satellite signal minus thermal noise, no gNB
interference at all) is at least 15 dB. No controller, null or model is involved, and ACIR does not enter.

For each scenario the test runs of the Stage B dataset (``metadata.json`` runs with split == "test") are re-simulated from
their recorded seeds, and the re-simulation is checked to reproduce the saved ``X_test.npy`` exactly, so these are the same
runs. Output: results/phase4/ceiling/ceiling.json and README.md.

    python measure_ceiling.py [--data-root data/phase3] [--out results/phase4/ceiling]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.config import SimulationConfig
from src.interference import compute_noise_power_dbm
from src.phase3_dataset_builder import make_windows
from src.phase4_metrics import coverage_probability, interference_free_sinr_db
from src.reproducibility import collect_run_info
from src.timeseries import TimeSeriesConfig, _fspl_db, simulate_time_series

SNR_THRESHOLD_DB = 15.0
HAND_ESTIMATE_SHARE = 0.808  # docs/PHASE4_DESIGN.md section 9 item 1: |x| <= 909 km of the +-1125 km pass ("about 81%")
SCENARIOS = ("urban", "rural")


def analytic_nadir_line_share(sim_config: SimulationConfig, ts_config: TimeSeriesConfig) -> float:
    """Share of slots with SNR >= 15 dB for a user on the sub-satellite line (no jitter, no user offset).

    Re-does the hand calculation with the simulator's own constants and slot grid, as a check of the arithmetic in the design doc.
    """
    sat = sim_config.satellite
    half_span_km = 0.5 * ts_config.satellite_speed_km_s * ts_config.n_slots * ts_config.slot_duration_s
    t_s = np.arange(ts_config.n_slots) * ts_config.slot_duration_s
    x_km = -half_span_km + ts_config.satellite_speed_km_s * t_s
    slant_km = np.sqrt(x_km**2 + sat.leo_altitude_km**2)
    signal_dbm = sat.tx_power_dbw + 30.0 + sat.tx_gain_dbi - _fspl_db(slant_km, sim_config.radio.carrier_frequency_hz)
    noise_dbm = compute_noise_power_dbm(sim_config.radio.bandwidth_hz, sim_config.radio.noise_figure_db)
    return coverage_probability(interference_free_sinr_db(signal_dbm, noise_dbm), SNR_THRESHOLD_DB)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spread(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,  # sample std
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(array.size),
    }


def measure_scenario(data_dir: Path) -> dict[str, Any]:
    """Interference-free coverage over the dataset's test runs, and a check that they match X_test.npy."""
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    sim_config = SimulationConfig()
    ts_config = TimeSeriesConfig(**metadata["timeseries_config"])
    test_runs = [run for run in metadata["runs"] if run["split"] == "test"]
    if not test_runs:
        raise ValueError(f"No test runs in {data_dir / 'metadata.json'}.")

    runs: list[dict[str, Any]] = []
    windows: list[np.ndarray] = []
    all_user_shares: list[float] = []
    covered = total = 0
    for run in test_runs:
        series = simulate_time_series(
            sim_config, ts_config, scenario_name=metadata["scenario"], user_count=run["user_count"],
            acir_db=run["acir_db"], seed=run["seed"],
        )
        snr_db = interference_free_sinr_db(series.signal_dbm, series.noise_power_dbm)  # (users, slots)
        user_shares = [coverage_probability(snr_db[user], SNR_THRESHOLD_DB) for user in range(snr_db.shape[0])]
        all_user_shares += user_shares
        covered += int(np.sum(snr_db >= SNR_THRESHOLD_DB))
        total += int(snr_db.size)
        runs.append(
            {
                "run_id": run["run_id"], "user_count": run["user_count"], "acir_db": run["acir_db"], "seed": run["seed"],
                "n_user_slots": int(snr_db.size),
                "share_snr_ge_threshold": coverage_probability(snr_db, SNR_THRESHOLD_DB),
                "min_snr_db": float(snr_db.min()), "mean_snr_db": float(snr_db.mean()), "max_snr_db": float(snr_db.max()),
            }
        )
        for user_series in series.features():
            windows.append(make_windows(user_series, horizon=metadata["horizon_slots"])[0])

    saved_x_test = np.load(data_dir / "X_test.npy")
    rebuilt = np.concatenate(windows, axis=0)
    reproduces = bool(rebuilt.shape == saved_x_test.shape and np.array_equal(rebuilt, saved_x_test))

    analytic = analytic_nadir_line_share(sim_config, ts_config)
    run_shares = [run["share_snr_ge_threshold"] for run in runs]
    pooled = covered / total
    return {
        "scenario": metadata["scenario"],
        "n_test_runs": len(runs),
        "n_user_series": len(all_user_shares),
        "n_user_slots": total,
        "runs": runs,
        "over_runs": _spread(run_shares),
        "over_user_series": _spread(all_user_shares),
        "pooled_share": pooled,
        "analytic_nadir_line_share": analytic,
        "hand_estimate_share": HAND_ESTIMATE_SHARE,
        "difference_vs_hand_estimate_pp": 100.0 * (pooled - HAND_ESTIMATE_SHARE),
        "analytic_minus_hand_estimate_pp": 100.0 * (analytic - HAND_ESTIMATE_SHARE),
        "reproduces_saved_X_test": reproduces,
        "dataset": {
            "directory": str(data_dir).replace("\\", "/"),
            "X_test_sha256": _sha256(data_dir / "X_test.npy"),
            "metadata_sha256": _sha256(data_dir / "metadata.json"),
            "generated_at_git_sha": metadata["provenance"]["git_sha"],
            "window_counts": metadata["window_counts"],
        },
    }


def render_readme(result: dict[str, Any]) -> str:
    def row(name: str) -> str:
        r = result["scenarios"][name]
        o, u = r["over_runs"], r["over_user_series"]
        return (f"| {name} | {r['n_test_runs']} | {100 * o['mean']:.2f}% | {100 * o['std']:.2f} pp | {100 * o['min']:.2f}% - {100 * o['max']:.2f}% | "
                f"{100 * r['pooled_share']:.2f}% | {100 * u['std']:.2f} pp | {r['analytic_nadir_line_share'] * 100:.2f}% | "
                f"{r['difference_vs_hand_estimate_pp']:+.2f} pp | {r['reproduces_saved_X_test']} |")

    ri = result["run_info"]
    pooled = [result["scenarios"][name]["pooled_share"] for name in SCENARIOS]
    if max(pooled) < 0.95:
        absolute_target_sentence = (
            f"The absolute 95% coverage target at {SNR_THRESHOLD_DB:g} dB is unattainable by this measurement as well (the highest measured ceiling is {100 * max(pooled):.2f}%)."
        )
    else:
        absolute_target_sentence = (
            f"The measured ceiling reaches 95% in at least one scenario ({100 * max(pooled):.2f}%), so the absolute target is not ruled out by link budget alone there."
        )
    hand = 100 * HAND_ESTIMATE_SHARE
    return f"""# Interference-free coverage ceiling (Phase 4, first implementation task)

Measures the quantity defined in `docs/PHASE4_DESIGN.md` section 9 item 1: the share of user-slots whose **interference-free SNR is at
least {SNR_THRESHOLD_DB:g} dB**, i.e. the highest coverage probability at a {SNR_THRESHOLD_DB:g} dB SINR threshold that any controller could reach.
No criterion, threshold or parameter was changed by this measurement.

## How it was measured

1. For each scenario, the **test runs** of the Stage B dataset (`data/phase3/<scenario>/metadata.json`, runs with split `test`: one per
   user-count x ACIR stratum) are re-simulated with `simulate_time_series` from their recorded seeds, with the default `SimulationConfig`
   and the dataset's own `TimeSeriesConfig` (600 slots of 0.5 s).
2. The re-simulation is checked to reproduce the saved `X_test.npy` bit for bit (column `reproduces X_test` below), so these are the same
   runs Stage B tested on.
3. Interference-free SNR = `signal_dbm - noise_power_dbm` per user and slot (`interference_free_sinr_db`); the share with SNR >= {SNR_THRESHOLD_DB:g} dB
   is `coverage_probability`. Noise is {result['noise_power_dbm']:.2f} dBm
   (bandwidth and noise figure from the default `SimulationConfig`). No model, controller or interference enters, and ACIR has no effect on the result.
4. Statistics: share per run (over all its user-slots); mean, sample standard deviation (n-1), min and max **over runs**; the pooled share
   over all user-slots; the standard deviation over individual user series is also given.
5. Cross-check of the hand calculation: the same share computed analytically for a user on the sub-satellite line (no jitter, no user offset)
   with the simulator's constants and slot grid.

## Results (SNR >= {SNR_THRESHOLD_DB:g} dB)

| scenario | test runs | mean over runs | std over runs | min - max over runs | pooled over user-slots | std over user series | analytic sub-satellite line | pooled minus {hand:.1f}% hand estimate | reproduces X_test |
|---|---|---|---|---|---|---|---|---|---|
{row('urban')}
{row('rural')}

## Comparison with the 81% hand estimate

The design document estimated about 81% (precisely {hand:.1f}%: `|x| <= 909 km` of the +-1125 km pass). The measured pooled shares are
{100 * result['scenarios']['urban']['pooled_share']:.2f}% (urban) and {100 * result['scenarios']['rural']['pooled_share']:.2f}% (rural), i.e. {result['scenarios']['urban']['difference_vs_hand_estimate_pp']:+.2f} pp and
{result['scenarios']['rural']['difference_vs_hand_estimate_pp']:+.2f} pp relative to the hand estimate. The analytic share with the simulator's own constants
and 0.5 s slot grid is {100 * result['scenarios']['urban']['analytic_nadir_line_share']:.2f}% ({result['scenarios']['urban']['analytic_minus_hand_estimate_pp']:+.2f} pp vs the hand estimate).

Consequences, with nothing changed: criterion A2 is `coverage >= 0.95 x ceiling` using the ceiling measured on the Phase 4 evaluation runs with this same
method. {absolute_target_sentence} The numbers here are for the Stage B test runs only; the ceiling
depends on the satellite pass, user positions and the 0.3 dB signal jitter, not on ACIR or interference.

## Provenance

* git SHA at measurement start: `{ri['git_sha'][:7]}`, tree dirty: {ri['git_dirty']} (files: {ri['git_dirty_files'] or 'none'}); python {ri['python']}, numpy {ri['numpy']}.
* Datasets generated at: urban `{result['scenarios']['urban']['dataset']['generated_at_git_sha'][:7]}`, rural `{result['scenarios']['rural']['dataset']['generated_at_git_sha'][:7]}`;
  `X_test.npy` SHA-256 in `ceiling.json`.
* Reproduce: `python measure_ceiling.py` (needs `data/phase3/urban` and `data/phase3/rural`, generated by `build_phase3_dataset.py`).
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/phase3")
    parser.add_argument("--out", default="results/phase4/ceiling")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"Refusing to overwrite {out}.")
    run_info = collect_run_info(0, extra={"argv": argv if argv is not None else sys.argv[1:], "note": "no randomness of its own; seeds come from the dataset metadata"})

    result: dict[str, Any] = {
        "description": "Interference-free coverage ceiling: share of user-slots with SNR >= threshold, over the Stage B test runs.",
        "snr_threshold_db": SNR_THRESHOLD_DB,
        "definition": "SNR = satellite signal dBm - thermal noise dBm (no gNB interference); share over all user-slots of a run",
        "scenarios": {name: measure_scenario(Path(args.data_root) / name) for name in SCENARIOS},
        "noise_power_dbm": compute_noise_power_dbm(
            SimulationConfig().radio.bandwidth_hz, SimulationConfig().radio.noise_figure_db
        ),
        "run_info": run_info,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "ceiling.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "README.md").write_text(render_readme(result), encoding="utf-8")
    for name, r in result["scenarios"].items():
        o = r["over_runs"]
        print(f"[{name}] ceiling mean over {o['n']} runs {100 * o['mean']:.2f}% (std {100 * o['std']:.2f} pp, min {100 * o['min']:.2f}%, max {100 * o['max']:.2f}%); "
              f"pooled {100 * r['pooled_share']:.2f}%; analytic {100 * r['analytic_nadir_line_share']:.2f}%; "
              f"vs hand {HAND_ESTIMATE_SHARE * 100:.1f}%: {r['difference_vs_hand_estimate_pp']:+.2f} pp; reproduces X_test: {r['reproduces_saved_X_test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
