"""CLI: simulate time series and write Phase 3 LSTM datasets.

Example (run from the "Phase 2" folder):
    python build_phase3_dataset.py --out data/phase3 --runs-per-point 8

For each scenario this writes, under --out:
    <scenario>/                  primary dataset: next-slot LOCAL-MEAN interference target
    <scenario>_instantaneous/    same runs, instantaneous (faded) interference target
    <scenario>_nosinr/           (only with --ablate-sinr) local-mean target, sinr_db channel zeroed
Each folder holds X_train.npy ... y_test.npy + metadata.json (channel parameters, seeds, split).
Train on one with:  python run_phase3_colab.py --data-root data/phase3/urban
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import SimulationConfig
from src.phase3_dataset_builder import DatasetBuildConfig, build_phase3_dataset, dataset_dir_name, save_dataset
from src.timeseries import TimeSeriesConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default="data/phase3", help="Output root directory.")
    parser.add_argument("--scenarios", nargs="+", default=["urban", "rural"], choices=["urban", "rural"])
    parser.add_argument("--targets", nargs="+", default=["local_mean", "instantaneous"],
                        choices=["local_mean", "instantaneous"], help="Target definitions to build.")
    parser.add_argument("--ablate-sinr", action="store_true",
                        help="Also build the local-mean dataset with the sinr_db channel zeroed (ablation).")
    parser.add_argument("--runs-per-point", type=int, default=6, help="Runs per (user_count, ACIR) pair (>= 3).")
    parser.add_argument("--n-slots", type=int, default=600, help="Slots per run.")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 2, 3], help="Predict this many slots ahead.")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed (a per-scenario offset is added).")
    args = parser.parse_args()

    sim_config = SimulationConfig()
    ts_config = TimeSeriesConfig(n_slots=args.n_slots)
    variants = [(target, ()) for target in args.targets]
    if args.ablate_sinr:
        variants.append(("local_mean", ("sinr_db",)))

    for scenario in args.scenarios:
        for target, mask in variants:
            build = DatasetBuildConfig(
                scenario=scenario,
                runs_per_point=args.runs_per_point,
                horizon_slots=args.horizon,
                base_seed=args.seed,
                target=target,
                mask_features=mask,
            )
            split, metadata = build_phase3_dataset(sim_config, ts_config, build)
            out_dir = save_dataset(split, metadata, Path(args.out) / dataset_dir_name(scenario, target, mask))
            counts = metadata["window_counts"]
            print(
                f"[{scenario}/{target}{'/mask=' + ','.join(mask) if mask else ''}] "
                f"train={counts['train']} val={counts['val']} test={counts['test']} -> {out_dir}"
            )


if __name__ == "__main__":
    main()
