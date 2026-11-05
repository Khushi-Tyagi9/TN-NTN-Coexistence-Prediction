"""Colab-friendly entrypoint for Phase 3 staged training and mitigation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.phase3_drl import DrlEnvConfig, PPOTrainingConfig
from src.phase3_pipeline import Phase3Gates, run_phase3_pipeline, to_serializable_summary
from src.phase3_prediction import LSTMTrainingConfig, TransformerTrainingConfig
from src.reproducibility import collect_run_info


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 pipeline (LSTM -> Transformer fallback if needed -> ZF -> optional PPO)."
    )
    parser.add_argument("--data-root", type=str, required=True, help="Directory containing Phase 3 .npy files.")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario name prefix (e.g., urban, rural).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the split, model init, batch shuffling and PPO.")
    parser.add_argument("--checkpoint-dir", type=str, default="results/phase3/checkpoints", help="Directory for model checkpoints.")
    parser.add_argument("--summary-json", type=str, default="results/phase3/phase3_summary.json", help="Output JSON summary path.")
    parser.add_argument("--lstm-hidden-size", type=int, default=96, help="LSTM hidden size in [64,128].")
    parser.add_argument("--lstm-layers", type=int, default=1, choices=[1, 2], help="Number of LSTM layers.")
    parser.add_argument("--lstm-epochs", type=int, default=120, help="Max LSTM epochs.")
    parser.add_argument("--transformer-d-model", type=int, default=96, help="Transformer model width.")
    parser.add_argument("--transformer-heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--transformer-layers", type=int, default=2, help="Transformer encoder layers.")
    parser.add_argument("--transformer-epochs", type=int, default=140, help="Max Transformer epochs.")
    parser.add_argument("--target-accuracy", type=float, default=0.80, help="Legacy within-10%%-error threshold (informational only, not gated on).")
    parser.add_argument("--min-skill", type=float, default=0.15,
                        help="Gate: model MAE (dB) must be at least this fraction below persistence MAE.")
    parser.add_argument("--zf-required-db", type=float, default=15.0, help="Minimum ZF suppression gate for DRL.")
    parser.add_argument("--desired-angle-rad", type=float, default=0.0, help="Desired TN steering direction in radians.")
    parser.add_argument("--ntn-angle-rad", type=float, default=0.35, help="NTN direction in radians.")
    parser.add_argument(
        "--max-tracking-offset-rad",
        type=float,
        default=0.12,
        help="Max NTN angle offset inferred from predicted interference.",
    )
    parser.add_argument("--antenna-count", type=int, default=64, help="Antenna count for ZF evaluation.")
    parser.add_argument(
        "--min-final-skill",
        type=float,
        default=0.05,
        help="Final gate (reported, never blocks mitigation): MAE this fraction below the best of persistence / 10-slot linear MA / linear AR(10).",
    )
    parser.add_argument(
        "--allow-window-shuffle",
        action="store_true",
        help="Allow a random per-window split of unsplit X.npy/y.npy when no groups.npy exists (LEAKY; avoid).",
    )
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help="Train/evaluate the predictors only; do not run the mitigation stage.",
    )
    parser.add_argument(
        "--no-transformer-fallback",
        action="store_true",
        help="Do not train the Transformer even if the LSTM misses the prototype gate.",
    )
    parser.add_argument("--run-drl", action="store_true", help="Train PPO after ZF gate passes.")
    return parser


def _read_dataset_metadata(data_root: Path) -> dict | None:
    """metadata.json written by build_phase3_dataset.py (None for hand-made datasets)."""
    path = data_root / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_training_configs(args: argparse.Namespace) -> tuple[LSTMTrainingConfig, TransformerTrainingConfig]:
    """Training configs from CLI args; ``--seed`` seeds init and shuffling of both models."""
    lstm_cfg = LSTMTrainingConfig(
        hidden_size=args.lstm_hidden_size,
        num_layers=args.lstm_layers,
        max_epochs=args.lstm_epochs,
        seed=args.seed,
    )
    transformer_cfg = TransformerTrainingConfig(
        d_model=args.transformer_d_model,
        num_heads=args.transformer_heads,
        num_layers=args.transformer_layers,
        max_epochs=args.transformer_epochs,
        seed=args.seed,
    )
    return lstm_cfg, transformer_cfg


def _print_prediction_report(payload: dict) -> None:
    dataset = payload.get("dataset")
    if dataset is not None:
        print(f"Dataset target: {dataset['target_kind']} ({dataset['target']})")
    lstm_payload = payload["prediction"]["lstm"]
    print(f"LSTM test within-10%-error accuracy (legacy, informational): {lstm_payload['test_within_10pct_accuracy']:.4f}")
    for model_name in ("lstm", "transformer"):
        model_payload = payload["prediction"][model_name]
        db = None if model_payload is None else model_payload["test_db_metrics"]
        if db is None:
            continue
        print(
            f"[{model_name}] test MAE {db['mae_db']:.3f} dB | RMSE {db['rmse_db']:.3f} dB | "
            f"within 1 dB {db['within_1db']:.3f} | within 3 dB {db['within_3db']:.3f}"
        )
        print(
            f"[{model_name}] baselines MAE (dB): persistence {db['persistence_mae_db']:.3f} | "
            f"10-slot linear MA {db['moving_average_mae_db']:.3f} | linear AR(10) {db['ar_mae_db']:.3f} "
            f"(best: {db['best_baseline']})"
        )
        print(
            f"[{model_name}] skill vs persistence {db['skill_vs_persistence']:+.3f} "
            f"(need >= {db['min_skill_required']:.2f}) -> prototype gate: {db['meets_prototype_target']}"
        )
        print(
            f"[{model_name}] skill vs best baseline {db['skill_vs_best_baseline']:+.3f} "
            f"(need >= {db['min_final_skill_required']:.2f}) -> final gate: {db['meets_final_target']}"
        )
    print(f"LSTM prototype gate passed (gates mitigation): {lstm_payload['meets_target']}")
    print(f"LSTM final gate passed (reported only): {lstm_payload['meets_final_target']}")
    print(f"Transformer fallback executed: {payload['prediction']['transformer_executed']}")


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    gates = Phase3Gates(
        lstm_target_accuracy=args.target_accuracy,
        lstm_min_skill_vs_persistence=args.min_skill,
        lstm_min_final_skill=args.min_final_skill,
        zf_required_suppression_db=args.zf_required_db,
    )
    lstm_cfg, transformer_cfg = build_training_configs(args)

    result = run_phase3_pipeline(
        data_root=args.data_root,
        scenario=args.scenario,
        seed=args.seed,
        desired_angle_rad=args.desired_angle_rad,
        ntn_angle_rad=args.ntn_angle_rad,
        antenna_count=args.antenna_count,
        max_tracking_offset_rad=args.max_tracking_offset_rad,
        gates=gates,
        lstm_config=lstm_cfg,
        transformer_config=transformer_cfg,
        checkpoint_dir=args.checkpoint_dir,
        allow_window_shuffle=args.allow_window_shuffle,
        run_transformer_if_eligible=not args.no_transformer_fallback,
        skip_mitigation=args.prediction_only,
        train_drl=args.run_drl,
        env_config=DrlEnvConfig(seed=args.seed),
        ppo_config=PPOTrainingConfig(seed=args.seed),
    )
    payload = to_serializable_summary(result)
    payload["run_info"] = collect_run_info(
        args.seed,
        extra={
            "argv": argv if argv is not None else sys.argv[1:],
            "data_root": str(args.data_root),
            "scenario": args.scenario,
        },
    )
    dataset_meta = _read_dataset_metadata(Path(args.data_root))
    if dataset_meta is not None:
        payload["dataset"] = {
            key: dataset_meta.get(key)
            for key in ("scenario", "target_kind", "target", "target_definition", "masked_features", "provenance")
        }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_prediction_report(payload)
    mitigation = payload["mitigation"]
    if mitigation is None:
        print(f"Mitigation not run: {payload['mitigation_skipped_reason']}")
    else:
        print(f"ZF suppression (dB): {mitigation['suppression_db']:.4f}")
        print(f"Mitigation predictor: {mitigation['predictor_used']}")
        print(f"ZF gate passed: {mitigation['zf_gate_passed']}")
        print(f"PPO trained: {mitigation['ppo_trained']}")
    print(f"Summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
