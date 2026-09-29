"""Colab-friendly entrypoint for Phase 3 staged training and mitigation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.phase3_pipeline import Phase3Gates, run_phase3_pipeline, to_serializable_summary
from src.phase3_prediction import LSTMTrainingConfig, TransformerTrainingConfig


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 pipeline (LSTM -> Transformer fallback if needed -> ZF -> optional PPO)."
    )
    parser.add_argument("--data-root", type=str, required=True, help="Directory containing Phase 3 .npy files.")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario name prefix (e.g., urban, rural).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--checkpoint-dir", type=str, default="results/phase3/checkpoints", help="Directory for model checkpoints.")
    parser.add_argument("--summary-json", type=str, default="results/phase3/phase3_summary.json", help="Output JSON summary path.")
    parser.add_argument("--lstm-hidden-size", type=int, default=96, help="LSTM hidden size in [64,128].")
    parser.add_argument("--lstm-layers", type=int, default=1, choices=[1, 2], help="Number of LSTM layers.")
    parser.add_argument("--lstm-epochs", type=int, default=120, help="Max LSTM epochs.")
    parser.add_argument("--transformer-d-model", type=int, default=96, help="Transformer model width.")
    parser.add_argument("--transformer-heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--transformer-layers", type=int, default=2, help="Transformer encoder layers.")
    parser.add_argument("--transformer-epochs", type=int, default=140, help="Max Transformer epochs.")
    parser.add_argument("--target-accuracy", type=float, default=0.80, help="Prototype gate threshold for within-10%-error accuracy.")
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
    parser.add_argument("--run-drl", action="store_true", help="Train PPO after ZF gate passes.")
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    gates = Phase3Gates(
        lstm_target_accuracy=args.target_accuracy,
        zf_required_suppression_db=args.zf_required_db,
    )
    lstm_cfg = LSTMTrainingConfig(
        hidden_size=args.lstm_hidden_size,
        num_layers=args.lstm_layers,
        max_epochs=args.lstm_epochs,
    )
    transformer_cfg = TransformerTrainingConfig(
        d_model=args.transformer_d_model,
        num_heads=args.transformer_heads,
        num_layers=args.transformer_layers,
        max_epochs=args.transformer_epochs,
    )

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
        train_drl=args.run_drl,
    )
    payload = to_serializable_summary(result)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"LSTM test within-10%-error accuracy: {payload['prediction']['lstm']['test_within_10pct_accuracy']:.4f}")
    print(f"LSTM gate passed: {payload['prediction']['lstm']['meets_target']}")
    print(f"Transformer fallback executed: {payload['prediction']['transformer_executed']}")
    print(f"ZF suppression (dB): {payload['mitigation']['suppression_db']:.4f}")
    print(f"Mitigation predictor: {payload['mitigation']['predictor_used']}")
    print(f"ZF gate passed: {payload['mitigation']['zf_gate_passed']}")
    print(f"PPO trained: {payload['mitigation']['ppo_trained']}")
    print(f"Summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
