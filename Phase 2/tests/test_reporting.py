"""Tests for report artifacts and quality-gate validation utilities."""

import csv
from dataclasses import replace
import json

import pytest

from src.config import SimulationConfig
from src.reporting import (
    ReportBuilder,
    ReportData,
    build_serializable_baseline_payload,
    evaluate_quality_gates,
    generate_baseline_artifacts,
    validate_baseline_realism_constraints,
    validate_physical_plausibility,
    validate_reproducibility,
    validate_run_count_consistency,
)
from src.simulation import SimulationLogRecord, run_deterministic_baseline, run_monte_carlo_baseline


def test_report_data_defaults() -> None:
    """ReportData should be constructible with default metadata."""
    assert ReportData().metadata == {}


def test_serializable_baseline_payload_structure() -> None:
    """Serializable payload should include stable baseline record sections."""
    result = run_monte_carlo_baseline(
        SimulationConfig(),
        monte_carlo_runs=1,
        scenario_order=("urban",),
        user_count_sweep=(4,),
        acir_sweep_db=(20.0, 30.0),
        base_seed=13,
    )

    payload = build_serializable_baseline_payload(result)

    assert payload["schema_version"] == "baseline-v1"
    assert payload["simulation_type"] == "monte_carlo"
    assert {"summary_records", "aggregate_records", "traces_records", "trace_records"} <= set(payload)
    assert payload["trace_records"] == payload["traces_records"]
    summary_record = payload["summary_records"][0]
    assert {"run_id", "scenario_name", "user_count", "acir_db", "metadata"} <= set(summary_record)
    assert isinstance(summary_record["metadata"], dict)
    json.dumps(payload)


def test_generate_baseline_artifacts_writes_files(tmp_path) -> None:
    """Artifact generation should emit JSON and CSV outputs to the configured location."""
    result = run_deterministic_baseline(
        SimulationConfig(),
        scenario_name="urban",
        user_count=4,
        acir_db=30.0,
        seed=12,
    )

    output_root = tmp_path / "baseline-artifacts"
    artifacts = generate_baseline_artifacts(result, output_root=output_root, artifact_stem="unit")

    assert artifacts.output_root == output_root
    assert artifacts.metrics_dir == output_root / "metrics"
    assert artifacts.traces_dir == output_root / "traces"
    assert artifacts.plots_dir == output_root / "plots"
    assert artifacts.metrics_dir.is_dir()
    assert artifacts.traces_dir.is_dir()
    assert artifacts.plots_dir.is_dir()
    assert artifacts.summary_json.exists()
    assert artifacts.summary_csv.exists()
    assert artifacts.aggregate_json.exists()
    assert artifacts.aggregate_csv.exists()
    assert artifacts.traces_json.exists()
    assert artifacts.traces_csv.exists()
    assert artifacts.sinr_vs_acir_plot.exists()

    summary_payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    assert summary_payload["schema_version"] == "baseline-v1"
    assert summary_payload["record_type"] == "summary"
    assert summary_payload["records"][0]["run_id"] == result.run.run_id
    aggregate_payload = json.loads(artifacts.aggregate_json.read_text(encoding="utf-8"))
    assert aggregate_payload["record_type"] == "aggregate"
    traces_payload = json.loads(artifacts.traces_json.read_text(encoding="utf-8"))
    assert traces_payload["record_type"] == "traces"
    assert len(traces_payload["records"]) == len(result.run.sinr.user_metrics)

    with artifacts.summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows
    assert summary_rows[0]["run_id"] == result.run.run_id

    with artifacts.aggregate_csv.open("r", encoding="utf-8", newline="") as handle:
        aggregate_rows = list(csv.DictReader(handle))
    assert aggregate_rows

    with artifacts.traces_csv.open("r", encoding="utf-8", newline="") as handle:
        trace_rows = list(csv.DictReader(handle))
    assert len(trace_rows) == len(result.run.sinr.user_metrics)


def test_report_builder_generates_plot_file(tmp_path) -> None:
    """ReportBuilder should generate and return a concrete SINR-vs-ACIR plot artifact."""
    result = run_monte_carlo_baseline(
        SimulationConfig(),
        monte_carlo_runs=1,
        scenario_order=("urban",),
        user_count_sweep=(4,),
        acir_sweep_db=(20.0, 30.0),
        base_seed=21,
    )

    artifacts = ReportBuilder(output_root=tmp_path, artifact_stem="builder").build(ReportData(result=result))

    assert artifacts.sinr_vs_acir_plot == tmp_path / "plots" / "builder_sinr_vs_acir.png"
    assert artifacts.sinr_vs_acir_plot.exists()
    assert artifacts.sinr_vs_acir_plot.stat().st_size > 0


def test_quality_gates_pass_for_normal_deterministic_baseline() -> None:
    """Baseline deterministic output should satisfy realism, plausibility, and reproducibility gates."""
    config = SimulationConfig()
    result = run_deterministic_baseline(config, scenario_name="urban", user_count=4, acir_db=30.0, seed=12)

    report = evaluate_quality_gates(result, config=config)

    assert report.passed
    assert report.failure_reasons == ()


def test_baseline_realism_gate_fails_with_clear_reason() -> None:
    """Realism gate should fail with explicit reason when counts violate baseline constraints."""
    result = run_deterministic_baseline(SimulationConfig(), scenario_name="urban", user_count=4, acir_db=25.0, seed=7)
    bad_run = replace(
        result.run,
        metadata={**result.run.metadata, "hap_count": 0},
    )

    gate = validate_baseline_realism_constraints(bad_run)

    assert not gate.passed
    assert any("hap_count=0" in reason for reason in gate.reasons)


def test_baseline_realism_gate_fails_when_gnb_count_is_not_one() -> None:
    """Realism gate should fail when metadata gNB count is outside baseline."""
    result = run_deterministic_baseline(SimulationConfig(), scenario_name="urban", user_count=4, acir_db=25.0, seed=11)
    bad_run = replace(
        result.run,
        metadata={**result.run.metadata, "gnb_count": 2},
    )

    gate = validate_baseline_realism_constraints(bad_run)

    assert not gate.passed
    assert any("gnb_count=2" in reason for reason in gate.reasons)


def test_physical_plausibility_gate_fails_on_unordered_summary() -> None:
    """Plausibility gate should reject outputs where min/mean/max ordering is broken."""
    result = run_deterministic_baseline(SimulationConfig(), scenario_name="urban", user_count=4, acir_db=30.0, seed=5)
    bad_summary = replace(
        result.run.sinr,
        min_sinr_db=result.run.sinr.mean_sinr_db + 1.0,
        max_sinr_db=result.run.sinr.mean_sinr_db - 1.0,
    )
    bad_run = replace(result.run, sinr=bad_summary)

    gate = validate_physical_plausibility(bad_run, result.aggregate)

    assert not gate.passed
    assert any("ordering failed" in reason.lower() for reason in gate.reasons)


def test_run_count_gate_fails_for_mismatched_monte_carlo_aggregate_count() -> None:
    """Run-count gate should fail when aggregate.run_count disagrees with run list length."""
    result = run_monte_carlo_baseline(
        SimulationConfig(),
        monte_carlo_runs=1,
        scenario_order=("urban",),
        user_count_sweep=(4,),
        acir_sweep_db=(30.0,),
        base_seed=9,
    )
    first_sweep = result.sweep_results[0]
    bad_sweep = replace(
        first_sweep,
        aggregate=replace(first_sweep.aggregate, run_count=2),
    )
    bad_result = replace(result, sweep_results=(bad_sweep,))

    gate = validate_run_count_consistency(bad_result)

    assert not gate.passed
    assert any("aggregate.run_count=2" in reason for reason in gate.reasons)


def test_reproducibility_gate_fails_for_nondeterministic_runner() -> None:
    """Reproducibility gate should surface non-identical outputs with a clear failure message."""
    config = SimulationConfig()
    invocation_counter = {"count": 0}

    def unstable_runner(
        cfg: SimulationConfig,
        *,
        scenario_name: str | None = None,
        user_count: int | None = None,
        acir_db: float | None = None,
        seed: int | None = None,
    ):
        invocation_counter["count"] += 1
        base = run_deterministic_baseline(
            cfg,
            scenario_name=scenario_name,
            user_count=user_count,
            acir_db=acir_db,
            seed=seed,
        )
        return replace(
            base,
            metadata_records=base.metadata_records
            + (
                SimulationLogRecord(
                    event="unstable_marker",
                    metadata={"invocation": invocation_counter["count"]},
                ),
            ),
        )

    gate = validate_reproducibility(
        config,
        scenario_name="urban",
        user_count=4,
        acir_db=30.0,
        seed=42,
        deterministic_runner=unstable_runner,
    )

    assert not gate.passed
    assert any("reproducibility check failed" in reason.lower() for reason in gate.reasons)

