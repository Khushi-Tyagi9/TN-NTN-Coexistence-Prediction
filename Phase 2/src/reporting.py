"""Reporting, baseline artifact generation, and quality-gate helpers."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Callable

import matplotlib

from .config import SimulationConfig
from .simulation import (
    AggregatedSINRMetrics,
    BaselineRunResult,
    DeterministicSimulationResult,
    MonteCarloSimulationResult,
    run_deterministic_baseline,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_MIN_USERS = 4
BASELINE_MAX_USERS = 5
BASELINE_GNB_COUNT = 1
BASELINE_ARTIFACT_SCHEMA_VERSION = "baseline-v1"

SimulationResult = DeterministicSimulationResult | MonteCarloSimulationResult


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    """Typed pass/fail result for one quality gate."""

    gate_name: str
    passed: bool
    reasons: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.gate_name.strip():
            raise ValueError("gate_name cannot be empty.")
        if self.passed and self.reasons:
            raise ValueError("passed results cannot contain failure reasons.")
        if not self.passed and not self.reasons:
            raise ValueError("failed results must include one or more reasons.")


@dataclass(frozen=True, slots=True)
class QualityGateReport:
    """Collection of gate checks used to block or allow progression."""

    results: tuple[QualityGateResult, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("results cannot be empty.")

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        return tuple(
            reason
            for result in self.results
            if not result.passed
            for reason in result.reasons
        )

    def assert_passed(self) -> None:
        if self.passed:
            return
        raise ValueError(f"Quality gates failed: {'; '.join(self.failure_reasons)}")


@dataclass(frozen=True, slots=True)
class BaselineSummaryRecord:
    """Per-run summary record suitable for JSON/CSV export."""

    simulation_type: str
    run_id: str
    scenario_name: str
    user_count: int
    acir_db: float
    run_index: int
    seed: int
    mean_sinr_db: float
    min_sinr_db: float
    max_sinr_db: float
    mean_signal_power_dbm: float
    mean_interference_power_dbm: float
    noise_power_dbm: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_for_serialization(asdict(self))


@dataclass(frozen=True, slots=True)
class BaselineAggregateRecord:
    """Aggregate metrics for one deterministic or Monte Carlo sweep point."""

    simulation_type: str
    scenario_name: str
    user_count: int
    acir_db: float
    run_count: int
    mean_sinr_db: float
    min_sinr_db: float
    max_sinr_db: float
    mean_signal_power_dbm: float
    mean_interference_power_dbm: float

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_for_serialization(asdict(self))


@dataclass(frozen=True, slots=True)
class BaselineTraceRecord:
    """Per-user trace record for downstream model training and benchmarking."""

    simulation_type: str
    run_id: str
    scenario_name: str
    user_count: int
    acir_db: float
    run_index: int
    seed: int
    user_id: str
    serving_satellite_id: str
    signal_distance_km: float
    signal_path_loss_db: float
    signal_power_dbm: float
    interference_power_dbm: float
    noise_power_dbm: float
    sinr_db: float
    interferer_count: int

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_for_serialization(asdict(self))


@dataclass(frozen=True, slots=True)
class BaselineArtifactPaths:
    """Filesystem locations for generated baseline artifacts."""

    output_root: Path
    metrics_dir: Path
    traces_dir: Path
    plots_dir: Path
    summary_json: Path
    summary_csv: Path
    aggregate_json: Path
    aggregate_csv: Path
    traces_json: Path
    traces_csv: Path
    sinr_vs_acir_plot: Path


@dataclass(frozen=True)
class ReportData:
    """Generic report payload container."""

    result: SimulationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ReportBuilder:
    """Build baseline artifacts from simulation outputs."""

    def __init__(
        self,
        *,
        output_root: str | Path | None = None,
        artifact_stem: str = "baseline",
    ) -> None:
        self._output_root = output_root
        self._artifact_stem = artifact_stem

    def build(self, data: ReportData) -> BaselineArtifactPaths:
        """Generate baseline artifacts from provided simulation result data."""
        if data.result is None:
            raise ValueError("ReportData.result must be provided.")
        output_root = data.metadata.get("output_root", self._output_root)
        artifact_stem = str(data.metadata.get("artifact_stem", self._artifact_stem))
        return generate_baseline_artifacts(
            data.result,
            output_root=output_root,
            artifact_stem=artifact_stem,
        )


def _to_int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _is_finite_or_negative_infinity(value: float) -> bool:
    return value == float("-inf") or isfinite(value)


def _default_results_root() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def _sanitize_for_serialization(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_for_serialization(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_sanitize_for_serialization(item) for item in value]
    if isinstance(value, list):
        return [_sanitize_for_serialization(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def _simulation_type(result: SimulationResult) -> str:
    return "deterministic" if isinstance(result, DeterministicSimulationResult) else "monte_carlo"


def _iter_runs(result: SimulationResult) -> tuple[BaselineRunResult, ...]:
    if isinstance(result, DeterministicSimulationResult):
        return (result.run,)
    return tuple(
        run
        for sweep in result.sweep_results
        for run in sweep.runs
    )


def build_baseline_summary_records(result: SimulationResult) -> tuple[BaselineSummaryRecord, ...]:
    """Build per-run baseline summary records from deterministic or Monte Carlo outputs."""
    simulation_type = _simulation_type(result)
    records: list[BaselineSummaryRecord] = []
    for run in _iter_runs(result):
        records.append(
            BaselineSummaryRecord(
                simulation_type=simulation_type,
                run_id=run.run_id,
                scenario_name=run.parameters.scenario_name,
                user_count=run.parameters.user_count,
                acir_db=run.parameters.acir_db,
                run_index=run.parameters.run_index,
                seed=run.parameters.seed,
                mean_sinr_db=run.sinr.mean_sinr_db,
                min_sinr_db=run.sinr.min_sinr_db,
                max_sinr_db=run.sinr.max_sinr_db,
                mean_signal_power_dbm=run.sinr.mean_signal_power_dbm,
                mean_interference_power_dbm=run.sinr.mean_interference_power_dbm,
                noise_power_dbm=run.sinr.noise_power_dbm,
                metadata=dict(run.metadata),
            )
        )
    return tuple(records)


def build_baseline_aggregate_records(result: SimulationResult) -> tuple[BaselineAggregateRecord, ...]:
    """Build deterministic/sweep aggregate records from simulation outputs."""
    simulation_type = _simulation_type(result)
    if isinstance(result, DeterministicSimulationResult):
        run = result.run
        aggregate = result.aggregate
        return (
            BaselineAggregateRecord(
                simulation_type=simulation_type,
                scenario_name=run.parameters.scenario_name,
                user_count=run.parameters.user_count,
                acir_db=run.parameters.acir_db,
                run_count=aggregate.run_count,
                mean_sinr_db=aggregate.mean_sinr_db,
                min_sinr_db=aggregate.min_sinr_db,
                max_sinr_db=aggregate.max_sinr_db,
                mean_signal_power_dbm=aggregate.mean_signal_power_dbm,
                mean_interference_power_dbm=aggregate.mean_interference_power_dbm,
            ),
        )

    return tuple(
        BaselineAggregateRecord(
            simulation_type=simulation_type,
            scenario_name=sweep.scenario_name,
            user_count=sweep.user_count,
            acir_db=sweep.acir_db,
            run_count=sweep.aggregate.run_count,
            mean_sinr_db=sweep.aggregate.mean_sinr_db,
            min_sinr_db=sweep.aggregate.min_sinr_db,
            max_sinr_db=sweep.aggregate.max_sinr_db,
            mean_signal_power_dbm=sweep.aggregate.mean_signal_power_dbm,
            mean_interference_power_dbm=sweep.aggregate.mean_interference_power_dbm,
        )
        for sweep in result.sweep_results
    )


def build_baseline_trace_records(result: SimulationResult) -> tuple[BaselineTraceRecord, ...]:
    """Build per-user trace records from deterministic or Monte Carlo outputs."""
    simulation_type = _simulation_type(result)
    records: list[BaselineTraceRecord] = []
    for run in _iter_runs(result):
        for user_metric in run.sinr.user_metrics:
            records.append(
                BaselineTraceRecord(
                    simulation_type=simulation_type,
                    run_id=run.run_id,
                    scenario_name=run.parameters.scenario_name,
                    user_count=run.parameters.user_count,
                    acir_db=run.parameters.acir_db,
                    run_index=run.parameters.run_index,
                    seed=run.parameters.seed,
                    user_id=user_metric.user_id,
                    serving_satellite_id=user_metric.serving_satellite_id,
                    signal_distance_km=user_metric.signal_distance_km,
                    signal_path_loss_db=user_metric.signal_path_loss_db,
                    signal_power_dbm=user_metric.signal_power_dbm,
                    interference_power_dbm=user_metric.interference_power_dbm,
                    noise_power_dbm=user_metric.noise_power_dbm,
                    sinr_db=user_metric.sinr_db,
                    interferer_count=len(user_metric.interferers),
                )
            )
    return tuple(records)


def build_serializable_baseline_payload(result: SimulationResult) -> dict[str, Any]:
    """Build a stable, serializable artifact payload for baseline outputs."""
    summary_records = build_baseline_summary_records(result)
    aggregate_records = build_baseline_aggregate_records(result)
    trace_records = build_baseline_trace_records(result)
    trace_payload = [record.to_dict() for record in trace_records]
    return {
        "schema_version": BASELINE_ARTIFACT_SCHEMA_VERSION,
        "simulation_type": _simulation_type(result),
        "summary_records": [record.to_dict() for record in summary_records],
        "aggregate_records": [record.to_dict() for record in aggregate_records],
        "traces_records": trace_payload,
        "trace_records": trace_payload,
    }


def resolve_baseline_artifact_paths(
    *,
    output_root: str | Path | None = None,
    artifact_stem: str = "baseline",
) -> BaselineArtifactPaths:
    """Resolve and create baseline artifact output directories and file paths."""
    if not artifact_stem.strip():
        raise ValueError("artifact_stem cannot be empty.")

    root = Path(output_root) if output_root is not None else _default_results_root() / "baseline"
    metrics_dir = root / "metrics"
    traces_dir = root / "traces"
    plots_dir = root / "plots"
    for directory in (metrics_dir, traces_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return BaselineArtifactPaths(
        output_root=root,
        metrics_dir=metrics_dir,
        traces_dir=traces_dir,
        plots_dir=plots_dir,
        summary_json=metrics_dir / f"{artifact_stem}_summary.json",
        summary_csv=metrics_dir / f"{artifact_stem}_summary.csv",
        aggregate_json=metrics_dir / f"{artifact_stem}_aggregate.json",
        aggregate_csv=metrics_dir / f"{artifact_stem}_aggregate.csv",
        traces_json=traces_dir / f"{artifact_stem}_traces.json",
        traces_csv=traces_dir / f"{artifact_stem}_traces.csv",
        sinr_vs_acir_plot=plots_dir / f"{artifact_stem}_sinr_vs_acir.png",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_sanitize_for_serialization(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows_for_csv(records: tuple[BaselineSummaryRecord, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        payload = record.to_dict()
        metadata = payload.pop("metadata", {})
        payload["metadata_json"] = json.dumps(metadata, sort_keys=True)
        rows.append(payload)
    return rows


def _generate_sinr_vs_acir_plot(
    aggregate_records: tuple[BaselineAggregateRecord, ...],
    *,
    output_path: Path,
) -> None:
    grouped: dict[tuple[str, int], list[BaselineAggregateRecord]] = defaultdict(list)
    for record in aggregate_records:
        grouped[(record.scenario_name, record.user_count)].append(record)

    figure, axis = plt.subplots(figsize=(6, 4))
    for scenario_key in sorted(grouped):
        scenario_name, user_count = scenario_key
        sorted_records = sorted(grouped[scenario_key], key=lambda record: record.acir_db)
        axis.plot(
            [record.acir_db for record in sorted_records],
            [record.mean_sinr_db for record in sorted_records],
            marker="o",
            label=f"{scenario_name} / users={user_count}",
        )

    axis.set_title("Baseline SINR vs ACIR")
    axis.set_xlabel("ACIR (dB)")
    axis.set_ylabel("Mean SINR (dB)")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if grouped:
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def generate_baseline_artifacts(
    result: SimulationResult,
    *,
    output_root: str | Path | None = None,
    artifact_stem: str = "baseline",
) -> BaselineArtifactPaths:
    """Generate baseline JSON/CSV/plot artifacts under results subfolders."""
    paths = resolve_baseline_artifact_paths(output_root=output_root, artifact_stem=artifact_stem)
    summary_records = build_baseline_summary_records(result)
    aggregate_records = build_baseline_aggregate_records(result)
    trace_records = build_baseline_trace_records(result)

    _write_json(
        paths.summary_json,
        {
            "schema_version": BASELINE_ARTIFACT_SCHEMA_VERSION,
            "record_type": "summary",
            "records": [record.to_dict() for record in summary_records],
        },
    )
    _write_csv(paths.summary_csv, _summary_rows_for_csv(summary_records))

    _write_json(
        paths.aggregate_json,
        {
            "schema_version": BASELINE_ARTIFACT_SCHEMA_VERSION,
            "record_type": "aggregate",
            "records": [record.to_dict() for record in aggregate_records],
        },
    )
    _write_csv(paths.aggregate_csv, [record.to_dict() for record in aggregate_records])

    _write_json(
        paths.traces_json,
        {
            "schema_version": BASELINE_ARTIFACT_SCHEMA_VERSION,
            "record_type": "traces",
            "records": [record.to_dict() for record in trace_records],
        },
    )
    _write_csv(paths.traces_csv, [record.to_dict() for record in trace_records])

    _generate_sinr_vs_acir_plot(aggregate_records, output_path=paths.sinr_vs_acir_plot)
    return paths


def validate_baseline_realism_constraints(run: BaselineRunResult) -> QualityGateResult:
    """Validate explicit baseline realism constraints using run outputs."""
    reasons: list[str] = []
    metadata = run.metadata

    hap_count = _to_int_or_none(metadata.get("hap_count"))
    if hap_count is None:
        reasons.append("Missing integer metadata field: hap_count.")
    elif hap_count < 1:
        reasons.append(f"Baseline realism failed: hap_count={hap_count}, expected >= 1.")

    gnb_count = _to_int_or_none(metadata.get("gnb_count"))
    if gnb_count is None:
        reasons.append("Missing integer metadata field: gnb_count.")
    elif gnb_count != BASELINE_GNB_COUNT:
        reasons.append(f"Baseline realism failed: gnb_count={gnb_count}, expected {BASELINE_GNB_COUNT}.")

    satellite_count = _to_int_or_none(metadata.get("satellite_count"))
    if satellite_count is None:
        reasons.append("Missing integer metadata field: satellite_count.")
    elif satellite_count < 1:
        reasons.append(
            f"Baseline realism failed: satellite_count={satellite_count}, expected >= 1."
        )

    generated_user_count = _to_int_or_none(metadata.get("generated_user_count"))
    if generated_user_count is None:
        reasons.append("Missing integer metadata field: generated_user_count.")
    elif not (BASELINE_MIN_USERS <= generated_user_count <= BASELINE_MAX_USERS):
        reasons.append(
            "Baseline realism failed: generated_user_count="
            f"{generated_user_count}, expected in [{BASELINE_MIN_USERS}, {BASELINE_MAX_USERS}]."
        )

    parameter_user_count = run.parameters.user_count
    if not (BASELINE_MIN_USERS <= parameter_user_count <= BASELINE_MAX_USERS):
        reasons.append(
            "Baseline realism failed: parameters.user_count="
            f"{parameter_user_count}, expected in [{BASELINE_MIN_USERS}, {BASELINE_MAX_USERS}]."
        )

    sinr_user_count = len(run.sinr.user_metrics)
    if not (BASELINE_MIN_USERS <= sinr_user_count <= BASELINE_MAX_USERS):
        reasons.append(
            "Baseline realism failed: len(sinr.user_metrics)="
            f"{sinr_user_count}, expected in [{BASELINE_MIN_USERS}, {BASELINE_MAX_USERS}]."
        )
    if parameter_user_count != sinr_user_count:
        reasons.append(
            "Baseline realism failed: parameters.user_count "
            f"({parameter_user_count}) != len(sinr.user_metrics) ({sinr_user_count})."
        )
    if generated_user_count is not None and generated_user_count != sinr_user_count:
        reasons.append(
            "Baseline realism failed: generated_user_count "
            f"({generated_user_count}) != len(sinr.user_metrics) ({sinr_user_count})."
        )

    return QualityGateResult(
        gate_name="baseline_realism_constraints",
        passed=not reasons,
        reasons=tuple(reasons),
        details={"run_id": run.run_id},
    )


def validate_physical_plausibility(
    run: BaselineRunResult,
    aggregate: AggregatedSINRMetrics | None = None,
) -> QualityGateResult:
    """Validate physically plausible and internally ordered SINR outputs."""
    reasons: list[str] = []
    summary = run.sinr

    if not isfinite(summary.mean_sinr_db):
        reasons.append("Non-finite mean_sinr_db detected.")
    if not isfinite(summary.min_sinr_db):
        reasons.append("Non-finite min_sinr_db detected.")
    if not isfinite(summary.max_sinr_db):
        reasons.append("Non-finite max_sinr_db detected.")
    if summary.min_sinr_db > summary.mean_sinr_db or summary.mean_sinr_db > summary.max_sinr_db:
        reasons.append("SINR ordering failed: expected min_sinr_db <= mean_sinr_db <= max_sinr_db.")
    if not isfinite(summary.mean_signal_power_dbm):
        reasons.append("Non-finite mean_signal_power_dbm detected.")
    if not _is_finite_or_negative_infinity(summary.mean_interference_power_dbm):
        reasons.append("mean_interference_power_dbm must be finite or -inf.")
    if not isfinite(summary.noise_power_dbm):
        reasons.append("Non-finite noise_power_dbm detected.")

    for user_metric in summary.user_metrics:
        if not isfinite(user_metric.sinr_db):
            reasons.append(f"User {user_metric.user_id} has non-finite sinr_db.")

    if aggregate is not None:
        if aggregate.run_count <= 0:
            reasons.append("aggregate.run_count must be > 0.")
        if not isfinite(aggregate.mean_sinr_db):
            reasons.append("Aggregate mean_sinr_db is non-finite.")
        if not isfinite(aggregate.min_sinr_db):
            reasons.append("Aggregate min_sinr_db is non-finite.")
        if not isfinite(aggregate.max_sinr_db):
            reasons.append("Aggregate max_sinr_db is non-finite.")
        if aggregate.min_sinr_db > aggregate.mean_sinr_db or aggregate.mean_sinr_db > aggregate.max_sinr_db:
            reasons.append(
                "Aggregate SINR ordering failed: expected min_sinr_db <= mean_sinr_db <= max_sinr_db."
            )

    return QualityGateResult(
        gate_name="physical_plausibility",
        passed=not reasons,
        reasons=tuple(reasons),
        details={"run_id": run.run_id},
    )


def validate_run_count_consistency(
    result: DeterministicSimulationResult | MonteCarloSimulationResult,
) -> QualityGateResult:
    """Validate run counts are positive and consistent with payload shape."""
    reasons: list[str] = []

    if isinstance(result, DeterministicSimulationResult):
        if result.aggregate.run_count <= 0:
            reasons.append("Deterministic aggregate.run_count must be > 0.")
        if result.aggregate.run_count != 1:
            reasons.append(
                f"Deterministic aggregate.run_count={result.aggregate.run_count}, expected 1."
            )
        details: dict[str, Any] = {"run_id": result.run.run_id}
    else:
        if result.monte_carlo_runs <= 0:
            reasons.append("monte_carlo_runs must be > 0.")
        for sweep in result.sweep_results:
            actual_runs = len(sweep.runs)
            if actual_runs <= 0:
                reasons.append(
                    f"Sweep ({sweep.scenario_name}, users={sweep.user_count}, acir={sweep.acir_db}) has no runs."
                )
            if sweep.aggregate.run_count <= 0:
                reasons.append(
                    f"Sweep ({sweep.scenario_name}, users={sweep.user_count}, acir={sweep.acir_db}) "
                    "aggregate.run_count must be > 0."
                )
            if sweep.aggregate.run_count != actual_runs:
                reasons.append(
                    f"Sweep ({sweep.scenario_name}, users={sweep.user_count}, acir={sweep.acir_db}) "
                    f"aggregate.run_count={sweep.aggregate.run_count} != len(runs)={actual_runs}."
                )
        details = {"sweep_count": len(result.sweep_results)}

    return QualityGateResult(
        gate_name="run_count_consistency",
        passed=not reasons,
        reasons=tuple(reasons),
        details=details,
    )


def validate_reproducibility(
    config: SimulationConfig,
    *,
    scenario_name: str | None = None,
    user_count: int | None = None,
    acir_db: float | None = None,
    seed: int | None = None,
    deterministic_runner: Callable[..., DeterministicSimulationResult] = run_deterministic_baseline,
) -> QualityGateResult:
    """Run deterministic baseline twice and check for exact reproducibility."""
    first = deterministic_runner(
        config,
        scenario_name=scenario_name,
        user_count=user_count,
        acir_db=acir_db,
        seed=seed,
    )
    second = deterministic_runner(
        config,
        scenario_name=scenario_name,
        user_count=user_count,
        acir_db=acir_db,
        seed=seed,
    )

    passed = first == second
    reasons = ()
    if not passed:
        reasons = (
            "Reproducibility check failed: deterministic outputs differ for identical seed/config.",
        )

    resolved_seed = first.run.parameters.seed
    return QualityGateResult(
        gate_name="deterministic_reproducibility",
        passed=passed,
        reasons=reasons,
        details={
            "scenario_name": first.run.parameters.scenario_name,
            "user_count": first.run.parameters.user_count,
            "acir_db": first.run.parameters.acir_db,
            "seed": resolved_seed,
        },
    )


def evaluate_quality_gates(
    result: DeterministicSimulationResult | MonteCarloSimulationResult,
    *,
    config: SimulationConfig | None = None,
    include_reproducibility: bool = True,
) -> QualityGateReport:
    """Evaluate do-not-advance gates for simulation outputs."""
    checks: list[QualityGateResult] = []

    if isinstance(result, DeterministicSimulationResult):
        checks.append(validate_baseline_realism_constraints(result.run))
        checks.append(validate_physical_plausibility(result.run, result.aggregate))
        checks.append(validate_run_count_consistency(result))
        if include_reproducibility and config is not None:
            checks.append(
                validate_reproducibility(
                    config,
                    scenario_name=result.run.parameters.scenario_name,
                    user_count=result.run.parameters.user_count,
                    acir_db=result.run.parameters.acir_db,
                    seed=result.run.parameters.seed,
                )
            )
    else:
        for sweep in result.sweep_results:
            for run in sweep.runs:
                checks.append(validate_baseline_realism_constraints(run))
                checks.append(validate_physical_plausibility(run, sweep.aggregate))
        checks.append(validate_run_count_consistency(result))

    return QualityGateReport(results=tuple(checks))
