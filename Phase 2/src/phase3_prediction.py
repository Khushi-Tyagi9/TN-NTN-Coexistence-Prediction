"""Phase 3 forecasting models and training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .reproducibility import seed_everything
from .phase3_data import EXPECTED_FEATURE_COUNT, EXPECTED_LOOKBACK_SLOTS, Phase3DatasetSplit
from .phase3_metrics import (
    DEFAULT_MIN_FINAL_SKILL,
    DEFAULT_MIN_SKILL_VS_PERSISTENCE,
    DbMetrics,
    PredictionMetrics,
    evaluate_db_metrics,
    evaluate_prediction_metrics,
    fit_ar_baseline,
    meets_final_target,
    meets_skill_gate,
)


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for Phase 3 forecasting. Install Colab dependencies from "
            "requirements_phase3_colab.txt."
        ) from exc
    return torch, nn


def _resolve_device(torch: Any, requested_device: str | None) -> str:
    if requested_device is not None:
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, slots=True)
class LSTMTrainingConfig:
    """LSTM training controls."""

    hidden_size: int = 96
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 120
    early_stopping_patience: int = 12
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    device: str | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be >= 0.")
        if not (64 <= self.hidden_size <= 128):
            raise ValueError("hidden_size must be within [64, 128] for project constraints.")
        if self.num_layers not in {1, 2}:
            raise ValueError("num_layers must be either 1 or 2.")
        if self.dropout < 0:
            raise ValueError("dropout must be >= 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be > 0.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be > 0.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be > 0.")


@dataclass(frozen=True, slots=True)
class TransformerTrainingConfig:
    """Transformer training controls."""

    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 192
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 140
    early_stopping_patience: int = 14
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    device: str | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be >= 0.")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0.")
        if self.feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be > 0.")
        if self.dropout < 0:
            raise ValueError("dropout must be >= 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be > 0.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be > 0.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be > 0.")


@dataclass(frozen=True, slots=True)
class PredictionModelResult:
    """Training/evaluation outputs for one prediction model."""

    model_name: str
    best_epoch: int
    best_validation_loss: float
    validation_metrics: PredictionMetrics
    test_metrics: PredictionMetrics
    target_accuracy: float
    meets_target: bool
    checkpoint_path: str | None = None
    last_test_prediction: float = 0.0
    last_test_target: float = 0.0
    test_target_min: float = 0.0
    test_target_max: float = 0.0
    validation_db_metrics: DbMetrics | None = None
    test_db_metrics: DbMetrics | None = None
    min_skill: float = DEFAULT_MIN_SKILL_VS_PERSISTENCE
    meets_final_target: bool = False  # does NOT gate mitigation; only meets_target does
    min_final_skill: float = DEFAULT_MIN_FINAL_SKILL

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_skill < 1.0):
            raise ValueError("min_skill must be in [0, 1).")
        if not (0.0 <= self.min_final_skill < 1.0):
            raise ValueError("min_final_skill must be in [0, 1).")
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if self.best_epoch <= 0:
            raise ValueError("best_epoch must be > 0.")
        if self.best_validation_loss < 0:
            raise ValueError("best_validation_loss must be >= 0.")
        if not (0 < self.target_accuracy <= 1):
            raise ValueError("target_accuracy must be in (0, 1].")
        for field_name in ("last_test_prediction", "last_test_target", "test_target_min", "test_target_max"):
            if not np.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite.")
        if self.test_target_min > self.test_target_max:
            raise ValueError("test_target_min cannot exceed test_target_max.")


@dataclass(frozen=True, slots=True)
class PredictionStageResult:
    """Full prediction stage result with optional Transformer leg."""

    lstm: PredictionModelResult
    transformer: PredictionModelResult | None
    transformer_executed: bool


def meets_prediction_target(metrics: PredictionMetrics, *, target_accuracy: float = 0.80) -> bool:
    """Legacy within-10%-relative-error check. INFORMATIONAL ONLY.

    On dBm targets this is trivially met by a persistence baseline (AUDIT.md C1);
    stage gating uses ``meets_skill_gate`` instead."""
    if not (0 < target_accuracy <= 1):
        raise ValueError("target_accuracy must be in (0, 1].")
    return metrics.within_10pct_accuracy >= target_accuracy


def _make_regression_loaders(split: Phase3DatasetSplit, batch_size: int, seed: int = 0) -> tuple[Any, Any, Any]:
    torch, _ = _require_torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    def _dataset(x: np.ndarray, y: np.ndarray) -> Any:
        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        return torch.utils.data.TensorDataset(x_tensor, y_tensor)

    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        _dataset(split.X_train, split.y_train), batch_size=batch_size, shuffle=True, generator=shuffle_generator
    )
    val_loader = torch.utils.data.DataLoader(_dataset(split.X_val, split.y_val), batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(_dataset(split.X_test, split.y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    """Train-split normalization parameters reused across split subsets."""

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    def __post_init__(self) -> None:
        if self.x_mean.shape != (1, 1, EXPECTED_FEATURE_COUNT):
            raise ValueError("x_mean must have shape (1, 1, EXPECTED_FEATURE_COUNT).")
        if self.x_std.shape != (1, 1, EXPECTED_FEATURE_COUNT):
            raise ValueError("x_std must have shape (1, 1, EXPECTED_FEATURE_COUNT).")
        if not np.all(np.isfinite(self.x_mean)) or not np.all(np.isfinite(self.x_std)):
            raise ValueError("x_mean and x_std must be finite.")
        if np.any(self.x_std <= 0):
            raise ValueError("x_std entries must be > 0.")
        if not np.isfinite(self.y_mean):
            raise ValueError("y_mean must be finite.")
        if not np.isfinite(self.y_std) or self.y_std <= 0:
            raise ValueError("y_std must be finite and > 0.")

    def transform_features(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.x_mean) / self.x_std).astype(np.float32)

    def transform_targets(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=np.float32) - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        return (np.asarray(y_scaled, dtype=np.float32) * self.y_std + self.y_mean).astype(np.float32)


def _fit_normalization(split: Phase3DatasetSplit) -> NormalizationStats:
    x_train = np.asarray(split.X_train, dtype=np.float32)
    y_train = np.asarray(split.y_train, dtype=np.float32)
    x_mean = np.mean(x_train, axis=(0, 1), keepdims=True)
    x_std = np.std(x_train, axis=(0, 1), keepdims=True)
    x_std = np.where(x_std > 1e-8, x_std, 1.0)
    y_mean = float(np.mean(y_train))
    y_std_raw = float(np.std(y_train))
    y_std = y_std_raw if y_std_raw > 1e-8 else 1.0
    return NormalizationStats(
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean,
        y_std=y_std,
    )


def _apply_normalization(split: Phase3DatasetSplit, stats: NormalizationStats) -> Phase3DatasetSplit:
    return Phase3DatasetSplit(
        X_train=stats.transform_features(split.X_train),
        y_train=stats.transform_targets(split.y_train),
        X_val=stats.transform_features(split.X_val),
        y_val=stats.transform_targets(split.y_val),
        X_test=stats.transform_features(split.X_test),
        y_test=stats.transform_targets(split.y_test),
    )


def _fit_model(
    *,
    model: Any,
    train_loader: Any,
    val_loader: Any,
    max_epochs: int,
    early_stopping_patience: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    device: str,
) -> tuple[Any, int, float]:
    torch, _ = _require_torch()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    patience_counter = 0

    model.to(device)
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        model.eval()
        weighted_loss_sum = 0.0
        sample_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                batch_size = int(batch_x.shape[0])
                weighted_loss_sum += float(loss.item()) * batch_size
                sample_count += batch_size

        if sample_count <= 0:
            raise ValueError("Validation split is empty; cannot train model.")
        val_loss = weighted_loss_sum / sample_count
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    if best_state is None or best_epoch <= 0:
        raise RuntimeError("Training did not produce a valid best checkpoint.")
    model.load_state_dict(best_state)
    return model, best_epoch, float(best_val_loss)


def _predict(model: Any, data: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    torch, _ = _require_torch()
    model.eval()
    data_loader = torch.utils.data.DataLoader(torch.from_numpy(np.asarray(data, dtype=np.float32)), batch_size=batch_size)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x.to(device)
            predictions = model(batch_x)
            outputs.append(predictions.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).reshape(-1)


def _build_lstm_model(config: LSTMTrainingConfig) -> Any:
    _, nn = _require_torch()

    class LSTMForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=EXPECTED_FEATURE_COUNT,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.output = nn.Linear(config.hidden_size, 1)

        def forward(self, x: Any) -> Any:
            sequence_outputs, _ = self.lstm(x)
            return self.output(sequence_outputs[:, -1, :])

    return LSTMForecaster()


def _build_transformer_model(config: TransformerTrainingConfig) -> Any:
    torch, nn = _require_torch()

    class TransformerForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(EXPECTED_FEATURE_COUNT, config.d_model)
            self.position = nn.Parameter(torch.zeros(1, EXPECTED_LOOKBACK_SLOTS, config.d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.output = nn.Linear(config.d_model, 1)

        def forward(self, x: Any) -> Any:
            encoded = self.input_projection(x) + self.position[:, : x.shape[1], :]
            transformed = self.encoder(encoded)
            return self.output(transformed[:, -1, :])

    return TransformerForecaster()


def _save_checkpoint(model: Any, checkpoint_path: Path) -> None:
    torch, _ = _require_torch()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def train_lstm_forecaster(
    split: Phase3DatasetSplit,
    *,
    config: LSTMTrainingConfig | None = None,
    target_accuracy: float = 0.80,
    min_skill: float = DEFAULT_MIN_SKILL_VS_PERSISTENCE,
    min_final_skill: float = DEFAULT_MIN_FINAL_SKILL,
    checkpoint_path: str | Path | None = None,
) -> PredictionModelResult:
    """Train and evaluate the LSTM forecaster."""
    torch, _ = _require_torch()
    cfg = LSTMTrainingConfig() if config is None else config
    device = _resolve_device(torch, cfg.device)
    seed_everything(cfg.seed)  # before model init so weights and shuffling are reproducible
    normalization = _fit_normalization(split)
    normalized_split = _apply_normalization(split, normalization)

    model = _build_lstm_model(cfg)
    train_loader, val_loader, _ = _make_regression_loaders(normalized_split, cfg.batch_size, seed=cfg.seed)
    model, best_epoch, best_val_loss = _fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.early_stopping_patience,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        gradient_clip_norm=cfg.gradient_clip_norm,
        device=device,
    )

    val_predictions_scaled = _predict(model, normalized_split.X_val, batch_size=cfg.batch_size, device=device)
    test_predictions_scaled = _predict(model, normalized_split.X_test, batch_size=cfg.batch_size, device=device)
    val_predictions = normalization.inverse_targets(val_predictions_scaled)
    test_predictions = normalization.inverse_targets(test_predictions_scaled)
    validation_metrics = evaluate_prediction_metrics(split.y_val, val_predictions)
    test_metrics = evaluate_prediction_metrics(split.y_test, test_predictions)
    ar_coefficients = fit_ar_baseline(split.X_train, split.y_train)  # baseline fitted on train only
    validation_db_metrics = evaluate_db_metrics(
        split.y_val, val_predictions, split.X_val, ar_coefficients=ar_coefficients
    )
    test_db_metrics = evaluate_db_metrics(
        split.y_test, test_predictions, split.X_test, ar_coefficients=ar_coefficients
    )
    last_test_prediction = float(test_predictions[-1])
    last_test_target = float(np.asarray(split.y_test, dtype=np.float32)[-1])
    test_target_min = float(np.min(split.y_test))
    test_target_max = float(np.max(split.y_test))

    saved_checkpoint: str | None = None
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        _save_checkpoint(model, checkpoint)
        saved_checkpoint = str(checkpoint)

    return PredictionModelResult(
        model_name="lstm",
        best_epoch=best_epoch,
        best_validation_loss=best_val_loss,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        target_accuracy=target_accuracy,
        meets_target=meets_skill_gate(test_db_metrics, min_skill=min_skill),
        validation_db_metrics=validation_db_metrics,
        test_db_metrics=test_db_metrics,
        min_skill=min_skill,
        meets_final_target=meets_final_target(test_db_metrics, min_final_skill=min_final_skill),
        min_final_skill=min_final_skill,
        checkpoint_path=saved_checkpoint,
        last_test_prediction=last_test_prediction,
        last_test_target=last_test_target,
        test_target_min=test_target_min,
        test_target_max=test_target_max,
    )


def train_transformer_forecaster(
    split: Phase3DatasetSplit,
    *,
    config: TransformerTrainingConfig | None = None,
    target_accuracy: float = 0.80,
    min_skill: float = DEFAULT_MIN_SKILL_VS_PERSISTENCE,
    min_final_skill: float = DEFAULT_MIN_FINAL_SKILL,
    checkpoint_path: str | Path | None = None,
) -> PredictionModelResult:
    """Train and evaluate the Transformer forecaster."""
    torch, _ = _require_torch()
    cfg = TransformerTrainingConfig() if config is None else config
    device = _resolve_device(torch, cfg.device)
    seed_everything(cfg.seed)  # before model init so weights and shuffling are reproducible
    normalization = _fit_normalization(split)
    normalized_split = _apply_normalization(split, normalization)

    model = _build_transformer_model(cfg)
    train_loader, val_loader, _ = _make_regression_loaders(normalized_split, cfg.batch_size, seed=cfg.seed)
    model, best_epoch, best_val_loss = _fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.early_stopping_patience,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        gradient_clip_norm=cfg.gradient_clip_norm,
        device=device,
    )

    val_predictions_scaled = _predict(model, normalized_split.X_val, batch_size=cfg.batch_size, device=device)
    test_predictions_scaled = _predict(model, normalized_split.X_test, batch_size=cfg.batch_size, device=device)
    val_predictions = normalization.inverse_targets(val_predictions_scaled)
    test_predictions = normalization.inverse_targets(test_predictions_scaled)
    validation_metrics = evaluate_prediction_metrics(split.y_val, val_predictions)
    test_metrics = evaluate_prediction_metrics(split.y_test, test_predictions)
    ar_coefficients = fit_ar_baseline(split.X_train, split.y_train)  # baseline fitted on train only
    validation_db_metrics = evaluate_db_metrics(
        split.y_val, val_predictions, split.X_val, ar_coefficients=ar_coefficients
    )
    test_db_metrics = evaluate_db_metrics(
        split.y_test, test_predictions, split.X_test, ar_coefficients=ar_coefficients
    )
    last_test_prediction = float(test_predictions[-1])
    last_test_target = float(np.asarray(split.y_test, dtype=np.float32)[-1])
    test_target_min = float(np.min(split.y_test))
    test_target_max = float(np.max(split.y_test))

    saved_checkpoint: str | None = None
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        _save_checkpoint(model, checkpoint)
        saved_checkpoint = str(checkpoint)

    return PredictionModelResult(
        model_name="transformer",
        best_epoch=best_epoch,
        best_validation_loss=best_val_loss,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        target_accuracy=target_accuracy,
        meets_target=meets_skill_gate(test_db_metrics, min_skill=min_skill),
        validation_db_metrics=validation_db_metrics,
        test_db_metrics=test_db_metrics,
        min_skill=min_skill,
        meets_final_target=meets_final_target(test_db_metrics, min_final_skill=min_final_skill),
        min_final_skill=min_final_skill,
        checkpoint_path=saved_checkpoint,
        last_test_prediction=last_test_prediction,
        last_test_target=last_test_target,
        test_target_min=test_target_min,
        test_target_max=test_target_max,
    )


def run_prediction_stage(
    split: Phase3DatasetSplit,
    *,
    lstm_config: LSTMTrainingConfig | None = None,
    transformer_config: TransformerTrainingConfig | None = None,
    target_accuracy: float = 0.80,
    min_skill: float = DEFAULT_MIN_SKILL_VS_PERSISTENCE,
    min_final_skill: float = DEFAULT_MIN_FINAL_SKILL,
    run_transformer_if_eligible: bool = True,
    checkpoint_dir: str | Path | None = None,
) -> PredictionStageResult:
    """Run LSTM first, then the Transformer fallback only if the LSTM misses the skill-vs-persistence gate."""
    output_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    lstm_checkpoint = None if output_dir is None else output_dir / "lstm_best.pt"
    lstm_result = train_lstm_forecaster(
        split,
        config=lstm_config,
        target_accuracy=target_accuracy,
        min_skill=min_skill,
        min_final_skill=min_final_skill,
        checkpoint_path=lstm_checkpoint,
    )

    transformer_result: PredictionModelResult | None = None
    transformer_executed = False
    if run_transformer_if_eligible and not lstm_result.meets_target:
        transformer_executed = True
        transformer_checkpoint = None if output_dir is None else output_dir / "transformer_best.pt"
        transformer_result = train_transformer_forecaster(
            split,
            config=transformer_config,
            target_accuracy=target_accuracy,
            min_skill=min_skill,
            min_final_skill=min_final_skill,
            checkpoint_path=transformer_checkpoint,
        )

    return PredictionStageResult(
        lstm=lstm_result,
        transformer=transformer_result,
        transformer_executed=transformer_executed,
    )
