"""Real (tiny, CPU) training tests for the LSTM/Transformer forecasters (audit M2).

These execute the code that the monkeypatched stage/pipeline tests never touch: normalization,
data loaders, ``_fit_model``, ``_predict``, checkpoints and scoring. They are skipped, loudly
(``-rs`` in pytest.ini), when PyTorch is not installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed: training code is NOT exercised")

from src.phase3_data import Phase3DatasetSplit  # noqa: E402
from src.phase3_prediction import (  # noqa: E402
    LSTMTrainingConfig,
    TransformerTrainingConfig,
    _build_lstm_model,
    _build_transformer_model,
    _make_regression_loaders,
    _predict,
    run_prediction_stage,
    train_lstm_forecaster,
    train_transformer_forecaster,
)


def _faded_split(n_train: int = 384, n_val: int = 96, n_test: int = 96, seed: int = 0) -> Phase3DatasetSplit:
    """Learnable toy task: target = hidden local level, inputs = level + independent 4 dB noise per slot."""
    rng = np.random.default_rng(seed)

    def make(n: int) -> tuple[np.ndarray, np.ndarray]:
        level = rng.normal(-80.0, 6.0, size=(n, 1))
        x = np.zeros((n, 10, 3), dtype=np.float32)
        x[:, :, 0] = level + rng.normal(0.0, 4.0, size=(n, 10))
        x[:, :, 1] = rng.normal(-73.0, 1.0, size=(n, 10))
        x[:, :, 2] = x[:, :, 1] - x[:, :, 0]
        return x, level[:, 0].astype(np.float32)

    (xt, yt), (xv, yv), (xs, ys) = make(n_train), make(n_val), make(n_test)
    return Phase3DatasetSplit(xt, yt, xv, yv, xs, ys)


def _lstm_cfg(**overrides) -> LSTMTrainingConfig:
    kwargs = dict(hidden_size=64, max_epochs=3, early_stopping_patience=5, batch_size=64, device="cpu", seed=7)
    kwargs.update(overrides)
    return LSTMTrainingConfig(**kwargs)


def _transformer_cfg(**overrides) -> TransformerTrainingConfig:
    kwargs = dict(d_model=16, num_heads=2, num_layers=1, feedforward_dim=32, max_epochs=2,
                  early_stopping_patience=5, batch_size=64, device="cpu", seed=7)
    kwargs.update(overrides)
    return TransformerTrainingConfig(**kwargs)


def test_lstm_trains_scores_in_db_and_reports_both_gates() -> None:
    result = train_lstm_forecaster(_faded_split(), config=_lstm_cfg())

    assert result.model_name == "lstm"
    assert 1 <= result.best_epoch <= 3
    assert np.isfinite(result.best_validation_loss)
    db = result.test_db_metrics
    assert db is not None and result.validation_db_metrics is not None
    assert np.isfinite(db.mae_db) and db.mae_db > 0
    assert db.ar_mae_db is not None and db.moving_average_mae_db is not None  # all baselines present
    assert db.best_baseline in {"persistence", "moving_average_10", "linear_ar_10"}
    assert isinstance(result.meets_target, bool) and isinstance(result.meets_final_target, bool)
    assert result.test_target_min <= result.last_test_target <= result.test_target_max


def test_lstm_beats_persistence_on_a_learnable_task() -> None:
    result = train_lstm_forecaster(_faded_split(), config=_lstm_cfg(max_epochs=30, early_stopping_patience=10))

    db = result.test_db_metrics
    assert db.mae_db < db.persistence_mae_db  # learned to average out the per-slot noise
    assert db.skill_vs_persistence > 0.15
    assert result.meets_target


def test_checkpoint_is_written_and_reloadable(tmp_path) -> None:
    cfg = _lstm_cfg()
    split = _faded_split()
    checkpoint = tmp_path / "ckpt" / "lstm_best.pt"

    result = train_lstm_forecaster(split, config=cfg, checkpoint_path=checkpoint)

    assert result.checkpoint_path == str(checkpoint) and checkpoint.exists()
    model = _build_lstm_model(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    predictions = _predict(model, split.X_test[:10], batch_size=4, device="cpu")
    assert predictions.shape == (10,) and np.all(np.isfinite(predictions))


def test_same_seed_is_bit_reproducible_and_different_seed_differs() -> None:
    split = _faded_split()
    first = train_lstm_forecaster(split, config=_lstm_cfg(seed=11))
    second = train_lstm_forecaster(split, config=_lstm_cfg(seed=11))
    other = train_lstm_forecaster(split, config=_lstm_cfg(seed=12))

    assert first.test_db_metrics.mae_db == second.test_db_metrics.mae_db
    assert first.last_test_prediction == second.last_test_prediction
    assert first.best_validation_loss == second.best_validation_loss
    assert first.last_test_prediction != other.last_test_prediction


def test_transformer_trains_and_scores() -> None:
    result = train_transformer_forecaster(_faded_split(), config=_transformer_cfg())

    assert result.model_name == "transformer"
    assert np.isfinite(result.test_db_metrics.mae_db)
    assert result.test_db_metrics.best_baseline is not None


def test_transformer_is_reproducible() -> None:
    split = _faded_split()
    a = train_transformer_forecaster(split, config=_transformer_cfg(seed=3))
    b = train_transformer_forecaster(split, config=_transformer_cfg(seed=3))
    assert a.test_db_metrics.mae_db == b.test_db_metrics.mae_db


def test_real_prediction_stage_runs_transformer_only_when_lstm_misses_the_gate() -> None:
    outcome = run_prediction_stage(
        _faded_split(),
        lstm_config=_lstm_cfg(max_epochs=1),
        transformer_config=_transformer_cfg(max_epochs=1),
        min_skill=0.15,
    )

    assert outcome.transformer_executed == (not outcome.lstm.meets_target)
    assert (outcome.transformer is not None) == outcome.transformer_executed


def test_train_loader_shuffle_is_seeded() -> None:
    split = _faded_split(n_train=64)

    def first_batch(seed: int) -> np.ndarray:
        loader, _, _ = _make_regression_loaders(split, batch_size=16, seed=seed)
        return next(iter(loader))[1].numpy().reshape(-1)

    assert np.array_equal(first_batch(5), first_batch(5))
    assert not np.array_equal(first_batch(5), first_batch(6))


def test_model_output_shapes() -> None:
    x = torch.zeros(5, 10, 3)
    assert _build_lstm_model(_lstm_cfg())(x).shape == (5, 1)
    assert _build_transformer_model(_transformer_cfg())(x).shape == (5, 1)


def test_empty_validation_split_is_rejected() -> None:
    base = _faded_split()
    empty = Phase3DatasetSplit(
        base.X_train, base.y_train, base.X_val[:0], base.y_val[:0], base.X_test, base.y_test
    )
    with pytest.raises(ValueError, match="Validation split is empty"):
        train_lstm_forecaster(empty, config=_lstm_cfg())
