"""Tests for seeding and run provenance (audit M3)."""

import json
import random

import numpy as np
import pytest

from run_phase3_colab import build_argument_parser, build_training_configs
from src.phase3_prediction import LSTMTrainingConfig, TransformerTrainingConfig
from src.reproducibility import collect_run_info, git_state, seed_everything


def test_seed_everything_makes_python_and_numpy_deterministic() -> None:
    seed_everything(123)
    first = (random.random(), float(np.random.rand()))
    seed_everything(123)
    second = (random.random(), float(np.random.rand()))
    seed_everything(124)
    third = (random.random(), float(np.random.rand()))

    assert first == second
    assert first != third


def test_seed_everything_seeds_torch_when_installed() -> None:
    torch = pytest.importorskip("torch", reason="PyTorch not installed")
    seed_everything(5)
    a = torch.rand(3)
    seed_everything(5)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_seed_everything_validates_input() -> None:
    with pytest.raises(ValueError):
        seed_everything(-1)
    with pytest.raises(TypeError):
        seed_everything(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        seed_everything(True)  # type: ignore[arg-type]


def test_collect_run_info_is_json_safe_and_records_seed_and_versions() -> None:
    info = collect_run_info(42, extra={"data_root": "x"})

    json.dumps(info)  # must be serialisable
    assert info["seed"] == 42 and info["data_root"] == "x"
    assert info["numpy"] == np.__version__
    assert info["python"].count(".") >= 1
    assert {"torch", "scipy", "platform", "generated_utc", "git_sha", "git_dirty", "cuda_available"} <= info.keys()


def test_git_state_reports_sha_in_this_repo() -> None:
    state = git_state()
    assert set(state) == {"git_sha", "git_dirty"}
    if state["git_sha"] is not None:  # None only when git is unavailable
        assert len(state["git_sha"]) == 40


def test_training_configs_validate_seed() -> None:
    with pytest.raises(ValueError):
        LSTMTrainingConfig(seed=-1)
    with pytest.raises(ValueError):
        TransformerTrainingConfig(seed=-1)
    assert LSTMTrainingConfig().seed == 42 and TransformerTrainingConfig().seed == 42


def test_cli_seed_reaches_both_model_configs() -> None:
    args = build_argument_parser().parse_args(["--data-root", "x", "--seed", "123"])
    lstm_cfg, transformer_cfg = build_training_configs(args)

    assert lstm_cfg.seed == 123 and transformer_cfg.seed == 123
