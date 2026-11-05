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
    assert set(state) == {"git_sha", "git_dirty", "git_dirty_files"}
    if state["git_dirty_files"] is not None:
        assert state["git_dirty"] == bool(state["git_dirty_files"])
    if state["git_sha"] is not None:  # None only when git is unavailable
        assert len(state["git_sha"]) == 40
    assert not any('"' in path for path in (state["git_dirty_files"] or []))  # no stray quote characters


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


# --- git_dirty_files must hold clean paths (no stray quote characters) ---------------------------

import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

from src.reproducibility import _parse_nul_separated_paths  # noqa: E402

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def test_parse_nul_separated_paths_keeps_paths_verbatim() -> None:
    raw = "Phase 2/run stage b.py\0src/x.py\0Phase 2/a b/c d.txt\0"

    assert _parse_nul_separated_paths(raw) == ["Phase 2/a b/c d.txt", "Phase 2/run stage b.py", "src/x.py"]
    assert _parse_nul_separated_paths("") == []
    assert _parse_nul_separated_paths("\0\0") == []
    assert _parse_nul_separated_paths('he said "hi".txt\0') == ['he said "hi".txt']  # nothing is stripped or unquoted


def _isolated_git(repo, *args):
    """Run git in a throwaway repo without touching anyone's git config (config isolated, identity via env)."""
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
    subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)


@needs_git
def test_git_state_paths_with_spaces_have_no_stray_quotes(tmp_path) -> None:
    repo = tmp_path / "repo"
    (repo / "Phase 2").mkdir(parents=True)
    (repo / "Phase 2" / "my file.py").write_text("one\n")
    (repo / "Phase 2" / "second.py").write_text("one\n")
    (repo / "plain.py").write_text("one\n")
    _isolated_git(repo, "init", "-q")
    _isolated_git(repo, "add", ".")
    _isolated_git(repo, "commit", "-q", "-m", "init")

    clean = git_state(repo)
    assert clean["git_dirty"] is False and clean["git_dirty_files"] == [] and len(clean["git_sha"]) == 40

    (repo / "Phase 2" / "my file.py").write_text("two\n")  # unstaged change, path with a space
    (repo / "plain.py").write_text("two\n")
    (repo / "Phase 2" / "second.py").write_text("two\n")
    _isolated_git(repo, "add", "Phase 2/second.py")  # staged change
    (repo / "untracked file.py").write_text("x\n")  # untracked: not counted

    dirty = git_state(repo)
    assert dirty["git_dirty"] is True
    assert dirty["git_dirty_files"] == ["Phase 2/my file.py", "Phase 2/second.py", "plain.py"]
    assert not any('"' in path for path in dirty["git_dirty_files"])
    assert dirty["git_dirty_files"][0].startswith("Phase 2/")  # the first entry is no longer missing a character


@needs_git
def test_git_state_in_a_directory_that_is_not_a_repository(tmp_path) -> None:
    state = git_state(tmp_path)

    assert state == {"git_sha": None, "git_dirty": None, "git_dirty_files": None}


def test_run_info_carries_the_clean_dirty_file_list(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.reproducibility.git_state",
        lambda *a, **k: {"git_sha": "a" * 40, "git_dirty": True, "git_dirty_files": ["Phase 2/x y.py"]},
    )

    info = collect_run_info(1)

    assert info["git_dirty_files"] == ["Phase 2/x y.py"] and info["git_dirty"] is True
