"""Tests for the DRL environment, configs and the ZF gate in front of PPO (audit M2)."""

import numpy as np
import pytest

from src.phase3_drl import (
    DrlEnvConfig,
    PPOTrainingConfig,
    assert_zf_validated,
    create_interference_mitigation_env,
    train_ppo_mitigator,
)


def test_assert_zf_validated_gate() -> None:
    assert_zf_validated(zf_suppression_db=15.0)  # boundary passes
    with pytest.raises(ValueError, match="ZF stage gate not met"):
        assert_zf_validated(zf_suppression_db=14.99)
    with pytest.raises(ValueError, match="finite"):
        assert_zf_validated(zf_suppression_db=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        assert_zf_validated(zf_suppression_db=float("nan"))


def test_ppo_training_is_blocked_before_importing_anything_when_zf_fails() -> None:
    with pytest.raises(ValueError, match="ZF stage gate not met"):
        train_ppo_mitigator(zf_suppression_db=3.0)


def test_drl_config_validation() -> None:
    with pytest.raises(ValueError):
        DrlEnvConfig(episode_length=0)
    with pytest.raises(ValueError):
        DrlEnvConfig(power_levels_db=())
    with pytest.raises(ValueError):
        DrlEnvConfig(seed=-1)
    with pytest.raises(ValueError):
        PPOTrainingConfig(total_timesteps=0)
    with pytest.raises(ValueError):
        PPOTrainingConfig(gamma=0.0)
    with pytest.raises(ValueError):
        PPOTrainingConfig(clip_range=0.0)


def test_env_reset_step_contract_and_termination() -> None:
    pytest.importorskip("gymnasium", reason="gymnasium not installed: DRL environment is NOT exercised")
    env = create_interference_mitigation_env(DrlEnvConfig(episode_length=5, seed=1))

    observation, info = env.reset(seed=1)
    assert observation.shape == (4,) and observation.dtype == np.float32 and info == {}
    assert env.observation_space.contains(observation)

    terminated = False
    for step in range(5):
        observation, reward, terminated, truncated, info = env.step(0)
        assert np.isfinite(reward) and not truncated and info["target_sinr_db"] == 15.0
        assert terminated == (step == 4)
    assert terminated


def test_env_actions_move_power_and_band_indices_within_bounds() -> None:
    pytest.importorskip("gymnasium", reason="gymnasium not installed: DRL environment is NOT exercised")
    cfg = DrlEnvConfig(seed=2)
    env = create_interference_mitigation_env(cfg)
    env.reset(seed=2)
    start_power = env.unwrapped._power_index

    env.step(1)
    assert env.unwrapped._power_index == min(start_power + 1, len(cfg.power_levels_db) - 1)
    for _ in range(10):
        env.step(2)
    assert env.unwrapped._power_index == 0  # clamped at the lowest level
    for _ in range(10):
        env.step(1)
    assert env.unwrapped._power_index == len(cfg.power_levels_db) - 1  # clamped at the highest level
    band_before = env.unwrapped._band_index
    env.step(3)
    assert env.unwrapped._band_index == (band_before + 1) % len(cfg.band_offsets_db)
    with pytest.raises(ValueError, match="Unsupported action"):
        env.step(9)


def test_env_is_deterministic_for_a_seed() -> None:
    pytest.importorskip("gymnasium", reason="gymnasium not installed: DRL environment is NOT exercised")

    def rollout() -> list[float]:
        env = create_interference_mitigation_env(DrlEnvConfig(episode_length=8, seed=4))
        env.reset(seed=4)
        return [env.step(i % 4)[1] for i in range(8)]

    assert rollout() == rollout()


def test_ppo_smoke_training_after_zf_gate_passes() -> None:
    pytest.importorskip("stable_baselines3", reason="stable-baselines3 not installed: PPO is NOT exercised")
    model = train_ppo_mitigator(
        zf_suppression_db=20.0,
        env_config=DrlEnvConfig(episode_length=16, seed=0),
        ppo_config=PPOTrainingConfig(total_timesteps=128, n_steps=64, batch_size=32, seed=0),
    )

    env = create_interference_mitigation_env(DrlEnvConfig(episode_length=16, seed=0))
    observation, _ = env.reset(seed=0)
    action, _ = model.predict(observation, deterministic=True)
    assert int(action) in range(4)
