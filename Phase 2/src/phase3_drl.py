"""Phase 3 DRL environment and PPO training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_gymnasium() -> Any:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "gymnasium is required for Phase 3 DRL. Install Colab dependencies from "
            "requirements_phase3_colab.txt."
        ) from exc
    return gym


def _require_stable_baselines3() -> Any:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for PPO training. Install Colab dependencies from "
            "requirements_phase3_colab.txt."
        ) from exc
    return PPO


@dataclass(frozen=True, slots=True)
class DrlEnvConfig:
    """Configuration for the mitigation control environment."""

    episode_length: int = 64
    target_sinr_db: float = 15.0
    base_signal_db: float = -85.0
    base_interference_db: float = -95.0
    power_levels_db: tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0, 6.0)
    band_offsets_db: tuple[float, ...] = (0.0, 1.5, 3.0)
    interference_noise_std_db: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.episode_length <= 0:
            raise ValueError("episode_length must be > 0.")
        if not self.power_levels_db:
            raise ValueError("power_levels_db cannot be empty.")
        if not self.band_offsets_db:
            raise ValueError("band_offsets_db cannot be empty.")
        if self.interference_noise_std_db < 0:
            raise ValueError("interference_noise_std_db must be >= 0.")
        if self.seed < 0:
            raise ValueError("seed must be >= 0.")


@dataclass(frozen=True, slots=True)
class PPOTrainingConfig:
    """PPO controls for mitigation training."""

    total_timesteps: int = 20_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be > 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma must be in (0, 1].")
        if not (0 < self.gae_lambda <= 1):
            raise ValueError("gae_lambda must be in (0, 1].")
        if self.clip_range <= 0:
            raise ValueError("clip_range must be > 0.")
        if self.seed < 0:
            raise ValueError("seed must be >= 0.")


def assert_zf_validated(*, zf_suppression_db: float, minimum_required_db: float = 15.0) -> None:
    """Block DRL training unless ZF stage has passed minimum suppression."""
    if not np.isfinite(zf_suppression_db):
        raise ValueError("zf_suppression_db must be finite.")
    if zf_suppression_db < minimum_required_db:
        raise ValueError(
            f"ZF stage gate not met: suppression {zf_suppression_db:.2f} dB < required {minimum_required_db:.2f} dB."
        )


def create_interference_mitigation_env(config: DrlEnvConfig | None = None) -> Any:
    """Build a gymnasium-compatible control environment."""
    gym = _require_gymnasium()
    cfg = DrlEnvConfig() if config is None else config

    class InterferenceMitigationEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self._config = cfg
            self._rng = np.random.default_rng(cfg.seed)
            self.observation_space = gym.spaces.Box(
                low=np.array([-60.0, -150.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([60.0, 0.0, float(len(cfg.power_levels_db) - 1), float(len(cfg.band_offsets_db) - 1)], dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = gym.spaces.Discrete(4)

            self._step = 0
            self._power_index = len(cfg.power_levels_db) // 2
            self._band_index = 0
            self._predicted_interference_db = cfg.base_interference_db
            self._sinr_db = cfg.target_sinr_db - 5.0

        def _sample_predicted_interference(self) -> float:
            drift = self._rng.normal(loc=0.0, scale=1.0)
            return float(cfg.base_interference_db + drift)

        def _compute_sinr(self) -> float:
            realized_interference_db = (
                self._predicted_interference_db + self._rng.normal(loc=0.0, scale=cfg.interference_noise_std_db)
            )
            power_boost = cfg.power_levels_db[self._power_index]
            band_gain = cfg.band_offsets_db[self._band_index]
            return float(cfg.base_signal_db + power_boost + band_gain - realized_interference_db)

        def _observe(self) -> np.ndarray:
            return np.array(
                [self._sinr_db, self._predicted_interference_db, float(self._power_index), float(self._band_index)],
                dtype=np.float32,
            )

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self._step = 0
            self._power_index = len(cfg.power_levels_db) // 2
            self._band_index = 0
            self._predicted_interference_db = self._sample_predicted_interference()
            self._sinr_db = self._compute_sinr()
            return self._observe(), {}

        def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            previous_sinr = self._sinr_db
            if action == 1:
                self._power_index = min(self._power_index + 1, len(cfg.power_levels_db) - 1)
            elif action == 2:
                self._power_index = max(self._power_index - 1, 0)
            elif action == 3:
                self._band_index = (self._band_index + 1) % len(cfg.band_offsets_db)
            elif action != 0:
                raise ValueError(f"Unsupported action: {action}")

            self._predicted_interference_db = self._sample_predicted_interference()
            self._sinr_db = self._compute_sinr()
            self._step += 1

            reward = self._sinr_db - previous_sinr
            if self._sinr_db >= cfg.target_sinr_db:
                reward += 0.5
            terminated = self._step >= cfg.episode_length
            truncated = False
            info = {"target_sinr_db": cfg.target_sinr_db}
            return self._observe(), float(reward), terminated, truncated, info

    return InterferenceMitigationEnv()


def train_ppo_mitigator(
    *,
    zf_suppression_db: float,
    zf_required_db: float = 15.0,
    env_config: DrlEnvConfig | None = None,
    ppo_config: PPOTrainingConfig | None = None,
) -> Any:
    """Train PPO agent after ZF gate passes."""
    assert_zf_validated(zf_suppression_db=zf_suppression_db, minimum_required_db=zf_required_db)
    PPO = _require_stable_baselines3()

    env = create_interference_mitigation_env(env_config)
    cfg = PPOTrainingConfig() if ppo_config is None else ppo_config
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        verbose=0,
        seed=cfg.seed,
    )
    model.learn(total_timesteps=cfg.total_timesteps)
    return model
