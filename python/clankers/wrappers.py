"""Observation wrappers for Clanker gymnasium environments.

Provides wrappers that work at the single-env level, independent of the
vectorised-env transport (TCP, shared-memory, etc.).  This avoids the
coupling to ``VecNormalize`` that SB3 expects from its own ``VecEnv``.

Requires the ``sb3`` extra: ``pip install clankers[sb3]``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

try:
    import gymnasium
    from gymnasium import spaces as gym_spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required for clankers wrappers. Install with: pip install clankers[sb3]"
    ) from exc


class NormalizeObservation(gymnasium.ObservationWrapper):  # type: ignore[misc]
    """Normalize observations using running mean and variance.

    Uses Welford's online algorithm to track running statistics.
    Normalizes: ``obs = (obs - mean) / sqrt(var + epsilon)``

    Parameters
    ----------
    env : gymnasium.Env
        Environment to wrap.
    epsilon : float
        Small constant for numerical stability (default: 1e-8).
    clip : float
        Clip normalized obs to ``[-clip, clip]`` (default: 10.0).
    """

    def __init__(self, env: gymnasium.Env, epsilon: float = 1e-8, clip: float = 10.0) -> None:
        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip
        self._count: int = 0
        shape = env.observation_space.shape
        assert shape is not None, "observation_space must have a shape"
        self._mean = np.zeros(shape, dtype=np.float64)
        self._var = np.ones(shape, dtype=np.float64)
        self._welford_m2 = np.zeros(shape, dtype=np.float64)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Update running stats and return the normalized observation."""
        self._update_stats(observation)
        return self._normalize(observation)

    def _update_stats(self, obs: np.ndarray) -> None:
        """Update running mean/variance via Welford's online algorithm."""
        self._count += 1
        delta = obs - self._mean
        self._mean += delta / self._count
        delta2 = obs - self._mean
        self._welford_m2 += delta * delta2
        self._var = self._welford_m2 / max(self._count, 1)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize using current running stats."""
        normalized = (obs - self._mean) / np.sqrt(self._var + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)


class FrameStack(gymnasium.ObservationWrapper):  # type: ignore[misc]
    """Stack the last N observations along a new first axis.

    Useful for:

    * Image-based RL (stack 4 frames for motion information).
    * Partial observability (velocity estimation from position history).

    The wrapped ``observation_space`` gains an extra leading dimension of
    size ``n_frames``.  For example a ``Box(shape=(4,))`` with
    ``n_frames=3`` becomes ``Box(shape=(3, 4))``.

    Parameters
    ----------
    env : gymnasium.Env
        Environment to wrap.
    n_frames : int
        Number of frames to stack (default: 4).
    """

    def __init__(self, env: gymnasium.Env, n_frames: int = 4) -> None:
        super().__init__(env)
        self.n_frames = n_frames
        self._frames: deque[np.ndarray] = deque(maxlen=n_frames)

        obs_space = env.observation_space
        assert isinstance(obs_space, gym_spaces.Box), "FrameStack requires Box obs space"
        low = np.repeat(obs_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(obs_space.high[np.newaxis, ...], n_frames, axis=0)
        self.observation_space = gym_spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def reset(self, **kwargs: object) -> tuple[np.ndarray, dict]:
        """Reset the environment and fill the frame buffer with the initial obs."""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self._frames.append(obs)
        return self._get_obs(), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Append the new observation and return the stacked frames."""
        self._frames.append(observation)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Return stacked frames as a single array."""
        return np.stack(list(self._frames), axis=0)


class NormalizeReward(gymnasium.RewardWrapper):  # type: ignore[misc]
    """Normalize rewards using running variance estimate.

    Uses discounted return variance estimation:

    * Tracks running variance of discounted returns via Welford's algorithm.
    * Normalizes: ``reward = reward / sqrt(var + epsilon)``
    * Clips to ``[-clip, clip]``

    Parameters
    ----------
    env : gymnasium.Env
        Environment to wrap.
    gamma : float
        Discount factor for return estimation (default: 0.99).
    epsilon : float
        Small constant for numerical stability (default: 1e-8).
    clip : float
        Clip normalized reward to ``[-clip, clip]`` (default: 10.0).
    """

    def __init__(
        self,
        env: gymnasium.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self._return: float = 0.0  # running discounted return
        self._count: int = 0
        self._mean: float = 0.0
        self._var: float = 1.0
        self._welford_m2: float = 0.0

    def reward(self, reward: float) -> float:  # type: ignore[override]
        """Update running stats and return the normalized reward."""
        # Update discounted return estimate.
        self._return = self._return * self.gamma + reward
        # Update running variance via Welford on the return.
        self._count += 1
        delta = self._return - self._mean
        self._mean += delta / self._count
        delta2 = self._return - self._mean
        self._welford_m2 += delta * delta2
        self._var = self._welford_m2 / max(self._count, 1)
        # Normalize the reward (not the return) by return std.
        normalized = reward / (np.sqrt(self._var + self.epsilon))
        return float(np.clip(normalized, -self.clip, self.clip))

    def reset(self, **kwargs: object) -> tuple[np.ndarray, dict]:
        """Reset the environment and clear the discounted return."""
        self._return = 0.0  # reset discounted return on episode boundary
        return self.env.reset(**kwargs)


class ClipReward(gymnasium.RewardWrapper):  # type: ignore[misc]
    """Clip rewards to ``[min_reward, max_reward]``.

    Parameters
    ----------
    env : gymnasium.Env
        Environment to wrap.
    min_reward : float
        Minimum reward (default: -10.0).
    max_reward : float
        Maximum reward (default: 10.0).
    """

    def __init__(
        self,
        env: gymnasium.Env,
        min_reward: float = -10.0,
        max_reward: float = 10.0,
    ) -> None:
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, reward: float) -> float:  # type: ignore[override]
        """Return the reward clipped to the configured range."""
        return float(np.clip(reward, self.min_reward, self.max_reward))
