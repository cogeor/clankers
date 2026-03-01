"""Observation wrappers for Clanker gymnasium environments.

Provides wrappers that work at the single-env level, independent of the
vectorised-env transport (TCP, shared-memory, etc.).  This avoids the
coupling to ``VecNormalize`` that SB3 expects from its own ``VecEnv``.

Requires the ``sb3`` extra: ``pip install clanker-gym[sb3]``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

try:
    import gymnasium
    from gymnasium import spaces as gym_spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required for clanker_gym wrappers. "
        "Install with: pip install clanker-gym[sb3]"
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
        self._mean = np.zeros(env.observation_space.shape, dtype=np.float64)
        self._var = np.ones(env.observation_space.shape, dtype=np.float64)
        self._welford_m2 = np.zeros(env.observation_space.shape, dtype=np.float64)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Update running stats and return the normalized observation."""
        self._update_stats(obs)
        return self._normalize(obs)

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

        low = np.repeat(env.observation_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], n_frames, axis=0)
        self.observation_space = gym_spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs: object) -> tuple[np.ndarray, dict]:
        """Reset the environment and fill the frame buffer with the initial obs."""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self._frames.append(obs)
        return self._get_obs(), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Append the new observation and return the stacked frames."""
        self._frames.append(obs)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Return stacked frames as a single array."""
        return np.stack(list(self._frames), axis=0)
