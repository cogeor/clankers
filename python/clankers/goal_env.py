"""Goal-conditioned environment wrapper for HER-compatible RL.

Wraps a ClankerGymnasiumEnv and restructures observations into the
Gymnasium-Robotics GoalEnv format: {"observation", "achieved_goal", "desired_goal"}.

The compute_reward() method can be called externally with arbitrary
achieved/desired goals, enabling Hindsight Experience Replay (HER).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium
    from gymnasium import spaces as gym_spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium required for ClankerGoalEnv. pip install clankers[sb3]"
    ) from exc

from clankers.gymnasium_env import ClankerGymnasiumEnv


class GoalRewardFn(ABC):
    """Base class for goal-conditioned reward functions.

    Unlike RewardFunction which operates on flat obs, this operates on
    achieved_goal and desired_goal arrays -- enabling HER relabeling.
    """

    @abstractmethod
    def compute_reward(
        self,
        achieved_goal: NDArray[np.float32],
        desired_goal: NDArray[np.float32],
        info: dict[str, Any],
    ) -> float:
        """Compute reward from achieved and desired goals."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""


class SparseGoalReward(GoalRewardFn):
    """Sparse reward: 0.0 if L2(achieved - desired) < threshold, -1.0 otherwise.

    Matches Gymnasium-Robotics convention (negative sparse reward).
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self._threshold = threshold

    def compute_reward(
        self,
        achieved_goal: NDArray[np.float32],
        desired_goal: NDArray[np.float32],
        info: dict[str, Any],
    ) -> float:
        dist = float(np.linalg.norm(achieved_goal - desired_goal))
        return 0.0 if dist < self._threshold else -1.0

    @property
    def name(self) -> str:
        return "SparseGoalReward"


class DenseGoalReward(GoalRewardFn):
    """Dense reward: -L2(achieved - desired).

    Maximizing this reward minimizes distance to goal.
    """

    def compute_reward(
        self,
        achieved_goal: NDArray[np.float32],
        desired_goal: NDArray[np.float32],
        info: dict[str, Any],
    ) -> float:
        return -float(np.linalg.norm(achieved_goal - desired_goal))

    @property
    def name(self) -> str:
        return "DenseGoalReward"


class ClankerGoalEnv(gymnasium.Env):  # type: ignore[misc]
    """Goal-conditioned environment wrapper for HER-compatible RL.

    Wraps a ClankerGymnasiumEnv and provides:
    - Dict observation space: {observation, achieved_goal, desired_goal}
    - External compute_reward(achieved, desired, info) for HER relabeling
    - is_success in info dict

    Parameters
    ----------
    env : ClankerGymnasiumEnv
        The base environment to wrap.
    obs_indices : list[int] or slice
        Indices into the flat obs for the proprioceptive observation.
    achieved_goal_indices : list[int] or slice
        Indices into the flat obs for the achieved goal (e.g., EE position).
    goal_dim : int
        Dimensionality of the goal space.
    goal_reward_fn : GoalRewardFn or None
        Reward function operating on (achieved, desired). Defaults to
        SparseGoalReward with the given success_threshold.
    success_threshold : float
        Distance threshold for is_success (default 0.05).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        env: ClankerGymnasiumEnv,
        obs_indices: list[int] | slice,
        achieved_goal_indices: list[int] | slice,
        goal_dim: int,
        goal_reward_fn: GoalRewardFn | None = None,
        success_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self._env = env
        self._obs_indices = obs_indices
        self._achieved_goal_indices = achieved_goal_indices
        self._goal_dim = goal_dim
        self._reward_fn = goal_reward_fn or SparseGoalReward(success_threshold)
        self._success_threshold = success_threshold
        self._desired_goal = np.zeros(goal_dim, dtype=np.float32)

        # Build observation space as Dict
        base_space = env.observation_space
        assert isinstance(base_space, gym_spaces.Box)

        obs_low = base_space.low[obs_indices]
        obs_high = base_space.high[obs_indices]
        goal_low = base_space.low[achieved_goal_indices]
        goal_high = base_space.high[achieved_goal_indices]

        self.observation_space = gym_spaces.Dict(
            {
                "observation": gym_spaces.Box(obs_low, obs_high, dtype=np.float32),
                "achieved_goal": gym_spaces.Box(goal_low, goal_high, dtype=np.float32),
                "desired_goal": gym_spaces.Box(goal_low, goal_high, dtype=np.float32),
            }
        )
        self.action_space = env.action_space

    def set_goal(self, goal: NDArray[np.float32]) -> None:
        """Set the desired goal for the current episode."""
        self._desired_goal = np.asarray(goal, dtype=np.float32)

    def compute_reward(
        self,
        achieved_goal: NDArray[np.float32],
        desired_goal: NDArray[np.float32],
        info: dict[str, Any],
    ) -> float:
        """Compute reward -- callable externally for HER relabeling."""
        return self._reward_fn.compute_reward(achieved_goal, desired_goal, info)

    def _get_obs_dict(
        self, flat_obs: NDArray[np.float32]
    ) -> dict[str, NDArray[np.float32]]:
        return {
            "observation": flat_obs[self._obs_indices].copy(),
            "achieved_goal": flat_obs[self._achieved_goal_indices].copy(),
            "desired_goal": self._desired_goal.copy(),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        """Reset the environment.

        Returns
        -------
        observation : dict
            Dict with keys "observation", "achieved_goal", "desired_goal".
        info : dict
        """
        flat_obs, info = self._env.reset(seed=seed, options=options)
        obs_dict = self._get_obs_dict(flat_obs)
        return obs_dict, info

    def step(
        self,
        action: NDArray[np.float32] | int,
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        """Take one step.

        Returns
        -------
        observation : dict
            Dict with keys "observation", "achieved_goal", "desired_goal".
        reward : float
        terminated : bool
        truncated : bool
        info : dict
            Includes "is_success" key.
        """
        flat_obs, _, terminated, truncated, info = self._env.step(action)
        obs_dict = self._get_obs_dict(flat_obs)

        achieved = obs_dict["achieved_goal"]
        reward = self.compute_reward(achieved, self._desired_goal, info)

        dist = float(np.linalg.norm(achieved - self._desired_goal))
        info["is_success"] = dist < self._success_threshold

        return obs_dict, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the wrapped environment."""
        self._env.close()

    def render(self) -> None:
        """Delegate rendering to the wrapped environment."""
        self._env.render()
