"""Reward functions for Clankers training tasks.

All reward computation happens in Python during training. Rust provides
observations and state; Python computes rewards locally.

Reward functions operate on numpy arrays (observations, actions) and return
scalar float rewards. They can be composed via ``CompositeReward``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class RewardFunction(ABC):
    """Base class for all reward functions.

    Subclasses must implement :meth:`compute` and :attr:`name`.
    """

    @abstractmethod
    def compute(
        self,
        obs: NDArray[np.float32],
        action: NDArray[np.float32] | int | None = None,
        next_obs: NDArray[np.float32] | None = None,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Compute the reward for one step.

        Parameters
        ----------
        obs : np.ndarray
            Current observation.
        action : np.ndarray | int | None
            Action taken (may be None for obs-only rewards).
        next_obs : np.ndarray | None
            Next observation after action (may be None).
        info : dict | None
            Step info dict from the environment.

        Returns
        -------
        float
            Scalar reward value.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this reward function."""


class DistanceReward(RewardFunction):
    """Reward based on L2 distance between two positions in the observation.

    Returns ``-distance`` so that maximizing reward minimizes distance.
    Positions are extracted from the observation vector at the given indices.

    Parameters
    ----------
    pos_a_indices : list[int]
        Indices into the observation vector for entity A's position.
    pos_b_indices : list[int]
        Indices into the observation vector for entity B's position.
    """

    def __init__(self, pos_a_indices: list[int], pos_b_indices: list[int]) -> None:
        if len(pos_a_indices) != len(pos_b_indices):
            raise ValueError(
                f"Position index lengths must match: {len(pos_a_indices)} != {len(pos_b_indices)}"
            )
        self._pos_a = pos_a_indices
        self._pos_b = pos_b_indices

    def compute(
        self,
        obs: NDArray[np.float32],
        action: NDArray[np.float32] | int | None = None,
        next_obs: NDArray[np.float32] | None = None,
        info: dict[str, Any] | None = None,
    ) -> float:
        pos_a = obs[self._pos_a]
        pos_b = obs[self._pos_b]
        dist = float(np.linalg.norm(pos_a - pos_b))
        return -dist

    @property
    def name(self) -> str:
        return "DistanceReward"


class SparseReward(RewardFunction):
    """Binary reward: 1.0 when distance is below threshold, 0.0 otherwise.

    Parameters
    ----------
    pos_a_indices : list[int]
        Indices for entity A's position.
    pos_b_indices : list[int]
        Indices for entity B's position.
    threshold : float
        Distance threshold for success.
    """

    def __init__(
        self,
        pos_a_indices: list[int],
        pos_b_indices: list[int],
        threshold: float,
    ) -> None:
        if len(pos_a_indices) != len(pos_b_indices):
            raise ValueError(
                f"Position index lengths must match: {len(pos_a_indices)} != {len(pos_b_indices)}"
            )
        self._pos_a = pos_a_indices
        self._pos_b = pos_b_indices
        self._threshold = threshold

    def compute(
        self,
        obs: NDArray[np.float32],
        action: NDArray[np.float32] | int | None = None,
        next_obs: NDArray[np.float32] | None = None,
        info: dict[str, Any] | None = None,
    ) -> float:
        pos_a = obs[self._pos_a]
        pos_b = obs[self._pos_b]
        dist = float(np.linalg.norm(pos_a - pos_b))
        return 1.0 if dist < self._threshold else 0.0

    @property
    def name(self) -> str:
        return "SparseReward"


class ConstantReward(RewardFunction):
    """Returns a constant reward each step.

    Standard reward for CartPole-v1: +1 per step the pole is upright.

    Parameters
    ----------
    value : float
        Constant reward value (default: 1.0).
    """

    def __init__(self, value: float = 1.0) -> None:
        self._value = value

    def compute(
        self,
        obs: NDArray[np.float32],
        action: NDArray[np.float32] | int | None = None,
        next_obs: NDArray[np.float32] | None = None,
        info: dict[str, Any] | None = None,
    ) -> float:
        return self._value

    @property
    def name(self) -> str:
        return "ConstantReward"


class ActionPenaltyReward(RewardFunction):
    """Penalty proportional to the L2 norm squared of the action.

    Returns ``-scale * ||action||^2``, encouraging minimal control effort.

    Parameters
    ----------
    scale : float
        Penalty scale factor.
    """

    def __init__(self, scale: float = 0.01) -> None:
        self._scale = scale

    def compute(
        self,
        obs: NDArray[np.float32],
        action: NDArray[np.float32] | int | None = None,
        next_obs: NDArray[np.float32] | None = None,
        info: dict[str, Any] | None = None,
    ) -> float:
        if action is None:
            return 0.0
        if isinstance(action, (int, np.integer)):
            return 0.0
        a = np.asarray(action, dtype=np.float32)
        norm_sq = float(np.sum(a * a))
        return -self._scale * norm_sq

    @property
    def name(self) -> str:
        return "ActionPenaltyReward"


class CompositeReward(RewardFunction):
    """Weighted combination of multiple reward functions.

    The total reward is the sum of each component reward multiplied by
    its weight.

    Parameters
    ----------
    rewards : list[tuple[RewardFunction, float]] | None
        Initial list of (reward_fn, weight) pairs.
    """

    def __init__(
        self,
        rewards: list[tuple[RewardFunction, float]] | None = None,
    ) -> None:
        self._rewards: list[tuple[RewardFunction, float]] = rewards or []

    def add(self, reward: RewardFunction, weight: float = 1.0) -> CompositeReward:
        """Add a reward function with a weight. Returns self for chaining."""
        self._rewards.append((reward, weight))
        return self

    def compute(
        self,
        obs: NDArray[np.float32],
        action: NDArray[np.float32] | int | None = None,
        next_obs: NDArray[np.float32] | None = None,
        info: dict[str, Any] | None = None,
    ) -> float:
        total = 0.0
        for reward_fn, weight in self._rewards:
            total += reward_fn.compute(obs, action, next_obs, info) * weight
        return total

    def breakdown(
        self,
        obs: NDArray[np.float32],
        action: NDArray[np.float32] | int | None = None,
        next_obs: NDArray[np.float32] | None = None,
        info: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Compute each component and return (name, weighted_value) pairs."""
        return [(r.name, r.compute(obs, action, next_obs, info) * w) for r, w in self._rewards]

    @property
    def name(self) -> str:
        return "CompositeReward"
