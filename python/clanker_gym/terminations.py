"""Termination conditions for Clankers training tasks.

All task-level termination logic lives in Python during training.
Rust provides the ``terminated`` and ``truncated`` flags for episode
lifecycle (timeout/truncation), but success/failure detection based on
observation state is computed here.

Termination functions operate on numpy arrays (observations) and return
a boolean. They can be composed via ``CompositeTermination``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class TerminationFn(ABC):
    """Base class for all termination conditions.

    Subclasses must implement :meth:`is_terminated` and :attr:`name`.
    """

    @abstractmethod
    def is_terminated(
        self,
        obs: NDArray[np.float32],
        step_count: int = 0,
        info: dict[str, Any] | None = None,
    ) -> bool:
        """Check whether the episode should terminate.

        Parameters
        ----------
        obs : np.ndarray
            Current observation.
        step_count : int
            Number of steps taken in this episode.
        info : dict | None
            Step info dict from the environment.

        Returns
        -------
        bool
            True if the episode should terminate.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this termination condition."""


class SuccessTermination(TerminationFn):
    """Terminates when two positions in the observation are within a threshold.

    Parameters
    ----------
    pos_a_indices : list[int]
        Indices into the observation vector for entity A's position.
    pos_b_indices : list[int]
        Indices into the observation vector for entity B's position.
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

    def is_terminated(
        self,
        obs: NDArray[np.float32],
        step_count: int = 0,
        info: dict[str, Any] | None = None,
    ) -> bool:
        pos_a = obs[self._pos_a]
        pos_b = obs[self._pos_b]
        dist = float(np.linalg.norm(pos_a - pos_b))
        return dist < self._threshold

    @property
    def name(self) -> str:
        return "SuccessTermination"


class TimeoutTermination(TerminationFn):
    """Terminates when the step count exceeds a maximum.

    Parameters
    ----------
    max_steps : int
        Maximum number of steps before termination.
    """

    def __init__(self, max_steps: int) -> None:
        self._max_steps = max_steps

    def is_terminated(
        self,
        obs: NDArray[np.float32],
        step_count: int = 0,
        info: dict[str, Any] | None = None,
    ) -> bool:
        return step_count >= self._max_steps

    @property
    def name(self) -> str:
        return "TimeoutTermination"


class FailureTermination(TerminationFn):
    """Terminates when a value at a given observation index falls below a threshold.

    Useful for detecting when a robot has fallen (e.g., height drops below
    minimum).

    Parameters
    ----------
    obs_index : int
        Index into the observation vector to check.
    min_value : float
        Minimum allowed value. Terminates if obs[index] < min_value.
    """

    def __init__(self, obs_index: int, min_value: float) -> None:
        self._obs_index = obs_index
        self._min_value = min_value

    def is_terminated(
        self,
        obs: NDArray[np.float32],
        step_count: int = 0,
        info: dict[str, Any] | None = None,
    ) -> bool:
        return float(obs[self._obs_index]) < self._min_value

    @property
    def name(self) -> str:
        return "FailureTermination"


class BoundsTermination(TerminationFn):
    """Terminates when an observation value exceeds symmetric bounds.

    Checks ``abs(obs[index]) > threshold``.

    Parameters
    ----------
    obs_index : int
        Index into the observation vector to check.
    threshold : float
        Symmetric bound. Terminates if ``abs(obs[index]) > threshold``.
    label : str
        Name for this termination condition.
    """

    def __init__(self, obs_index: int, threshold: float, label: str = "BoundsTermination") -> None:
        self._obs_index = obs_index
        self._threshold = threshold
        self._label = label

    def is_terminated(
        self,
        obs: NDArray[np.float32],
        step_count: int = 0,
        info: dict[str, Any] | None = None,
    ) -> bool:
        return bool(abs(float(obs[self._obs_index])) > self._threshold)

    @property
    def name(self) -> str:
        return self._label


def cartpole_termination(
    angle_threshold: float = 0.2094,
    position_threshold: float = 2.4,
    angle_index: int = 2,
    position_index: int = 0,
) -> CompositeTermination:
    """Standard CartPole-v1 termination conditions.

    Terminates when:
    - Pole angle exceeds ±12 degrees (0.2094 rad)
    - Cart position exceeds ±2.4 meters

    Parameters
    ----------
    angle_threshold : float
        Max pole angle in radians (default: 0.2094 = 12 degrees).
    position_threshold : float
        Max cart position in meters (default: 2.4).
    angle_index : int
        Index of pole angle in observation vector (default: 2).
    position_index : int
        Index of cart position in observation vector (default: 0).
    """
    return CompositeTermination([
        BoundsTermination(angle_index, angle_threshold, "PoleAngleTermination"),
        BoundsTermination(position_index, position_threshold, "CartPositionTermination"),
    ])


class CompositeTermination(TerminationFn):
    """OR-composition of multiple termination conditions.

    Returns True if **any** contained condition is satisfied.

    Parameters
    ----------
    conditions : list[TerminationFn] | None
        Initial list of termination conditions.
    """

    def __init__(self, conditions: list[TerminationFn] | None = None) -> None:
        self._conditions: list[TerminationFn] = conditions or []

    def add(self, condition: TerminationFn) -> CompositeTermination:
        """Add a termination condition. Returns self for chaining."""
        self._conditions.append(condition)
        return self

    def is_terminated(
        self,
        obs: NDArray[np.float32],
        step_count: int = 0,
        info: dict[str, Any] | None = None,
    ) -> bool:
        return any(c.is_terminated(obs, step_count, info) for c in self._conditions)

    @property
    def name(self) -> str:
        return "CompositeTermination"
