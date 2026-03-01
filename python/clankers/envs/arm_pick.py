"""Arm pick-and-place task preset.

Creates a ClankerGymnasiumEnv configured for an arm pick-and-place task
with composite distance rewards and success termination.
"""

from __future__ import annotations

from typing import Any

from clankers.gymnasium_env import ClankerGymnasiumEnv
from clankers.rewards import ActionPenaltyReward, CompositeReward, DistanceReward
from clankers.terminations import CompositeTermination, SuccessTermination, TimeoutTermination


def make_arm_pick_env(
    host: str = "127.0.0.1",
    port: int = 9879,
    object_indices: list[int] | None = None,
    goal_indices: list[int] | None = None,
    ee_indices: list[int] | None = None,
    success_threshold: float = 0.05,
    max_steps: int = 500,
    action_penalty: float = 0.01,
    **kwargs: Any,
) -> ClankerGymnasiumEnv:
    """Create an arm pick-and-place environment.

    Reward is composite of:
    - Distance: object to goal (weight 1.0) -- primary: move object to target
    - Distance: EE to object (weight 0.5) -- secondary: reach toward object
    - Action penalty (weight action_penalty)

    Terminates on:
    - Success: object within success_threshold of goal
    - Timeout: max_steps exceeded

    Parameters
    ----------
    host : str
        Server address (default: 127.0.0.1).
    port : int
        Server port (default: 9879).
    object_indices : list[int] | None
        Observation indices for object position (default: [3, 4, 5]).
    goal_indices : list[int] | None
        Observation indices for goal position (default: [6, 7, 8]).
    ee_indices : list[int] | None
        Observation indices for end-effector position (default: [0, 1, 2]).
    success_threshold : float
        Distance for success (default: 0.05m).
    max_steps : int
        Episode length (default: 500).
    action_penalty : float
        Penalty scale for action magnitude (default: 0.01).
    **kwargs : Any
        Additional keyword arguments forwarded to ``ClankerGymnasiumEnv``.
    """
    if object_indices is None:
        object_indices = [3, 4, 5]
    if goal_indices is None:
        goal_indices = [6, 7, 8]
    if ee_indices is None:
        ee_indices = [0, 1, 2]

    reward_fn = CompositeReward([
        (DistanceReward(object_indices, goal_indices), 1.0),
        (DistanceReward(ee_indices, object_indices), 0.5),
        (ActionPenaltyReward(scale=action_penalty), 1.0),
    ])

    termination_fn = CompositeTermination([
        SuccessTermination(object_indices, goal_indices, success_threshold),
        TimeoutTermination(max_steps),
    ])

    return ClankerGymnasiumEnv(
        host=host,
        port=port,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        **kwargs,
    )
