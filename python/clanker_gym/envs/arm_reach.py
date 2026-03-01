"""Arm reach task preset.

Creates a ClankerGymnasiumEnv configured for a 6-DOF arm reaching task
with dense distance reward and success termination.
"""

from __future__ import annotations

from typing import Any

from clanker_gym.gymnasium_env import ClankerGymnasiumEnv
from clanker_gym.rewards import ActionPenaltyReward, CompositeReward, DistanceReward
from clanker_gym.terminations import CompositeTermination, SuccessTermination, TimeoutTermination


def make_arm_reach_env(
    host: str = "127.0.0.1",
    port: int = 9878,
    ee_indices: list[int] | None = None,
    goal_indices: list[int] | None = None,
    success_threshold: float = 0.05,
    max_steps: int = 200,
    action_penalty: float = 0.01,
    **kwargs: Any,
) -> ClankerGymnasiumEnv:
    """Create an arm reaching environment.

    The reward is a composite of:
    - Dense distance reward between end-effector and goal (weight 1.0)
    - Action penalty (weight action_penalty)

    Terminates on:
    - Success: EE within success_threshold of goal
    - Timeout: max_steps exceeded

    Parameters
    ----------
    host : str
        Server address (default: 127.0.0.1).
    port : int
        Server port (default: 9878).
    ee_indices : list[int] | None
        Observation indices for end-effector position (default: [0, 1, 2]).
    goal_indices : list[int] | None
        Observation indices for goal position (default: [6, 7, 8]).
    success_threshold : float
        Distance for success (default: 0.05m).
    max_steps : int
        Episode length (default: 200).
    action_penalty : float
        Penalty scale for action magnitude (default: 0.01).
    **kwargs : Any
        Additional keyword arguments forwarded to ``ClankerGymnasiumEnv``.
    """
    if ee_indices is None:
        ee_indices = [0, 1, 2]
    if goal_indices is None:
        goal_indices = [6, 7, 8]

    reward_fn = CompositeReward([
        (DistanceReward(ee_indices, goal_indices), 1.0),
        (ActionPenaltyReward(scale=action_penalty), 1.0),
    ])

    termination_fn = CompositeTermination([
        SuccessTermination(ee_indices, goal_indices, success_threshold),
        TimeoutTermination(max_steps),
    ])

    return ClankerGymnasiumEnv(
        host=host,
        port=port,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        **kwargs,
    )
