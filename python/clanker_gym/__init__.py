"""Clanker Gym: Python client for Clankers simulation training."""

from clanker_gym.spaces import Box, Discrete
from clanker_gym.client import GymClient
from clanker_gym.env import ClankerEnv
from clanker_gym.vec_env import ClankerVecEnv
from clanker_gym.rewards import (
    RewardFunction,
    DistanceReward,
    SparseReward,
    ActionPenaltyReward,
    CompositeReward,
)
from clanker_gym.terminations import (
    TerminationFn,
    SuccessTermination,
    TimeoutTermination,
    FailureTermination,
    CompositeTermination,
)

__all__ = [
    "Box",
    "Discrete",
    "GymClient",
    "ClankerEnv",
    "ClankerVecEnv",
    # Rewards
    "RewardFunction",
    "DistanceReward",
    "SparseReward",
    "ActionPenaltyReward",
    "CompositeReward",
    # Terminations
    "TerminationFn",
    "SuccessTermination",
    "TimeoutTermination",
    "FailureTermination",
    "CompositeTermination",
]

__version__ = "0.1.0"
