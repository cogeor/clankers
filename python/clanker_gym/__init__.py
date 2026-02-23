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

# Optional Gymnasium integration (requires `pip install clanker-gym[sb3]`).
try:
    from clanker_gym.gymnasium_env import ClankerGymnasiumEnv

    __all__.append("ClankerGymnasiumEnv")
except ImportError:
    pass

__version__ = "0.1.0"
