"""Clanker Gym: Python client for Clankers simulation training."""

from clanker_gym.client import GymClient
from clanker_gym.env import ClankerEnv
from clanker_gym.rewards import (
    ActionPenaltyReward,
    CompositeReward,
    ConstantReward,
    DistanceReward,
    RewardFunction,
    SparseReward,
)
from clanker_gym.spaces import Box, Discrete
from clanker_gym.terminations import (
    BoundsTermination,
    CompositeTermination,
    FailureTermination,
    SuccessTermination,
    TerminationFn,
    TimeoutTermination,
    cartpole_termination,
)
from clanker_gym.vec_env import ClankerVecEnv

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
    "ConstantReward",
    "ActionPenaltyReward",
    "CompositeReward",
    # Terminations
    "TerminationFn",
    "SuccessTermination",
    "TimeoutTermination",
    "FailureTermination",
    "BoundsTermination",
    "CompositeTermination",
    "cartpole_termination",
]

# Optional Gymnasium integration (requires `pip install clanker-gym[sb3]`).
try:
    from clanker_gym.gymnasium_env import (  # noqa: F401
        ClankerGymnasiumEnv,
        make_cartpole_gymnasium_env,
    )

    __all__.extend(["ClankerGymnasiumEnv", "make_cartpole_gymnasium_env"])
except ImportError:
    pass

# Optional SB3 VecEnv integration (requires `pip install clanker-gym[sb3]`).
try:
    from clanker_gym.sb3_vec_env import (  # noqa: F401
        ClankerSB3VecEnv,
        make_cartpole_sb3_vec_env,
    )

    __all__.extend(["ClankerSB3VecEnv", "make_cartpole_sb3_vec_env"])
except ImportError:
    pass

__version__ = "0.1.0"
