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
from clanker_gym.spaces import Box, Dict, Discrete
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
    "Dict",
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

# Optional GoalEnv wrapper for HER (requires `pip install clanker-gym[sb3]`).
try:
    from clanker_gym.goal_env import (  # noqa: F401
        ClankerGoalEnv,
        DenseGoalReward,
        GoalRewardFn,
        SparseGoalReward,
    )

    __all__.extend(
        ["ClankerGoalEnv", "GoalRewardFn", "SparseGoalReward", "DenseGoalReward"]
    )
except ImportError:
    pass

# Optional observation / reward wrappers (requires `pip install clanker-gym[sb3]`).
try:
    from clanker_gym.wrappers import (  # noqa: F401
        ClipReward,
        FrameStack,
        NormalizeObservation,
        NormalizeReward,
    )

    __all__.extend(["NormalizeObservation", "NormalizeReward", "FrameStack", "ClipReward"])
except ImportError:
    pass

# Optional evaluation utilities (requires `pip install clanker-gym[sb3]`).
try:
    from clanker_gym.evaluation import (  # noqa: F401
        EvalResult,
        evaluate_policy,
    )

    __all__.extend(["EvalResult", "evaluate_policy"])
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

# Optional MCAP episode loading (requires `pip install clanker-gym[mcap]`).
try:
    from clanker_gym.mcap_loader import (  # noqa: F401
        EpisodeDataset,
        McapEpisodeLoader,
    )

    __all__.extend(["McapEpisodeLoader", "EpisodeDataset"])
except ImportError:
    pass

# Task preset environments (requires `pip install clanker-gym[sb3]`).
try:
    from clanker_gym.envs.arm_reach import make_arm_reach_env  # noqa: F401
    from clanker_gym.envs.arm_pick import make_arm_pick_env  # noqa: F401

    __all__.extend(["make_arm_reach_env", "make_arm_pick_env"])
except ImportError:
    pass

# Auto-register gymnasium environments.
try:
    import clanker_gym.registration  # noqa: F401
except ImportError:
    pass

__version__ = "0.1.0"
