"""Task preset environments for Clankers.

Each module provides a factory function that returns a pre-configured
:class:`~clanker_gym.gymnasium_env.ClankerGymnasiumEnv`.
"""

from clanker_gym.envs.arm_reach import make_arm_reach_env
from clanker_gym.envs.arm_pick import make_arm_pick_env

__all__ = ["make_arm_reach_env", "make_arm_pick_env"]
