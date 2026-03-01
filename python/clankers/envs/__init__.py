"""Task preset environments for Clankers.

Each module provides a factory function that returns a pre-configured
:class:`~clankers.gymnasium_env.ClankerGymnasiumEnv`.
"""

from clankers.envs.arm_reach import make_arm_reach_env
from clankers.envs.arm_pick import make_arm_pick_env

__all__ = ["make_arm_reach_env", "make_arm_pick_env"]
