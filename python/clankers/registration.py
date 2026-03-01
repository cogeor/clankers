"""Gymnasium environment registration for Clanker tasks.

Call ``register_envs()`` to make environments available via ``gymnasium.make()``.
Registration happens automatically when this module is imported.
"""

from __future__ import annotations

_registered = False


def register_envs() -> None:
    """Register all Clanker task presets with gymnasium."""
    global _registered  # noqa: PLW0603
    if _registered:
        return

    try:
        import gymnasium
    except ImportError:
        return

    gymnasium.register(
        id="ClankerArmReach-v0",
        entry_point="clankers.envs.arm_reach:make_arm_reach_env",
        kwargs={},
    )

    gymnasium.register(
        id="ClankerArmPick-v0",
        entry_point="clankers.envs.arm_pick:make_arm_pick_env",
        kwargs={},
    )

    gymnasium.register(
        id="ClankerCartPole-v0",
        entry_point="clankers.gymnasium_env:make_cartpole_gymnasium_env",
        kwargs={},
    )

    _registered = True


# Auto-register on import.
register_envs()
