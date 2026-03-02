"""Tests for gymnasium environment registration."""

from __future__ import annotations

import gymnasium

import clankers.registration  # triggers register_envs()


class TestRegistration:
    """Verify that all Clanker envs are registered with gymnasium."""

    def test_arm_reach_registered(self) -> None:
        """ClankerArmReach-v0 should be in the gymnasium registry."""
        spec = gymnasium.spec("ClankerArmReach-v0")
        assert spec is not None
        assert spec.id == "ClankerArmReach-v0"

    def test_arm_pick_registered(self) -> None:
        """ClankerArmPick-v0 should be in the gymnasium registry."""
        spec = gymnasium.spec("ClankerArmPick-v0")
        assert spec is not None
        assert spec.id == "ClankerArmPick-v0"

    def test_cartpole_registered(self) -> None:
        """ClankerCartPole-v0 should be in the gymnasium registry."""
        spec = gymnasium.spec("ClankerCartPole-v0")
        assert spec is not None
        assert spec.id == "ClankerCartPole-v0"

    def test_idempotent_registration(self) -> None:
        """Calling register_envs() twice should not raise."""
        clankers.registration.register_envs()
        clankers.registration.register_envs()


class TestFactoryFunctions:
    """Verify that factory functions create properly configured envs."""

    def test_make_arm_reach_factory(self) -> None:
        """make_arm_reach_env produces a ClankerGymnasiumEnv."""
        from clankers.envs.arm_reach import make_arm_reach_env

        env = make_arm_reach_env()
        # Don't connect; just verify it was created with correct reward/termination
        assert env._reward_fn is not None
        assert env._termination_fn is not None
        env.close()

    def test_make_arm_pick_factory(self) -> None:
        """make_arm_pick_env produces a ClankerGymnasiumEnv."""
        from clankers.envs.arm_pick import make_arm_pick_env

        env = make_arm_pick_env()
        assert env._reward_fn is not None
        assert env._termination_fn is not None
        env.close()
